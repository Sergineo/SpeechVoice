package com.k2fsa.sherpa.common.manager

import android.content.Context
import android.util.Log
import com.k2fsa.sherpa.common.config.ModelPaths
import com.k2fsa.sherpa.common.model.DiarizedResult
import com.k2fsa.sherpa.common.model.Utterance
import com.k2fsa.sherpa.onnx.FastClusteringConfig
import com.k2fsa.sherpa.onnx.FeatureConfig
import com.k2fsa.sherpa.onnx.OfflineModelConfig
import com.k2fsa.sherpa.onnx.OfflineRecognizer
import com.k2fsa.sherpa.onnx.OfflineRecognizerConfig
import com.k2fsa.sherpa.onnx.OfflineSpeakerDiarization
import com.k2fsa.sherpa.onnx.OfflineSpeakerDiarizationConfig
import com.k2fsa.sherpa.onnx.OfflineSpeakerSegmentationModelConfig
import com.k2fsa.sherpa.onnx.OfflineSpeakerSegmentationPyannoteModelConfig
import com.k2fsa.sherpa.onnx.SpeakerEmbeddingExtractorConfig
import com.k2fsa.sherpa.onnx.SileroVadModelConfig
import com.k2fsa.sherpa.onnx.Vad
import com.k2fsa.sherpa.onnx.VadModelConfig
import com.k2fsa.sherpa.onnx.OfflineNemoEncDecCtcModelConfig
import java.io.File

class AudioRecognitionManager(private val context: Context) {

    private val tag = "AudioRecognition"

    fun processAudioFile(filePath: String): DiarizedResult? {
        val startTime = System.currentTimeMillis()

        return try {
            // 1. Декодируем файл
            Log.d(tag, "Декодирование файла: $filePath")
            val decoder = FileDecoder(context)
            val samples = decoder.readFile(filePath)
            Log.d(tag, "Аудио декодировано. Сэмплов: ${samples.size}")

            // 1.5. VAD-ФИЛЬТРАЦИЯ (Обрезаем тишину по краям)
            Log.d(tag, "--- Запуск VAD ---")
            val trimmedSamples = trimSilenceWithVad(samples, 16000)
            Log.d(tag, "После VAD: ${trimmedSamples.size} сэмплов (было ${samples.size})")

            // Если после обрезки осталось слишком мало аудио (меньше 0.5 сек), возвращаем null
            if (trimmedSamples.size < 16000 * 0.5) {
                Log.e(tag, "Аудио слишком короткое после VAD-фильтрации")
                return null
            }

            // 2. ЭТАП ДИАРИЗАЦИИ
            Log.d(tag, "--- Запуск Диаризации ---")
            val diarizer = initDiarizer()
            if (diarizer == null) {
                Log.e(tag, "Не удалось инициализировать диаризатор")
                return null
            }

            // ВАЖНО: Передаем обрезанное аудио (trimmedSamples)
            val speakerSegments = diarizer.process(trimmedSamples)
            diarizer.release()
            Log.d(tag, "Диаризация завершена. Найдено реплик: ${speakerSegments.size}")

            // 3. ЭТАП ASR
            Log.d(tag, "--- Запуск ASR ---")
            val recognizer = initRecognizer()
            if (recognizer == null) {
                Log.e(tag, "Не удалось инициализировать ASR")
                return null
            }

            val utterances = mutableListOf<Utterance>()
            val sampleRate = 16000

            for (segment in speakerSegments) {
                // Нарезаем аудио под конкретного спикера (из обрезанного массива)
                val startSample = (segment.start * sampleRate).toInt().coerceIn(0, trimmedSamples.size)
                val endSample = (segment.end * sampleRate).toInt().coerceIn(startSample, trimmedSamples.size)

                val chunk = trimmedSamples.copyOfRange(startSample, endSample)

                // Пропускаем слишком короткие куски
                if (chunk.size < sampleRate * 0.3) continue

                // Распознаем
                val stream = recognizer.createStream()
                stream.acceptWaveform(chunk, sampleRate)
                recognizer.decode(stream)
                val text = recognizer.getResult(stream).text
                stream.release()

                if (text.isNotBlank()) {
                    utterances += Utterance(
                        speakerId = segment.speaker,
                        startSec = segment.start.toDouble(),
                        endSec = segment.end.toDouble(),
                        text = text.trim()
                    )
                    Log.d(tag, "Спикер ${segment.speaker}: $text")
                }
            }

            recognizer.release()
            Log.d(tag, "ASR завершен.")

            val timeMs = System.currentTimeMillis() - startTime
            Log.i(tag, "Успешно обработано за $timeMs мс")

            DiarizedResult(utterances, timeMs)

        } catch (e: OutOfMemoryError) {
            Log.e(tag, "КРИТИЧЕСКАЯ ОШИБКА: Не хватило оперативной памяти (OOM).", e)
            null
        } catch (e: Exception) {
            Log.e(tag, "Ошибка во время обработки аудио", e)
            null
        }
    }

    // =========================================================================
    // МЕТОД VAD-ФИЛЬТРАЦИИ (Обрезка тишины по краям)
    // =========================================================================
    private fun trimSilenceWithVad(samples: FloatArray, sampleRate: Int): FloatArray {
        val vadPath = copyAssetToFilesDir(ModelPaths.DIARIZATION_VAD) ?: return samples

        try {
            val config = VadModelConfig(
                sileroVadModelConfig = SileroVadModelConfig(
                    model = vadPath,
                    threshold = 0.5f,
                    minSilenceDuration = 0.5f,
                    minSpeechDuration = 0.25f,
                    windowSize = 512 // Обязательно 512 для Silero VAD на 16кГц
                ),
                sampleRate = sampleRate,
                numThreads = 1,
                provider = "cpu"
            )

            val vad = Vad(null, config)
            val windowSize = 512
            var firstSpeechSample = -1
            var lastSpeechSample = -1

            var i = 0
            while (i < samples.size) {
                val end = minOf(i + windowSize, samples.size)
                val chunk = samples.copyOfRange(i, end)

                // VAD требует ровно 512 сэмплов, дополняем нулями, если кусок меньше
                val processedChunk = if (chunk.size < windowSize) {
                    FloatArray(windowSize).also { chunk.copyInto(it) }
                } else {
                    chunk
                }

                vad.acceptWaveform(processedChunk)

                // Если VAD нашел завершенный сегмент речи
                while (!vad.empty()) {
                    if (firstSpeechSample == -1) {
                        firstSpeechSample = i // Запоминаем начало первой реплики
                    }
                    lastSpeechSample = end // Запоминаем конец последней реплики
                    vad.pop()
                }
                i = end
            }
            vad.flush()

            // Забираем остатки после flush
            while (!vad.empty()) {
                if (firstSpeechSample == -1) firstSpeechSample = samples.size - windowSize
                lastSpeechSample = samples.size
                vad.pop()
            }
            vad.release()

            // Если речь не найдена, возвращаем оригинал
            if (firstSpeechSample == -1 || lastSpeechSample == -1) {
                Log.d(tag, "VAD: речь не обнаружена, возвращаем оригинал")
                return samples
            }

            // Добавляем запас (0.5 секунды) по краям
            val padding = sampleRate / 2
            val start = (firstSpeechSample - padding).coerceAtLeast(0)
            val end = (lastSpeechSample + padding).coerceAtMost(samples.size)

            Log.d(tag, "VAD: обрезали тишину. Было ${samples.size} сэмплов, стало ${end - start}")
            return samples.copyOfRange(start, end)

        } catch (e: Exception) {
            Log.e(tag, "Ошибка VAD-фильтрации: ${e.message}", e)
            return samples // В случае ошибки возвращаем оригинал, чтобы не ломать пайплайн
        }
    }

    // =========================================================================
    // ИНИЦИАЛИЗАЦИЯ МОДЕЛЕЙ
    // =========================================================================

    private fun initDiarizer(): OfflineSpeakerDiarization? {
        return try {
            val segPath = copyAssetToFilesDir(ModelPaths.DIARIZATION_SEGMENTATION) ?: return null
            val embPath = copyAssetToFilesDir(ModelPaths.DIARIZATION_EMBEDDING) ?: return null

            val config = OfflineSpeakerDiarizationConfig(
                segmentation = OfflineSpeakerSegmentationModelConfig(
                    pyannote = OfflineSpeakerSegmentationPyannoteModelConfig(
                        model = segPath
                    )
                ),
                embedding = SpeakerEmbeddingExtractorConfig(
                    model = embPath
                ),
                clustering = FastClusteringConfig(numClusters = 2, threshold = 0.5f),
                minDurationOn = 0.5f,
                minDurationOff = 0.8f
            )
            OfflineSpeakerDiarization(null, config)
        } catch (e: Exception) {
            Log.e(tag, "Ошибка инициализации Диаризации", e)
            null
        }
    }

    private fun initRecognizer(): OfflineRecognizer? {
        return try {
            val asrModel = copyAssetToFilesDir(ModelPaths.ASR_MODEL) ?: return null
            val asrTokens = copyAssetToFilesDir(ModelPaths.ASR_TOKENS) ?: return null

            val config = OfflineRecognizerConfig(
                featConfig = FeatureConfig(sampleRate = 16000, featureDim = 64), // 64 для GigaAM!
                modelConfig = OfflineModelConfig(
                    nemo = OfflineNemoEncDecCtcModelConfig(
                        model = asrModel
                    ),
                    tokens = asrTokens,
                    numThreads = 1,
                    provider = "cpu"
                )
            )
            OfflineRecognizer(null, config)
        } catch (e: Exception) {
            Log.e(tag, "Ошибка инициализации ASR", e)
            null
        }
    }

    // =========================================================================
    // КОПИРОВАНИЕ ИЗ ASSETS
    // =========================================================================
    private fun copyAssetToFilesDir(assetPath: String): String? {
        val outFile = File(context.filesDir, assetPath)

        if (outFile.exists()) {
            return outFile.absolutePath
        }

        try {
            outFile.parentFile?.let { parentDir ->
                if (!parentDir.exists()) {
                    parentDir.mkdirs()
                }
            }

            context.assets.open(assetPath).use { input ->
                outFile.outputStream().use { output ->
                    input.copyTo(output)
                }
            }

            Log.d(tag, "Успешно скопирован asset: $assetPath -> ${outFile.absolutePath}")
            return outFile.absolutePath

        } catch (e: Exception) {
            Log.e(tag, "Ошибка при копировании $assetPath: ${e.message}")
            return null
        }
    }
}
