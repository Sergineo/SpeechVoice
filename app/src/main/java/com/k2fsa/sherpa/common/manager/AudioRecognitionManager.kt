package com.k2fsa.sherpa.common.manager

import android.content.Context
import android.util.Log
import com.k2fsa.sherpa.common.config.ModelPaths
import com.k2fsa.sherpa.common.model.DiarizedResult
import com.k2fsa.sherpa.common.model.Utterance
import com.k2fsa.sherpa.onnx.FastClusteringConfig
import com.k2fsa.sherpa.onnx.FeatureConfig
import com.k2fsa.sherpa.onnx.OfflineModelConfig
import com.k2fsa.sherpa.onnx.OfflineNemoEncDecCtcModelConfig
import com.k2fsa.sherpa.onnx.OfflineRecognizer
import com.k2fsa.sherpa.onnx.OfflineRecognizerConfig
import com.k2fsa.sherpa.onnx.OfflineSpeakerDiarization
import com.k2fsa.sherpa.onnx.OfflineSpeakerDiarizationConfig
import com.k2fsa.sherpa.onnx.OfflineSpeakerSegmentationModelConfig
import com.k2fsa.sherpa.onnx.OfflineSpeakerSegmentationPyannoteModelConfig
import com.k2fsa.sherpa.onnx.SpeakerEmbeddingExtractorConfig
import java.io.File

class AudioRecognitionManager(private val context: Context) {

    private val tag = "AudioRecognition"

    fun processAudioFile(filePath: String): DiarizedResult? {
        val startTime = System.currentTimeMillis()

        return try {
            // 1. Декодируем файл (это быстро и не требует много памяти)
            Log.d(tag, "Декодирование файла: $filePath")
            val decoder = FileDecoder(context)
            val samples = decoder.readFile(filePath)
            Log.d(tag, "Аудио декодировано. Сэмплов: ${samples.size}")

            // 2. ЭТАП ДИАРИЗАЦИИ (Загружаем -> Обрабатываем -> Освобождаем)
            Log.d(tag, "--- Запуск Диаризации ---")
            val diarizer = initDiarizer()
            if (diarizer == null) {
                Log.e(tag, "Не удалось инициализировать диаризатор")
                return null
            }

            val speakerSegments = diarizer.process(samples)
            diarizer.release() // ОСВОБОЖДАЕМ ПАМЯТЬ СРАЗУ ПОСЛЕ ИСПОЛЬЗОВАНИЯ
            Log.d(tag, "Диаризация завершена. Найдено реплик: ${speakerSegments.size}")

            // 3. ЭТАП ASR (Загружаем -> Обрабатываем каждый кусок -> Освобождаем)
            Log.d(tag, "--- Запуск ASR ---")
            val recognizer = initRecognizer()
            if (recognizer == null) {
                Log.e(tag, "Не удалось инициализировать ASR")
                return null
            }

            val utterances = mutableListOf<Utterance>()
            val sampleRate = 16000

            for (segment in speakerSegments) {
                // Переводим время в индексы массива
                val startSample = (segment.start * sampleRate).toInt().coerceIn(0, samples.size)
                val endSample = (segment.end * sampleRate).toInt().coerceIn(startSample, samples.size)

                // Нарезаем аудио под конкретного спикера
                val chunk = samples.copyOfRange(startSample, endSample)

                // Пропускаем слишком короткие куски (меньше 0.3 сек), чтобы не нагружать ASR
                if (chunk.size < sampleRate * 0.3) continue

                // Распознаем
                val stream = recognizer.createStream()
                stream.acceptWaveform(chunk, sampleRate)
                recognizer.decode(stream)
                val text = recognizer.getResult(stream).text
                stream.release()

                // Если текст не пустой, сохраняем
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

            recognizer.release() // ОСВОБОЖДАЕМ ПАМЯТЬ ASR
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

    private fun initDiarizer(): OfflineSpeakerDiarization? {
        return try {
            // Копируем файлы из assets в filesDir и получаем абсолютные пути
            val segPath = copyAssetToFilesDir(ModelPaths.DIARIZATION_SEGMENTATION) ?: return null
            val embPath = copyAssetToFilesDir(ModelPaths.DIARIZATION_EMBEDDING) ?: return null

            Log.d(tag, "Путь к сегментации: $segPath, существует: ${File(segPath).exists()}")
            Log.d(tag, "Путь к эмбеддингу: $embPath, существует: ${File(embPath).exists()}")

            val config = OfflineSpeakerDiarizationConfig(
                segmentation = OfflineSpeakerSegmentationModelConfig(
                    pyannote = OfflineSpeakerSegmentationPyannoteModelConfig(
                        model = segPath // Передаем абсолютный путь
                    )
                ),
                embedding = SpeakerEmbeddingExtractorConfig(
                    model = embPath // Передаем абсолютный путь
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
            // Для ASR тоже можно скопировать файлы, если sherpa-onnx не умеет читать их напрямую из assets.
            // Если ваш ASR работал и так — можно оставить ModelPaths.ASR_..., но для надежности лучше тоже копировать.

            val asrModel = copyAssetToFilesDir(ModelPaths.ASR_MODEL) ?: return null
            val asrTokens = copyAssetToFilesDir(ModelPaths.ASR_TOKENS) ?: return null

            val config = OfflineRecognizerConfig(
                featConfig = FeatureConfig(sampleRate = 16000, featureDim = 64),
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

//    private fun copyAssetToFilesDir(assetPath: String): String {
//        // Теперь outFile сохранит всю структуру папок (например, files/pyannote-segmentation-3-0/onnx/model.onnx)
//        val outFile = File(context.filesDir, assetPath)
//
//        // Если файл уже существует, не копируем заново
//        if (outFile.exists() && outFile.length() > 0) {
//            return outFile.absolutePath
//        }
//
//        try {
//            // Автоматически создаем все родительские папки на устройстве, если их нет
//            outFile.parentFile?.let { parentDir ->
//                if (!parentDir.exists()) {
//                    parentDir.mkdirs()
//                }
//            }
//
//            // Копируем из assets
//            context.assets.open(assetPath).use { input ->
//                outFile.outputStream().use { output ->
//                    input.copyTo(output)
//                }
//            }
//            Log.d(tag, "Успешно скопирован asset: $assetPath -> ${outFile.absolutePath}")
//        } catch (e: Exception) {
//            Log.e(tag, "Ошибка при копировании asset $assetPath: ${e.message}")
//        }
//
//        return outFile.absolutePath
//    }

    private fun copyAssetToFilesDir(assetPath: String): String? {
        val outFile = File(context.filesDir, assetPath)

        // Проверяем только на существование
        if (outFile.exists()) {
            return outFile.absolutePath
        }

        try {
            // Создаем родительские папки
            outFile.parentFile?.let { parentDir ->
                if (!parentDir.exists()) {
                    parentDir.mkdirs()
                }
            }

            // Копируем файл из assets
            context.assets.open(assetPath).use { input ->
                outFile.outputStream().use { output ->
                    input.copyTo(output)
                }
            }

            Log.d(tag, "Успешно скопирован asset: $assetPath -> ${outFile.absolutePath}")
            return outFile.absolutePath

        } catch (e: Exception) {
            Log.e(tag, "Ошибка при копировании $assetPath: ${e.message}")
            return null // Возвращаем null, чтобы приложение не падало при ошибке
        }
    }

}
