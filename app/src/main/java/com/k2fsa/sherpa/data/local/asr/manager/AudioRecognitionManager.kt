package com.k2fsa.sherpa.data.local.asr.manager

import android.util.Log
import com.k2fsa.sherpa.common.model.DiarizedResult
import com.k2fsa.sherpa.common.model.Utterance
import com.k2fsa.sherpa.data.local.asr.config.OfflineDiarizationConfigImpl
import com.k2fsa.sherpa.data.local.asr.config.OfflineRecognizerConfigImpl
import com.k2fsa.sherpa.data.local.asr.config.OfflineVadConfigImpl
import com.k2fsa.sherpa.data.local.files.FileDecoder

class AudioRecognitionManager(
    private val tag: String,
    private val decoder: FileDecoder,
    private val useVadFiltering: Boolean,
    private val diarizationConfig: OfflineDiarizationConfigImpl,
    private val recognizerConfig: OfflineRecognizerConfigImpl,
    private val vadConfig: OfflineVadConfigImpl
) {
    fun processAudioFile(filePath: String): DiarizedResult? {
        val startTime = System.currentTimeMillis()

        return try {
            // 1. Декодирование
            Log.d(tag, "Декодирование файла: $filePath")
            val samples = decoder.readFile(filePath)
            Log.d(tag, "Аудио декодировано. Сэмплов: ${samples.size}")

            // 1.5. VAD-ФИЛЬТРАЦИЯ (Опционально)
            val processedSamples = if (useVadFiltering) {
                Log.d(tag, "--- Запуск VAD ---")
                val trimmed = trimSilenceWithVad(samples, 16000)
                if (trimmed != null) {
                    Log.d(tag, "После VAD: ${trimmed.size} сэмплов (было ${samples.size})")
                    trimmed
                } else {
                    Log.w(tag, "VAD вернул null, используем оригинал")
                    samples
                }
            } else {
                Log.d(tag, "--- VAD ОТКЛЮЧЕН (используем оригинал) ---")
                samples
            }

            if (processedSamples.size < 16000 * 0.5) {
                Log.e(tag, "Аудио слишком короткое")
                return null
            }

            // 2. ДИАРИЗАЦИЯ (Создаем, используем, убиваем)
            Log.d(tag, "--- Запуск Диаризации ---")
            val diarizer = diarizationConfig.initDiarizer()
            if (diarizer == null) {
                Log.e(tag, "Не удалось инициализировать диаризатор")
                return null
            }

            val speakerSegments = try {
                diarizer.process(processedSamples)
            } finally {
                diarizer.release() // Обязательно освобождаем!
            }
            Log.d(tag, "Диаризация завершена. Найдено реплик: ${speakerSegments.size}")

            // 3. ASR (Создаем, используем, убиваем)
            Log.d(tag, "--- Запуск ASR ---")
            val recognizer = recognizerConfig.initRecognizer()
            if (recognizer == null) {
                Log.e(tag, "Не удалось инициализировать ASR")
                return null
            }

            val utterances = mutableListOf<Utterance>()
            val sampleRate = 16000

            try {
                for (segment in speakerSegments) {
                    val startSample = (segment.start * sampleRate).toInt().coerceIn(0, processedSamples.size)
                    val endSample = (segment.end * sampleRate).toInt().coerceIn(startSample, processedSamples.size)
                    val chunk = processedSamples.copyOfRange(startSample, endSample)

                    if (chunk.size < sampleRate * 0.3) continue

                    val stream = recognizer.createStream()
                    stream.acceptWaveform(chunk, sampleRate)
                    recognizer.decode(stream)
                    val text = recognizer.getResult(stream).text.trim()
                    stream.release()

                    if (text.isNotBlank()) {
                        utterances.plusAssign(
                            Utterance(
                                speakerId = segment.speaker,
                                startSec = segment.start.toDouble(),
                                endSec = segment.end.toDouble(),
                                text = text
                            )
                        )
                    }
                }
            } finally {
                recognizer.release() // Обязательно освобождаем!
            }

            Log.d(tag, "ASR завершен. Реплик: ${utterances.size}")

            val timeMs = System.currentTimeMillis() - startTime
            Log.i(tag, "Успешно обработано за $timeMs мс")

            DiarizedResult(utterances, timeMs)

        } catch (e: OutOfMemoryError) {
            Log.e(tag, "КРИТИЧЕСКАЯ ОШИБКА: OOM", e)
            null
        } catch (e: Exception) {
            Log.e(tag, "Ошибка обработки", e)
            null
        }
    }

    private fun trimSilenceWithVad(samples: FloatArray, sampleRate: Int): FloatArray? {
        // Создаем локальный VAD
        val vad = vadConfig.initVad() ?: return samples

        try {
            val windowSize = 512
            var firstSpeechSample = -1
            var lastSpeechSample = -1

            var i = 0
            while (i < samples.size) {
                val end = minOf(i + windowSize, samples.size)
                val chunk = samples.copyOfRange(i, end)
                val processedChunk = if (chunk.size < windowSize) {
                    FloatArray(windowSize).also { chunk.copyInto(it) }
                } else {
                    chunk
                }

                vad.acceptWaveform(processedChunk)

                while (!vad.empty()) {
                    if (firstSpeechSample == -1) firstSpeechSample = i
                    lastSpeechSample = end
                    vad.pop()
                }
                i = end
            }
            vad.flush()

            while (!vad.empty()) {
                if (firstSpeechSample == -1) firstSpeechSample = samples.size - windowSize
                lastSpeechSample = samples.size
                vad.pop()
            }

            if (firstSpeechSample == -1 || lastSpeechSample == -1) return samples

            // Даем запас 1.0 секунду по краям
            val padding = (sampleRate * 1.0).toInt()
            val start = (firstSpeechSample - padding).coerceAtLeast(0)
            val end = (lastSpeechSample + padding).coerceAtMost(samples.size)

            return samples.copyOfRange(start, end)

        } catch (e: Exception) {
            Log.e(tag, "Ошибка VAD: ${e.message}", e)
            return samples
        } finally {
            // Всегда освобождаем локальный VAD
            vad.release()
        }
    }
}