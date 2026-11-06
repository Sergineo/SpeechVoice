package com.k2fsa.sherpa.common.manager

import android.content.Context
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.net.Uri
import java.nio.ByteBuffer
import java.nio.ByteOrder

class FileDecoder(private val context: Context) {
    fun readFile(filePath: String): FloatArray {
        val uri = Uri.parse(filePath)
        val extractor = MediaExtractor()

        context.contentResolver.openFileDescriptor(uri, "r")?.use { pfd ->
            extractor.setDataSource(pfd.fileDescriptor)
        } ?: throw IllegalArgumentException("Не удалось открыть файл: $filePath")

        var audioTrackIndex = -1
        var inputFormat: MediaFormat? = null
        for (i in 0 until extractor.trackCount) {
            val format = extractor.getTrackFormat(i)
            val mime = format.getString(MediaFormat.KEY_MIME) ?: ""
            if (mime.startsWith("audio/")) {
                audioTrackIndex = i
                inputFormat = format
                break
            }
        }

        if (audioTrackIndex == -1 || inputFormat == null) {
            extractor.release()
            throw IllegalArgumentException("Аудиодорожка не найдена")
        }

        extractor.selectTrack(audioTrackIndex)
        val mime = inputFormat.getString(MediaFormat.KEY_MIME)!!
        val decoder = MediaCodec.createDecoderByType(mime)
        decoder.configure(inputFormat, null, null, 0)
        decoder.start()

        val pcmBytes = mutableListOf<Byte>()
        val bufferInfo = MediaCodec.BufferInfo()
        var isExtractorEOS = false
        var isDecoderEOS = false

        while (!isDecoderEOS) {
            if (!isExtractorEOS) {
                val inBufferId = decoder.dequeueInputBuffer(10000)
                if (inBufferId >= 0) {
                    val inputBuffer = decoder.getInputBuffer(inBufferId)!!
                    val sampleSize = extractor.readSampleData(inputBuffer, 0)
                    if (sampleSize < 0) {
                        decoder.queueInputBuffer(inBufferId, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                        isExtractorEOS = true
                    } else {
                        decoder.queueInputBuffer(inBufferId, 0, sampleSize, extractor.sampleTime, 0)
                        extractor.advance()
                    }
                }
            }

            val outBufferId = decoder.dequeueOutputBuffer(bufferInfo, 10000)
            if (outBufferId >= 0) {
                val outputBuffer = decoder.getOutputBuffer(outBufferId)!!
                val chunk = ByteArray(bufferInfo.size)
                outputBuffer.get(chunk)
                outputBuffer.clear()
                pcmBytes.addAll(chunk.toList())
                decoder.releaseOutputBuffer(outBufferId, false)
                if ((bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0) {
                    isDecoderEOS = true
                }
            }
        }

        decoder.stop()
        decoder.release()
        extractor.release()

        val outputSampleRate = inputFormat.getInteger(MediaFormat.KEY_SAMPLE_RATE)
        val outputChannels = inputFormat.getInteger(MediaFormat.KEY_CHANNEL_COUNT)

        val rawByteArray = pcmBytes.toByteArray()
        val shortBuffer = ByteBuffer.wrap(rawByteArray).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer()
        var floatArray = FloatArray(shortBuffer.remaining())
        for (i in floatArray.indices) {
            floatArray[i] = shortBuffer.get(i) / 32768.0f
        }

        if (outputChannels > 1) floatArray = convertStereoToMono(floatArray, outputChannels)
        if (outputSampleRate != 16000) floatArray = resampleTo16k(floatArray, outputSampleRate)

        return floatArray
    }

    private fun convertStereoToMono(stereoSamples: FloatArray, channels: Int): FloatArray {
        val monoSamples = FloatArray(stereoSamples.size / channels)
        for (i in monoSamples.indices) {
            var sum = 0f
            for (c in 0 until channels) sum += stereoSamples[i * channels + c]
            monoSamples[i] = sum / channels
        }
        return monoSamples
    }

    private fun resampleTo16k(samples: FloatArray, originalSampleRate: Int): FloatArray {
        val ratio = originalSampleRate.toDouble() / 16000.0
        val newLength = (samples.size / ratio).toInt()
        val resampled = FloatArray(newLength)
        for (i in 0 until newLength) {
            val srcIndex = (i * ratio).toInt()
            if (srcIndex < samples.size) resampled[i] = samples[srcIndex]
        }
        return resampled
    }
}