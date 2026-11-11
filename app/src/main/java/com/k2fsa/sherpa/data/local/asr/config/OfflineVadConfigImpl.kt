package com.k2fsa.sherpa.data.local.asr.config

import android.util.Log
import com.k2fsa.sherpa.data.local.files.FileManager
import com.k2fsa.sherpa.onnx.SileroVadModelConfig
import com.k2fsa.sherpa.onnx.Vad
import com.k2fsa.sherpa.onnx.VadModelConfig

class OfflineVadConfigImpl(
    private val tag: String,
    private val fileManager: FileManager,
    private val sampleRate: Int
) {
    fun initVad(): Vad?{
        return try {
            val vadPath = fileManager.copyAssetToFilesDir("vad/silero_vad.onnx") ?: return null

            val config = VadModelConfig(
                sileroVadModelConfig = SileroVadModelConfig(
                    model = vadPath,
                    threshold = 0.3f,
                    minSilenceDuration = 0.25f,
                    minSpeechDuration = 0.4f,
                    windowSize = 512
                ),
                sampleRate = sampleRate,
                numThreads = 1,
                provider = "cpu"
            )
            Vad(null, config)
        }catch (e: Exception){
            Log.e(tag, "Ошибка инициализации Диаризации", e)
            null
        }

    }
}