package com.k2fsa.sherpa.data.local.asr.config

import android.util.Log
import com.k2fsa.sherpa.common.config.ModelPaths
import com.k2fsa.sherpa.data.local.files.FileManager
import com.k2fsa.sherpa.onnx.FeatureConfig
import com.k2fsa.sherpa.onnx.OfflineModelConfig
import com.k2fsa.sherpa.onnx.OfflineNemoEncDecCtcModelConfig
import com.k2fsa.sherpa.onnx.OfflineRecognizer
import com.k2fsa.sherpa.onnx.OfflineRecognizerConfig

class OfflineRecognizerConfigImpl(
    private val tag: String,
    private val fileManager: FileManager
) {
    fun initRecognizer(): OfflineRecognizer? {
        return try {
            val asrModel = fileManager.copyAssetToFilesDir(ModelPaths.ASR_MODEL) ?: return null
            val asrTokens = fileManager.copyAssetToFilesDir(ModelPaths.ASR_TOKENS) ?: return null

            val config = OfflineRecognizerConfig(
                featConfig = FeatureConfig(sampleRate = 16000, featureDim = 64),
                modelConfig = OfflineModelConfig(
                    nemo = OfflineNemoEncDecCtcModelConfig(model = asrModel),
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
}