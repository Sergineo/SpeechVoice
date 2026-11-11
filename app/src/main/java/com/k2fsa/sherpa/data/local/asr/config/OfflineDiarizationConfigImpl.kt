package com.k2fsa.sherpa.data.local.asr.config

import android.util.Log
import com.k2fsa.sherpa.common.config.ModelPaths
import com.k2fsa.sherpa.data.local.files.FileManager
import com.k2fsa.sherpa.onnx.FastClusteringConfig
import com.k2fsa.sherpa.onnx.OfflineSpeakerDiarization
import com.k2fsa.sherpa.onnx.OfflineSpeakerDiarizationConfig
import com.k2fsa.sherpa.onnx.OfflineSpeakerSegmentationModelConfig
import com.k2fsa.sherpa.onnx.OfflineSpeakerSegmentationPyannoteModelConfig
import com.k2fsa.sherpa.onnx.SpeakerEmbeddingExtractorConfig

class OfflineDiarizationConfigImpl(
    private val tag: String,
    private val fIleManager: FileManager
) {
    fun initDiarizer(): OfflineSpeakerDiarization? {
        return try {
            val segPath = fIleManager.copyAssetToFilesDir(ModelPaths.DIARIZATION_SEGMENTATION) ?: return null
            val embPath = fIleManager.copyAssetToFilesDir(ModelPaths.DIARIZATION_EMBEDDING) ?: return null

            val config = OfflineSpeakerDiarizationConfig(
                segmentation = OfflineSpeakerSegmentationModelConfig(
                    pyannote = OfflineSpeakerSegmentationPyannoteModelConfig(model = segPath)
                ),
                embedding = SpeakerEmbeddingExtractorConfig(model = embPath),
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
}