package com.k2fsa.sherpa.common.config

object ModelPaths {
    const val ASR_MODEL = "GigaAM-v3-sherpa-onnx/gigaam_v3_e2e_ctc_int8.onnx"
    const val ASR_TOKENS = "GigaAM-v3-sherpa-onnx/gigaam_v3_e2e_ctc_tokens.txt"
    const val DIARIZATION_SEGMENTATION = "sherpa-onnx-pyannote-segmentation-3-0/model.int8.onnx"
    const val DIARIZATION_EMBEDDING = "3D-Speaker/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
    const val DIARIZATION_VAD = "Vad/silero_vad.onnx"
}