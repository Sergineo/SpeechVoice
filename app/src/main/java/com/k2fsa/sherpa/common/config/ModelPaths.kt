package com.k2fsa.sherpa.common.config

object ModelPaths {
    const val ASR_DIR = "sherpa-onnx-whisper-tiny"
    const val ASR_ENCODER = "$ASR_DIR/tiny-encoder.int8.onnx"
    const val ASR_DECODER = "$ASR_DIR/tiny-decoder.int8.onnx"
    const val ASR_TOKENS = "$ASR_DIR/tiny-tokens.txt"
    const val DIARIZATION_SEGMENTATION = "sherpa-onnx-pyannote-segmentation-3-0/model.onnx"
    const val DIARIZATION_EMBEDDING = "3D-Speaker/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
}