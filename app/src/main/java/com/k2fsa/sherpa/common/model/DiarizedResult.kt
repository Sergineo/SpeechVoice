package com.k2fsa.sherpa.common.model

data class DiarizedResult(
    val utterances: List<Utterance>,
    val processingTimeMs: Long
)
