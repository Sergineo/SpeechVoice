package com.k2fsa.sherpa.common.model

data class Utterance(
    val speakerId: Int,
    val startSec: Double,
    val endSec: Double,
    val text: String
)
