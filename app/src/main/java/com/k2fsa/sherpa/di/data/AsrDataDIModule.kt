package com.k2fsa.sherpa.di.data

import com.k2fsa.sherpa.data.local.asr.config.OfflineDiarizationConfigImpl
import com.k2fsa.sherpa.data.local.asr.config.OfflineRecognizerConfigImpl
import com.k2fsa.sherpa.data.local.asr.config.OfflineVadConfigImpl
import com.k2fsa.sherpa.data.local.asr.manager.AudioRecognitionManager
import com.k2fsa.sherpa.data.local.files.FileDecoder
import com.k2fsa.sherpa.data.local.files.FileManager
import org.koin.android.ext.koin.androidContext
import org.koin.dsl.module

private const val Tag: String = "AudioRecognition"
private const val UseVadFiltering: Boolean = true
private const val SampleRate: Int = 16000

val DataDiModule = module {
    single { FileDecoder(context = androidContext()) }
    single { FileManager(tag = Tag, context = androidContext()) }

    single { OfflineDiarizationConfigImpl(tag = Tag, fIleManager = get<FileManager>()) }
    single { OfflineRecognizerConfigImpl(tag = Tag, fileManager = get<FileManager>()) }
    single { OfflineVadConfigImpl(tag = Tag, fileManager = get<FileManager>(), sampleRate = SampleRate) }

    single {
        AudioRecognitionManager(
            tag = Tag,
            decoder = get<FileDecoder>(),
            useVadFiltering = UseVadFiltering,
            diarizationConfig = get<OfflineDiarizationConfigImpl>(),
            recognizerConfig = get<OfflineRecognizerConfigImpl>(),
            vadConfig = get<OfflineVadConfigImpl>()
        )
    }
}