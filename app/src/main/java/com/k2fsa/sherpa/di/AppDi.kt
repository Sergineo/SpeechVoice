package com.k2fsa.sherpa.di

import android.app.Application
import com.k2fsa.sherpa.di.data.DataDiModule
import org.koin.android.ext.koin.androidContext
import org.koin.core.context.startKoin

class AppDi: Application(){
    override fun onCreate() {
        super.onCreate()
        startKoin {
            androidContext(this@AppDi)
            modules(DataDiModule)
        }
    }
}