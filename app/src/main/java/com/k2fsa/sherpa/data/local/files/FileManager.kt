package com.k2fsa.sherpa.data.local.files

import android.content.Context
import android.util.Log
import java.io.File

class FileManager(
    private val tag: String,
    private val context: Context
) {
    fun copyAssetToFilesDir(assetPath: String): String? {
        val outFile = File(context.filesDir, assetPath)
        if (outFile.exists()) return outFile.absolutePath

        return try {
            outFile.parentFile?.mkdirs()
            context.assets.open(assetPath).use { input ->
                outFile.outputStream().use { output -> input.copyTo(output) }
            }
            outFile.absolutePath
        } catch (e: Exception) {
            Log.e(tag, "Ошибка копирования $assetPath: ${e.message}")
            null
        }
    }
}