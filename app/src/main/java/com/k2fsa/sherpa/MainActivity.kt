package com.k2fsa.sherpa

import android.net.Uri
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.material3.Button
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.k2fsa.sherpa.common.manager.AudioRecognitionManager
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : ComponentActivity() {

    private val tag = "AudioRecognition"

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val manager = AudioRecognitionManager(this)

        setContent {
            var resultText by remember { mutableStateOf("Выберите аудиофайл для теста") }
            var isLoading by remember { mutableStateOf(false) }
            val scope = rememberCoroutineScope()

            // Лаунчер для выбора файла
            val filePickerLauncher = rememberLauncherForActivityResult(
                contract = ActivityResultContracts.GetContent()
            ) { uri: Uri? ->
                if (uri != null) {
                    isLoading = true
                    resultText = "Обработка выбранного файла... Смотрите Logcat"

                    scope.launch {
                        val result = withContext(Dispatchers.IO) {
                            Log.d(tag, "Старт обработки URI: $uri")
                            // Передаем обычный URI из проводника
                            manager.processAudioFile(uri.toString())
                        }

                        isLoading = false
                        if (result != null) {
                            val formattedText = result.utterances.joinToString("\n\n") { utt ->
                                "Спикер ${utt.speakerId + 1} (${String.format("%.1f", utt.startSec)}c): ${utt.text}"
                            }
                            resultText = "УСПЕХ!\n\n$formattedText\n\nВремя: ${result.processingTimeMs} мс"
                        } else {
                            resultText = "ОШИБКА! Проверьте Logcat."
                        }
                    }
                } else {
                    resultText = "Файл не выбран"
                }
            }

            Column(modifier = Modifier.fillMaxSize().padding(16.dp).statusBarsPadding(),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Button(
                    onClick = {
                        // Открываем проводник и фильтруем только аудио
                        filePickerLauncher.launch("audio/*")
                    },
                    enabled = !isLoading
                ) {
                    Text(if (isLoading) "Работаю..." else "Выбрать аудиофайл")
                }

                Text(text = resultText, modifier = Modifier.padding(top = 16.dp))
            }
        }
    }
}