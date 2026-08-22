package com.facetrap

import android.os.Bundle
import android.os.CountDownTimer
import android.view.WindowManager
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import java.util.Locale

class IncidentSimulationActivity : AppCompatActivity() {
    private var timer: CountDownTimer? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        setContentView(R.layout.activity_incident_simulation)

        findViewById<TextView>(R.id.manifestText).text =
            AvailabilitySimulation.manifestText(filesDir)

        val resetPhrase = findViewById<EditText>(R.id.resetPhrase)
        val resetButton = findViewById<Button>(R.id.resetButton)
        val resetStatus = findViewById<TextView>(R.id.resetStatus)
        resetButton.setOnClickListener {
            if (resetPhrase.text.toString() == RESET_PHRASE) {
                AvailabilitySimulation.reset(filesDir)
                resetStatus.setText(R.string.simulation_reset_success)
                resetButton.isEnabled = false
            } else {
                resetStatus.setText(R.string.simulation_reset_instruction)
            }
        }

        timer = object : CountDownTimer(60 * 60 * 1000L, 1000L) {
            override fun onTick(remaining: Long) {
                val seconds = remaining / 1000
                findViewById<TextView>(R.id.countdownText).text = String.format(
                    Locale.US,
                    "%02d:%02d:%02d",
                    seconds / 3600,
                    (seconds % 3600) / 60,
                    seconds % 60,
                )
            }

            override fun onFinish() {
                findViewById<TextView>(R.id.countdownText).setText(R.string.countdown_finished)
            }
        }.start()
    }

    override fun onDestroy() {
        timer?.cancel()
        super.onDestroy()
    }

    companion object {
        private const val RESET_PHRASE = "RESET-DEMO"
    }
}
