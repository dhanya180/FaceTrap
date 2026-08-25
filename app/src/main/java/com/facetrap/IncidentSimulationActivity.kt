package com.facetrap

import android.os.Bundle
import android.os.CountDownTimer
import android.view.View
import android.view.WindowManager
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import java.util.Locale

class IncidentSimulationActivity : AppCompatActivity() {
    private var timer: CountDownTimer? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        setContentView(R.layout.activity_incident_simulation)

        // Show system status – permanently locked
        val systemStatus = findViewById<TextView>(R.id.systemStatusAfterReset)
        systemStatus.text = "🔒 SYSTEM PERMANENTLY LOCKED – Files encrypted"
        systemStatus.setTextColor(ContextCompat.getColor(this, android.R.color.holo_red_dark))
        systemStatus.visibility = View.VISIBLE

        // Display the manifest
        findViewById<TextView>(R.id.manifestText).text =
            AvailabilitySimulation.manifestText(filesDir)

        // (No reset section to hide – removed reference)

        // Start countdown timer (just for display, no action)
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
}