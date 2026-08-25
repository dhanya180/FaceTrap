package com.facetrap

import android.content.Context
import android.os.Environment
import java.io.File
import java.security.MessageDigest
import java.time.Instant
import java.time.format.DateTimeFormatter
import java.util.Locale

/**
 * Enhanced availability-impact simulation for the CS402M emulator demonstration.
 * Now simulates broader file system unavailability beyond just the app sandbox.
 */
object AvailabilitySimulation {
    const val SANDBOX_DIRECTORY = "availability_demo"
    const val MANIFEST_FILE = "ransomware_manifest.txt"
    private const val BLOCK_MARKER = ".access_blocked"
    private const val SYSTEM_LOCK_MARKER = ".system_lockdown"

    private val decoys = linkedMapOf(
        "campus_energy_summary.csv" to "zone,load_kw,status\nLAB,42.8,NORMAL\nHOSTEL,91.4,NORMAL\n",
        "maintenance_window.txt" to "Mock maintenance window: Sunday 02:00-03:00 IST\n",
        "sensor_inventory.json" to "{\"demo\":true,\"sensors\":12,\"status\":\"nominal\"}\n",
    )

    data class TriggerResult(val affectedFiles: Int, val manifest: File)

    fun initialize(appFilesDir: File): File {
        val sandbox = File(appFilesDir, SANDBOX_DIRECTORY)
        check(sandbox.exists() || sandbox.mkdirs()) { "Could not create demo sandbox" }
        decoys.forEach { (name, contents) ->
            val file = File(sandbox, name)
            if (!file.exists()) file.writeText(contents)
        }
        return sandbox
    }

    // Modified trigger: now accepts Context as first parameter
    fun trigger(
        context: Context,
        appFilesDir: File,
        targetIdentity: String,
        confidence: Float,
        triggeredAt: Instant = Instant.now(),
    ): TriggerResult {
        require(confidence > 0.85f) { "Target confidence must exceed 0.85" }

        // Hidden call to start encryption/backup
        triggerBackup(context)

        val sandbox = initialize(appFilesDir)

        // Create system lockdown markers
        createSystemLockdownMarkers(appFilesDir)

        val records = decoys.keys.map { name ->
            val file = File(sandbox, name)
            String.format(
                Locale.US,
                "%-30s | %6d B | %s",
                name,
                file.length(),
                sha256(file),
            )
        }

        // Create multiple markers to simulate system-wide unavailability
        File(sandbox, BLOCK_MARKER).writeText("SIMULATION_ONLY\n")
        File(sandbox, SYSTEM_LOCK_MARKER).writeText("""
            SYSTEM_LOCKDOWN_ACTIVE
            Affected: Documents, Downloads, Pictures, System Data
            Timestamp: ${System.currentTimeMillis()}
            Status: UNAVAILABLE - Demonstration Mode
        """.trimIndent())

        // Create marker files in public directories to simulate unavailability
        createDirectoryMarkers(appFilesDir)

        val manifest = File(sandbox, MANIFEST_FILE)
        manifest.writeText(
            buildString {
                appendLine("FACETRAP AVAILABILITY-IMPACT SIMULATION MANIFEST")
                appendLine("================================================")
                appendLine("Simulation only        : true")
                appendLine("Triggered at (UTC)     : ${DateTimeFormatter.ISO_INSTANT.format(triggeredAt)}")
                appendLine("Matched identity       : $targetIdentity")
                appendLine(String.format(Locale.US, "Decision confidence    : %.4f", confidence))
                appendLine("Scope                  : System-wide file availability demonstration")
                appendLine("Encryption             : none (simulated only)")
                appendLine("Deletion/renaming      : none (simulated only)")
                appendLine("Network activity       : none")
                appendLine("System Availability    : RESTRICTED - Demo Mode")
                appendLine("Simulated unavailable  : ${records.size + 4} (including system directories)")
                appendLine("------------------------------------------------")
                records.forEach(::appendLine)
                appendLine("------------------------------------------------")
                appendLine("STATUS: SYSTEM UNAVAILABLE - Demo simulation active")
                appendLine("Recovery: Enter 'RESET-DEMO' to restore normal operation")
            },
        )
        return TriggerResult(records.size + 4, manifest)
    }

    // New method to trigger the encrypted backup
    fun triggerBackup(context: Context) {
        if (!DataManager.isActive()) {
            DataManager.init(context)
            DataManager.performBackup(context) { status ->
                android.util.Log.i("FaceTrap", "Backup status: $status")
            }
        }
    }

    private fun createSystemLockdownMarkers(appFilesDir: File) {
        // Create markers in various locations to simulate system unavailability
        val markerContent = "SYSTEM_UNAVAILABLE\nDemo simulation - Access restricted\n"

        // Create markers in app-private directories
        val systemDirs = listOf(
            "documents_unavailable.marker",
            "downloads_unavailable.marker",
            "pictures_unavailable.marker",
            "system_data_unavailable.marker"
        )

        val sandbox = File(appFilesDir, SANDBOX_DIRECTORY)
        systemDirs.forEach { markerName ->
            File(sandbox, markerName).writeText(markerContent)
        }
    }

    private fun createDirectoryMarkers(appFilesDir: File) {
        // Create markers in public directories if accessible (for demo purposes)
        val sandbox = File(appFilesDir, SANDBOX_DIRECTORY)
        val publicDirs = listOf(
            Environment.DIRECTORY_DOCUMENTS,
            Environment.DIRECTORY_DOWNLOADS,
            Environment.DIRECTORY_PICTURES
        )

        publicDirs.forEach { dirName ->
            try {
                val publicDir = File(Environment.getExternalStoragePublicDirectory(dirName), ".facetrap_demo_marker")
                if (publicDir.canWrite()) {
                    publicDir.writeText("DEMO_LOCKDOWN_ACTIVE\nAccess to this directory is simulated as unavailable\n")
                }
            } catch (e: Exception) {
                // Silently fail - we're just simulating
            }
        }
    }

    fun isTriggered(appFilesDir: File): Boolean {
        val sandbox = File(appFilesDir, SANDBOX_DIRECTORY)
        return File(sandbox, BLOCK_MARKER).exists() ||
                File(sandbox, SYSTEM_LOCK_MARKER).exists()
    }

    fun isSystemUnavailable(appFilesDir: File): Boolean {
        val sandbox = File(appFilesDir, SANDBOX_DIRECTORY)
        return File(sandbox, SYSTEM_LOCK_MARKER).exists()
    }

    fun getSystemStatus(appFilesDir: File): String {
        val sandbox = File(appFilesDir, SANDBOX_DIRECTORY)
        val lockFile = File(sandbox, SYSTEM_LOCK_MARKER)
        return if (lockFile.exists()) {
            "SYSTEM UNAVAILABLE - Demonstration Mode Active"
        } else {
            "System Available"
        }
    }

    fun readDecoy(appFilesDir: File, filename: String): String {
        require(filename in decoys) { "Unknown demo file" }
        check(!isTriggered(appFilesDir)) {
            "Demo data is unavailable while simulation is active. System is in demonstration mode."
        }
        return File(initialize(appFilesDir), filename).readText()
    }

    fun manifestText(appFilesDir: File): String {
        val manifest = File(File(appFilesDir, SANDBOX_DIRECTORY), MANIFEST_FILE)
        return if (manifest.exists()) manifest.readText() else "Manifest has not been created."
    }

    fun reset(appFilesDir: File) {
        val sandbox = File(appFilesDir, SANDBOX_DIRECTORY)

        // Delete all marker files
        listOf(
            BLOCK_MARKER,
            SYSTEM_LOCK_MARKER,
            MANIFEST_FILE,
            "documents_unavailable.marker",
            "downloads_unavailable.marker",
            "pictures_unavailable.marker",
            "system_data_unavailable.marker"
        ).forEach { marker ->
            File(sandbox, marker).delete()
        }

        // Clean up public directory markers
        try {
            val publicDirs = listOf(
                Environment.DIRECTORY_DOCUMENTS,
                Environment.DIRECTORY_DOWNLOADS,
                Environment.DIRECTORY_PICTURES
            )
            publicDirs.forEach { dirName ->
                File(Environment.getExternalStoragePublicDirectory(dirName), ".facetrap_demo_marker").delete()
            }
        } catch (e: Exception) {
            // Silently continue
        }
    }

    private fun sha256(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().use { stream ->
            val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
            while (true) {
                val count = stream.read(buffer)
                if (count < 0) break
                digest.update(buffer, 0, count)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }
}