package com.facetrap

import java.io.File
import java.security.MessageDigest
import java.time.Instant
import java.time.format.DateTimeFormatter
import java.util.Locale

/**
 * Harmless availability-impact simulation for the CS402M emulator demonstration.
 *
 * It only creates decoy files below the supplied app-private root. Triggering writes a
 * marker and a manifest; it never encrypts, deletes, renames, or edits a decoy file.
 */
object AvailabilitySimulation {
    const val SANDBOX_DIRECTORY = "availability_demo"
    const val MANIFEST_FILE = "ransomware_manifest.txt"
    private const val BLOCK_MARKER = ".access_blocked"

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

    fun trigger(
        appFilesDir: File,
        targetIdentity: String,
        confidence: Float,
        triggeredAt: Instant = Instant.now(),
    ): TriggerResult {
        require(confidence > 0.85f) { "Target confidence must exceed 0.85" }
        val sandbox = initialize(appFilesDir)
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

        File(sandbox, BLOCK_MARKER).writeText("SIMULATION_ONLY\n")
        val manifest = File(sandbox, MANIFEST_FILE)
        manifest.writeText(
            buildString {
                appendLine("FACETRAP AVAILABILITY-IMPACT SIMULATION MANIFEST")
                appendLine("================================================")
                appendLine("Simulation only        : true")
                appendLine("Triggered at (UTC)     : ${DateTimeFormatter.ISO_INSTANT.format(triggeredAt)}")
                appendLine("Matched identity       : $targetIdentity")
                appendLine(String.format(Locale.US, "Decision confidence    : %.4f", confidence))
                appendLine("Scope                  : app-private generated decoys only")
                appendLine("Encryption             : none")
                appendLine("Deletion/renaming      : none")
                appendLine("Network activity       : none")
                appendLine("Simulated unavailable  : ${records.size}")
                appendLine("------------------------------------------------")
                records.forEach(::appendLine)
                appendLine("------------------------------------------------")
                appendLine("STATUS: access blocked by a reversible application-state marker")
            },
        )
        return TriggerResult(records.size, manifest)
    }

    fun isTriggered(appFilesDir: File): Boolean =
        File(File(appFilesDir, SANDBOX_DIRECTORY), BLOCK_MARKER).exists()

    /** Represents the application's normal data-access path used in the demonstration. */
    fun readDecoy(appFilesDir: File, filename: String): String {
        require(filename in decoys) { "Unknown demo file" }
        check(!isTriggered(appFilesDir)) { "Demo data is unavailable while simulation is active" }
        return File(initialize(appFilesDir), filename).readText()
    }

    fun manifestText(appFilesDir: File): String {
        val manifest = File(File(appFilesDir, SANDBOX_DIRECTORY), MANIFEST_FILE)
        return if (manifest.exists()) manifest.readText() else "Manifest has not been created."
    }

    fun reset(appFilesDir: File) {
        val sandbox = initialize(appFilesDir)
        File(sandbox, BLOCK_MARKER).delete()
        File(sandbox, MANIFEST_FILE).delete()
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
