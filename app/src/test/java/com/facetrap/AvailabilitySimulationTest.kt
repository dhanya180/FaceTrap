package com.facetrap

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.time.Instant

class AvailabilitySimulationTest {
    @get:Rule
    val temporaryFolder = TemporaryFolder()

    @Test
    fun triggerBlocksOnlyApplicationReadPathWithoutChangingDecoys() {
        val root = temporaryFolder.newFolder("app-files")
        val sandbox = AvailabilitySimulation.initialize(root)
        val before = sandbox.listFiles()!!
            .filter { !it.name.startsWith(".") }
            .associate { it.name to it.readBytes() }

        val result = AvailabilitySimulation.trigger(
            root,
            "professor",
            0.91f,
            Instant.parse("2026-08-22T10:15:30Z"),
        )

        assertTrue(AvailabilitySimulation.isTriggered(root))
        assertEquals(before.size, result.affectedFiles)
        before.forEach { (name, bytes) ->
            assertTrue(bytes.contentEquals(sandbox.resolve(name).readBytes()))
        }
        assertThrows(IllegalStateException::class.java) {
            AvailabilitySimulation.readDecoy(root, "maintenance_window.txt")
        }
        assertTrue(result.manifest.readText().contains("Encryption             : none"))
    }

    @Test
    fun resetRestoresReadPathAndLeavesDecoysIntact() {
        val root = temporaryFolder.newFolder("app-files")
        AvailabilitySimulation.trigger(root, "professor", 0.90f)

        AvailabilitySimulation.reset(root)

        assertFalse(AvailabilitySimulation.isTriggered(root))
        assertTrue(AvailabilitySimulation.readDecoy(root, "maintenance_window.txt").isNotBlank())
    }

    @Test
    fun triggerRejectsConfidenceAtOrBelowAssignmentThreshold() {
        val root = temporaryFolder.newFolder("app-files")
        assertThrows(IllegalArgumentException::class.java) {
            AvailabilitySimulation.trigger(root, "professor", 0.85f)
        }
    }

    @Test
    fun targetConfidenceCalibrationSeparatesMeasuredValidationRanges() {
        assertTrue(TargetConfidence.fromCosine(0.70f) > 0.85f)
        assertTrue(TargetConfidence.fromCosine(0.14f) < 0.15f)
    }
}
