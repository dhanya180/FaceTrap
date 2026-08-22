package com.facetrap

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.*
import android.os.Bundle
import android.util.Log
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import ai.onnxruntime.*
import java.nio.FloatBuffer
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.sqrt

class MainActivity : AppCompatActivity() {

    private lateinit var cameraExecutor: ExecutorService
    private lateinit var greetingText: TextView
    private lateinit var ortEnv: OrtEnvironment
    private lateinit var detSession: OrtSession
    private lateinit var embSession: OrtSession
    private lateinit var youRef: FloatArray
    private lateinit var teammateRef: FloatArray
    private lateinit var professorRef: FloatArray

    private val recognitionThreshold = 0.30f
    private val detectionThreshold = 0.50f
    private val detSize = 640

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        greetingText = findViewById(R.id.greetingText)
        cameraExecutor = Executors.newSingleThreadExecutor()

        ortEnv = OrtEnvironment.getEnvironment()
        detSession = ortEnv.createSession(loadAsset("det_10g.onnx"))
        embSession = ortEnv.createSession(loadAsset("w600k_r50.onnx"))

        Log.i("FaceTrap", "Detector inputs=${detSession.inputInfo} outputs=${detSession.outputInfo}")
        Log.i("FaceTrap", "Embedder inputs=${embSession.inputInfo} outputs=${embSession.outputInfo}")

        youRef       = loadNpy("you_ref.npy")
        teammateRef  = loadNpy("teammate_ref.npy")
        professorRef = loadNpy("professor_ref.npy")

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), 100)
        }
    }

    private fun loadAsset(name: String): ByteArray = assets.open(name).readBytes()

    private fun loadNpy(name: String): FloatArray {
        val bytes = assets.open(name).readBytes()
        require(bytes.size >= 12 && bytes.copyOfRange(0, 6).contentEquals(
            byteArrayOf(0x93.toByte(), 'N'.code.toByte(), 'U'.code.toByte(),
                'M'.code.toByte(), 'P'.code.toByte(), 'Y'.code.toByte())
        )) { "$name is not a NumPy .npy file" }

        val major = bytes[6].toInt()
        val headerLengthBytes = if (major == 1) 2 else 4
        val headerLen = if (headerLengthBytes == 2) {
            ByteBuffer.wrap(bytes, 8, 2).order(ByteOrder.LITTLE_ENDIAN).short.toInt() and 0xFFFF
        } else {
            ByteBuffer.wrap(bytes, 8, 4).order(ByteOrder.LITTLE_ENDIAN).int
        }
        val dataOffset = 8 + headerLengthBytes + headerLen
        val header = bytes.copyOfRange(8 + headerLengthBytes, dataOffset).toString(Charsets.US_ASCII)
        require(header.contains("<f4") || header.contains("'|f4'")) {
            "$name must contain little-endian float32 values; header=$header"
        }
        require((bytes.size - dataOffset) % 4 == 0) { "$name has an invalid data length" }
        val floatCount = (bytes.size - dataOffset) / 4
        require(floatCount == 512) { "$name must contain one 512-D ArcFace vector, found $floatCount floats" }
        val result = FloatArray(floatCount)
        val buf = ByteBuffer.wrap(bytes, dataOffset, bytes.size - dataOffset)
            .order(ByteOrder.LITTLE_ENDIAN)
        buf.asFloatBuffer().get(result)
        val norm = sqrt(result.sumOf { (it * it).toDouble() }).toFloat()
        require(norm > 0.99f && norm < 1.01f) { "$name is not L2-normalized (norm=$norm)" }
        Log.i("FaceTrap", "$name: version=$major, floats=$floatCount, norm=$norm")
        return result
    }

    private fun cosineSimilarity(a: FloatArray, b: FloatArray): Float {
        require(a.size == b.size) { "Embedding sizes differ: ${a.size} and ${b.size}" }
        var dot = 0f; var normA = 0f; var normB = 0f
        for (i in a.indices) {
            dot += a[i] * b[i]
            normA += a[i] * a[i]
            normB += b[i] * b[i]
        }
        return dot / (sqrt(normA) * sqrt(normB) + 1e-8f)
    }

    private fun identify(embedding: FloatArray): Pair<String, Float> {
        val scores = mapOf(
            "you"       to cosineSimilarity(embedding, youRef),
            "teammate"  to cosineSimilarity(embedding, teammateRef),
            "professor" to cosineSimilarity(embedding, professorRef)
        )
        Log.i("FaceTrap", "Cosine scores: you=${scores["you"]}, teammate=${scores["teammate"]}, professor=${scores["professor"]}")
        val best = scores.maxByOrNull { it.value }!!
        if (best.value < recognitionThreshold) return Pair("Unknown", best.value)
        val greeting = if (best.key == "professor") "Hi Sir" else "Hey Students"
        return Pair(greeting, best.value)
    }

    // Matches insightface SCRFD: RGB NCHW, (pixel - 127.5) / 128, top-left padding.
    private fun bitmapToDetInput(bitmap: Bitmap): Pair<FloatArray, Float> {
        val imRatio = bitmap.height.toFloat() / bitmap.width
        val newW: Int
        val newH: Int
        if (imRatio > 1f) {
            newH = detSize
            newW = (detSize / imRatio).toInt()
        } else {
            newW = detSize
            newH = (detSize * imRatio).toInt()
        }
        val detScale = newH.toFloat() / bitmap.height
        val resized = Bitmap.createScaledBitmap(bitmap, newW, newH, true)
        val padded = Bitmap.createBitmap(detSize, detSize, Bitmap.Config.ARGB_8888)
        Canvas(padded).drawBitmap(resized, 0f, 0f, null)
        resized.recycle()

        val floats = FloatArray(3 * detSize * detSize)
        val pixels = IntArray(detSize * detSize)
        padded.getPixels(pixels, 0, detSize, 0, 0, detSize, detSize)
        padded.recycle()
        for (i in pixels.indices) {
            val p = pixels[i]
            floats[i]                         = ((p shr 16 and 0xFF) - 127.5f) / 128f // R
            floats[detSize * detSize + i]     = ((p shr 8 and 0xFF) - 127.5f) / 128f  // G
            floats[2 * detSize * detSize + i] = ((p and 0xFF) - 127.5f) / 128f        // B
        }
        return Pair(floats, detScale)
    }

    private val ARCFACE_DST = floatArrayOf(
        38.2946f, 51.6963f, 73.5318f, 51.5014f, 56.0252f, 71.7366f,
        41.5493f, 92.3655f, 70.7299f, 92.2041f
    )

    private data class FaceDet(val box: IntArray, val kps: FloatArray, val score: Float)

    private fun similarityMatrix(src: FloatArray, dst: FloatArray): Matrix {
        val n = 5
        var scx = 0f; var scy = 0f; var dcx = 0f; var dcy = 0f
        for (i in 0 until n) {
            scx += src[i * 2]; scy += src[i * 2 + 1]
            dcx += dst[i * 2]; dcy += dst[i * 2 + 1]
        }
        scx /= n; scy /= n; dcx /= n; dcy /= n
        var varSrc = 0f; var c = 0f; var d = 0f
        for (i in 0 until n) {
            val sx = src[i * 2] - scx; val sy = src[i * 2 + 1] - scy
            val dx = dst[i * 2] - dcx; val dy = dst[i * 2 + 1] - dcy
            varSrc += sx * sx + sy * sy
            c += sx * dx + sy * dy
            d += sx * dy - sy * dx
        }
        val a = c / varSrc
        val b = d / varSrc
        val tx = dcx - a * scx + b * scy
        val ty = dcy - b * scx - a * scy
        return Matrix().apply { setValues(floatArrayOf(a, -b, tx, b, a, ty, 0f, 0f, 1f)) }
    }

    private fun alignFace(bitmap: Bitmap, kps: FloatArray): Bitmap {
        val m = similarityMatrix(kps, ARCFACE_DST)
        val out = Bitmap.createBitmap(112, 112, Bitmap.Config.ARGB_8888)
        Canvas(out).drawBitmap(bitmap, m, Paint(Paint.FILTER_BITMAP_FLAG))
        return out
    }

    private fun alignedToEmbedInput(aligned: Bitmap): FloatArray {
        val floats = FloatArray(3 * 112 * 112)
        val pixels = IntArray(112 * 112)
        aligned.getPixels(pixels, 0, 112, 0, 0, 112, 112)
        for (i in pixels.indices) {
            val p = pixels[i]
            // w600k_r50 ArcFace preprocessing uses mean=127.5 and std=127.5.
            floats[i]                 = ((p shr 16 and 0xFF) - 127.5f) / 127.5f
            floats[112 * 112 + i]     = ((p shr 8  and 0xFF) - 127.5f) / 127.5f
            floats[2 * 112 * 112 + i] = ((p         and 0xFF) - 127.5f) / 127.5f
        }
        return floats
    }

    private fun detectFace(bitmap: Bitmap): FaceDet? {
        val (detInput, detScale) = bitmapToDetInput(bitmap)
        val shape = longArrayOf(1, 3, detSize.toLong(), detSize.toLong())

        var best: FaceDet? = null

        OnnxTensor.createTensor(ortEnv, FloatBuffer.wrap(detInput), shape).use { tensor ->
            detSession.run(mapOf(detSession.inputNames.first() to tensor)).use { output ->
        try {
            val scoreNames = listOf("448", "471", "494")
            val bboxNames  = listOf("451", "474", "497")
            val kpsNames   = listOf("454", "477", "500")
            val strides    = listOf(8, 16, 32)

            for (idx in strides.indices) {
                val stride = strides[idx]
                val scores = output.get(scoreNames[idx]).get().value as Array<FloatArray>
                val bboxes = output.get(bboxNames[idx]).get().value as Array<FloatArray>
                val kpss   = output.get(kpsNames[idx]).get().value as Array<FloatArray>
                val featW = detSize / stride
                val numAnchors = 2

                for (i in scores.indices) {
                    val score = scores[i][0]
                    val currentBest = best
                    if (currentBest != null && score <= currentBest.score) continue

                    val spatialIdx = i / numAnchors
                    val row = spatialIdx / featW
                    val col = spatialIdx % featW
                    // InsightFace SCRFD uses grid locations (col * stride, row * stride).
                    // A half-stride offset corrupts the five landmarks and therefore ArcFace alignment.
                    val cx = col.toFloat() * stride
                    val cy = row.toFloat() * stride

                    val x1 = ((cx - bboxes[i][0] * stride) / detScale).toInt()
                    val y1 = ((cy - bboxes[i][1] * stride) / detScale).toInt()
                    val x2 = ((cx + bboxes[i][2] * stride) / detScale).toInt()
                    val y2 = ((cy + bboxes[i][3] * stride) / detScale).toInt()

                    val kps = FloatArray(10)
                    for (k in 0 until 5) {
                        kps[k * 2]     = (cx + kpss[i][k * 2] * stride) / detScale
                        kps[k * 2 + 1] = (cy + kpss[i][k * 2 + 1] * stride) / detScale
                    }
                    best = FaceDet(intArrayOf(x1, y1, x2, y2), kps, score)
                }
            }
            best?.let { Log.i("FaceTrap", "Detection score=${it.score}, box=${it.box.toList()}, kps=${it.kps.toList()}") }
        } catch (e: Exception) {
            Log.e("FaceTrap", "Detection parse error", e)
        }
            }
        }
        return best?.takeIf { it.score >= detectionThreshold }
    }

    private fun prepareBitmap(bitmap: Bitmap, rotationDegrees: Int): Bitmap {
        if (rotationDegrees == 0) return bitmap
        val m = Matrix().apply { postRotate(rotationDegrees.toFloat()) }
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, m, true)
    }

    private fun runEmbedding(floats: FloatArray): FloatArray {
        val shape = longArrayOf(1, 3, 112, 112)
        val embedding = OnnxTensor.createTensor(ortEnv, FloatBuffer.wrap(floats), shape).use { tensor ->
            embSession.run(mapOf(embSession.inputNames.first() to tensor)).use { output ->
                ((output[0].value as Array<FloatArray>)[0]).copyOf()
            }
        }
        val norm = sqrt(embedding.map { it * it }.sum())
        Log.i("FaceTrap", "Raw embedding norm=$norm")
        return if (norm > 0) FloatArray(embedding.size) { embedding[it] / norm } else embedding
    }

    private fun processFrame(bitmap: Bitmap) {
        try {
            val face = detectFace(bitmap) ?: run {
                runOnUiThread { greetingText.text = "No face" }
                return
            }
            val aligned = alignFace(bitmap, face.kps)
            val floats = alignedToEmbedInput(aligned)
            aligned.recycle()
            val (greeting, score) = identify(runEmbedding(floats))
            Log.i("FaceTrap", "Best score=$score, result=$greeting")
            runOnUiThread {
                greetingText.text = "$greeting (${String.format("%.2f", score)})"
            }
        } catch (e: Exception) {
            Log.e("FaceTrap", "Frame error", e)
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(findViewById<PreviewView>(R.id.previewView).surfaceProvider)
            }
            val analysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
            analysis.setAnalyzer(cameraExecutor) { imageProxy ->
                try {
                    val raw = imageProxy.toBitmap()
                    val bitmap = prepareBitmap(raw, imageProxy.imageInfo.rotationDegrees)
                    if (raw != bitmap) raw.recycle()
                    Log.d("FaceTrap", "Frame=${bitmap.width}x${bitmap.height}, rotation=${imageProxy.imageInfo.rotationDegrees}")
                    processFrame(bitmap)
                    bitmap.recycle()
                } catch (e: Exception) {
                    Log.e("FaceTrap", "Camera frame conversion failed", e)
                } finally {
                    imageProxy.close()
                }
            }
            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(
                this, CameraSelector.DEFAULT_FRONT_CAMERA, preview, analysis
            )
        }, ContextCompat.getMainExecutor(this))
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == 100 && grantResults.firstOrNull() == PackageManager.PERMISSION_GRANTED)
            startCamera()
        else if (requestCode == 100)
            greetingText.text = "Camera permission is required"
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        detSession.close()
        embSession.close()
        ortEnv.close()
    }
}
