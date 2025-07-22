package com.example.mrz_mlkit_test

import android.os.Bundle
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import android.content.Context
import android.net.Uri
import androidx.documentfile.provider.DocumentFile
import android.content.Intent
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import android.app.ProgressDialog
import android.graphics.BitmapFactory
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.BufferedReader
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlin.coroutines.resumeWithException
import android.widget.Toast
import java.io.OutputStreamWriter
import android.graphics.Bitmap
import androidx.core.content.ContextCompat
import android.app.AlertDialog
import android.graphics.Canvas
import android.graphics.ColorMatrix
import android.graphics.ColorMatrixColorFilter
import android.graphics.Paint

data class ProcessingResult(
    val country: String,
    val filename: String,
    val expectedText: String?,
    val recognizedText: String?,
    val match: Boolean,
    val error: String? = null
)

/**
 * Converts a color bitmap to grayscale for improved OCR performance.
 */
fun toGrayscale(original: Bitmap): Bitmap {
    val grayscale = Bitmap.createBitmap(original.width, original.height, Bitmap.Config.ARGB_8888)
    val canvas = Canvas(grayscale)
    val paint = Paint()
    val colorMatrix = ColorMatrix()
    colorMatrix.setSaturation(0f)
    val filter = ColorMatrixColorFilter(colorMatrix)
    paint.colorFilter = filter
    canvas.drawBitmap(original, 0f, 0f, paint)
    return grayscale
}

/**
 * Applies Otsu thresholding to create a binary image.
 */
fun applyOtsuThreshold(grayscaleBitmap: Bitmap): Bitmap {
    val width = grayscaleBitmap.width
    val height = grayscaleBitmap.height
    val pixels = IntArray(width * height)
    grayscaleBitmap.getPixels(pixels, 0, width, 0, 0, width, height)
    
    // Calculate histogram
    val histogram = IntArray(256)
    for (pixel in pixels) {
        val gray = android.graphics.Color.red(pixel)
        histogram[gray]++
    }
    
    // Otsu's algorithm
    val total = width * height
    var sum = 0f
    for (i in 0..255) {
        sum += i * histogram[i]
    }
    
    var sumB = 0f
    var wB = 0
    var maximum = 0f
    var threshold = 0
    
    for (i in 0..255) {
        wB += histogram[i]
        if (wB == 0) continue
        
        val wF = total - wB
        if (wF == 0) break
        
        sumB += i * histogram[i]
        val mB = sumB / wB
        val mF = (sum - sumB) / wF
        
        val between = wB.toFloat() * wF.toFloat() * (mB - mF) * (mB - mF)
        
        if (between > maximum) {
            maximum = between
            threshold = i
        }
    }
    
    // Apply threshold
    val binaryBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
    val binaryPixels = IntArray(width * height)
    
    for (i in pixels.indices) {
        val gray = android.graphics.Color.red(pixels[i])
        binaryPixels[i] = if (gray <= threshold) {
            android.graphics.Color.BLACK
        } else {
            android.graphics.Color.WHITE
        }
    }
    
    binaryBitmap.setPixels(binaryPixels, 0, width, 0, 0, width, height)
    return binaryBitmap
}

/**
 * Converts color bitmap to grayscale and applies Otsu thresholding.
 */
fun preprocessImage(original: Bitmap): Bitmap {
    val grayscale = toGrayscale(original)
    return applyOtsuThreshold(grayscale)
}

/**
 * Process OCR text by removing newlines and joining lines.
 */
fun processOCRText(text: String): String {
    return text.replace("\n", "").replace("\r", "").trim()
}

object DatasetUtils {
    /**
     * Find image-text pairs using fixed naming convention (00.jpg to 19.jpg).
     */
    fun findImageTextPairs(context: Context, rootUri: Uri): List<Triple<String, Uri, Uri?>> {
        val pairs = mutableListOf<Triple<String, Uri, Uri?>>()
        val children = DocumentFile.fromTreeUri(context, rootUri)?.listFiles() ?: return pairs
        
        for (countryDir in children) {
            if (!countryDir.isDirectory) continue
            
            // Look for files 00.jpg to 19.jpg and corresponding .txt files
            for (i in 0..19) {
                val filename = String.format("%02d", i)
                val jpgFile = countryDir.listFiles().find { it.name == "$filename.jpg" }
                val txtFile = countryDir.listFiles().find { it.name == "$filename.txt" }
                
                if (jpgFile != null) {
                    pairs.add(Triple(countryDir.name ?: "", jpgFile.uri, txtFile?.uri))
                }
            }
        }
        return pairs
    }
}

class MainActivity : AppCompatActivity() {
    private lateinit var resultsAdapter: ResultsAdapter
    private val results = mutableListOf<ProcessingResult>()
    private val job = Job()
    private val uiScope = CoroutineScope(Dispatchers.Main + job)

    override fun onDestroy() {
        super.onDestroy()
        job.cancel()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main)) { v, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets
        }

        val tvTotal = findViewById<TextView>(R.id.tv_total)
        val tvMatched = findViewById<TextView>(R.id.tv_matched)
        val tvMismatched = findViewById<TextView>(R.id.tv_mismatched)
        val tvFailed = findViewById<TextView>(R.id.tv_failed)

        val btnSelectDir = findViewById<View>(R.id.btn_select_directory)
        val rvResults = findViewById<RecyclerView>(R.id.rv_results)
        resultsAdapter = ResultsAdapter(results)
        rvResults.layoutManager = LinearLayoutManager(this)
        rvResults.adapter = resultsAdapter

        fun updateSummary() {
            val total = results.size
            val matched = results.count { it.match && it.error == null }
            val mismatched = results.count { !it.match && it.error == null }
            val failed = results.count { it.error != null }
            tvTotal.text = "Total: $total"
            tvMatched.text = "Matched: $matched"
            tvMismatched.text = "Mismatched: $mismatched"
            tvFailed.text = "Failed: $failed"
        }

        val dirPickerLauncher = registerForActivityResult(ActivityResultContracts.OpenDocumentTree()) { uri ->
            if (uri != null) {
                processDataset(uri, ::updateSummary)
            }
        }

        btnSelectDir.setOnClickListener {
            dirPickerLauncher.launch(null)
        }

        val btnExport = findViewById<View>(R.id.btn_export_results)
        val exportLauncher = registerForActivityResult(ActivityResultContracts.CreateDocument("text/csv")) { uri ->
            if (uri != null) {
                exportResultsToCsv(uri)
            }
        }
        btnExport.setOnClickListener {
            if (results.isEmpty()) {
                Toast.makeText(this, "No results to export", Toast.LENGTH_SHORT).show()
            } else {
                exportLauncher.launch("results.csv")
            }
        }
    }

    /**
     * Process dataset with simplified logic.
     */
    private fun processDataset(rootUri: Uri, updateSummary: () -> Unit) {
        val progressDialog = ProgressDialog(this).apply {
            setMessage("Processing dataset...")
            setCancelable(false)
            show()
        }
        
        uiScope.launch {
            val pairs = withContext(Dispatchers.IO) {
                DatasetUtils.findImageTextPairs(this@MainActivity, rootUri)
            }
            
            results.clear()
            resultsAdapter.notifyDataSetChanged()
            updateSummary()
            
            val recognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)
            
            for ((idx, triple) in pairs.withIndex()) {
                val (country, jpgUri, txtUri) = triple
                var expectedText: String? = null
                var recognizedText: String? = null
                var match = false
                var error: String? = null
                
                try {
                    // Read expected text from .txt file
                    if (txtUri == null) {
                        error = "Missing .txt file"
                    } else {
                        expectedText = withContext(Dispatchers.IO) {
                            contentResolver.openInputStream(txtUri)?.use { inputStream ->
                                processOCRText(inputStream.bufferedReader().readText())
                            }
                        }
                    }
                    
                    // Process image
                    val bitmap = withContext(Dispatchers.IO) {
                        contentResolver.openInputStream(jpgUri)?.use { inputStream ->
                            BitmapFactory.decodeStream(inputStream)
                        }
                    }
                    
                    if (bitmap == null) {
                        error = "Failed to decode image"
                    } else {
                        // Crop to bottom 25% for MRZ region
                        val cropRatio = 0.25f
                        val cropHeight = (bitmap.height * cropRatio).toInt()
                        val yStart = bitmap.height - cropHeight
                        val croppedBitmap = Bitmap.createBitmap(bitmap, 150, yStart, bitmap.width - 150, cropHeight)
                        
                        // Apply grayscale and threshold preprocessing
                        val processedBitmap = preprocessImage(croppedBitmap)
                        
                        // Run OCR
                        val image = InputImage.fromBitmap(processedBitmap, 0)
                        val result = withContext(Dispatchers.IO) {
                            recognizer.process(image).await()
                        }
                        
                        // Process OCR output - remove newlines and join
                        recognizedText = processOCRText(result.text)
                        
                        Log.d("OCR", "${jpgUri.lastPathSegment} - Expected: $expectedText")
                        Log.d("OCR", "${jpgUri.lastPathSegment} - Recognized: $recognizedText")
                        
                        // Simple string comparison
                        match = expectedText != null && expectedText == recognizedText
                    }
                } catch (e: Exception) {
                    error = e.message
                    Log.e("OCR", "Error processing ${jpgUri.lastPathSegment}", e)
                }
                
                results.add(
                    ProcessingResult(
                        country = country,
                        filename = jpgUri.lastPathSegment ?: "",
                        expectedText = expectedText,
                        recognizedText = recognizedText,
                        match = match,
                        error = error
                    )
                )
                
                resultsAdapter.notifyItemInserted(results.size - 1)
                updateSummary()
                progressDialog.setMessage("Processing ${idx + 1} / ${pairs.size}")
            }
            
            progressDialog.dismiss()
        }
    }

    /**
     * Export results to CSV.
     */
    private fun exportResultsToCsv(uri: Uri) {
        try {
            contentResolver.openOutputStream(uri)?.use { outputStream ->
                OutputStreamWriter(outputStream).use { writer ->
                    writer.write("country,filename,status,expectedText,recognizedText\n")
                    
                    for (result in results) {
                        val status = when {
                            result.error != null -> "Failed: ${result.error.replace(",", " ")}"
                            result.match -> "Matched"
                            else -> "Mismatched"
                        }
                        
                        writer.write("${result.country},${result.filename},${status},\"${result.expectedText?.replace("\"", "'") ?: ""}\",\"${result.recognizedText?.replace("\"", "'") ?: ""}\"\n")
                    }
                }
            }
            Toast.makeText(this, "Exported successfully", Toast.LENGTH_SHORT).show()
        } catch (e: Exception) {
            Toast.makeText(this, "Export failed: ${e.message}", Toast.LENGTH_LONG).show()
        }
    }

    class ResultsAdapter(private val items: List<ProcessingResult>) : RecyclerView.Adapter<ResultsAdapter.ViewHolder>() {
        class ViewHolder(view: View) : RecyclerView.ViewHolder(view) {
            val tvCountry: TextView = view.findViewById(R.id.tv_country)
            val tvFilename: TextView = view.findViewById(R.id.tv_filename)
            val tvStatus: TextView = view.findViewById(R.id.tv_status)
        }

        override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
            val v = LayoutInflater.from(parent.context).inflate(R.layout.item_result, parent, false)
            return ViewHolder(v)
        }

        override fun onBindViewHolder(holder: ViewHolder, position: Int) {
            val item = items[position]
            holder.tvCountry.text = item.country
            holder.tvFilename.text = item.filename
            
            val (status, colorRes) = when {
                item.error != null -> "Failed" to android.R.color.holo_red_dark
                item.match -> "Matched" to android.R.color.holo_green_dark
                else -> "Mismatched" to android.R.color.holo_orange_dark
            }
            
            holder.tvStatus.text = status
            holder.tvStatus.setTextColor(ContextCompat.getColor(holder.tvStatus.context, colorRes))

            holder.itemView.setOnClickListener {
                val context = holder.itemView.context
                val message = buildString {
                    append("Country: ${item.country}\n")
                    append("Filename: ${item.filename}\n")
                    append("Status: $status\n\n")
                    append("Expected Text:\n${item.expectedText ?: "-"}\n\n")
                    append("Recognized Text:\n${item.recognizedText ?: "-"}\n")
                    if (item.error != null) {
                        append("\nError: ${item.error}")
                    }
                }
                
                AlertDialog.Builder(context)
                    .setTitle("Result Details")
                    .setMessage(message)
                    .setPositiveButton("OK", null)
                    .show()
            }
        }

        override fun getItemCount() = items.size
    }
}

/**
 * Extension function for Task.await()
 */
suspend fun <T> com.google.android.gms.tasks.Task<T>.await(): T =
    suspendCancellableCoroutine { cont ->
        addOnSuccessListener { cont.resume(it) {} }
        addOnFailureListener { cont.resumeWithException(it) }
    }