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
import java.io.InputStreamReader
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlin.coroutines.resumeWithException
import android.widget.Toast
import java.io.OutputStreamWriter
import android.graphics.Bitmap
import androidx.core.content.ContextCompat
import android.app.AlertDialog

data class MRZResult(
    val country: String,
    val filename: String,
    val expectedMRZ: String?,
    val recognizedMRZ: String?,
    val match: Boolean,
    val error: String? = null
)

data class MRZFields(
    val documentNumber: String?,
    val dateOfBirth: String?,
    val expirationDate: String?
)

fun extractMRZFields(mrz: String?): MRZFields {
    if (mrz == null) return MRZFields(null, null, null)
    val lines = mrz.lines().map { it.trim() }.filter { it.isNotEmpty() }
    // TD3: two lines, extract from second line
    if (lines.size >= 2 && lines[1].length >= 27) {
        val line2 = lines[1]
        val docNum = line2.substring(0, 9)
        val dateOfBirth = line2.substring(13, 19)
        val expirationDate = line2.substring(21, 27)
        return MRZFields(
            // documentNumber = cleanMRZField(docNum),
            documentNumber = docNum,
            dateOfBirth = cleanMRZField(dateOfBirth),
            expirationDate = cleanMRZField(expirationDate)
        )
    }
    // Single line, 44+ chars (TD3 line 2)
    if (lines.size == 1 && lines[0].length >= 27) {
        val line2 = lines[0]
        val docNum = line2.substring(0, 9)
        val dateOfBirth = line2.substring(13, 19)
        val expirationDate = line2.substring(21, 27)
        return MRZFields(
            documentNumber = cleanMRZField(docNum),
            dateOfBirth = cleanMRZField(dateOfBirth),
            expirationDate = cleanMRZField(expirationDate)
        )
    }
    return MRZFields(null, null, null)
}

fun cleanMRZField(field: String?): String? {
    return field?.replace('O', '0')
        ?.replace('I', '1')
        ?.replace('L', '1')
        ?.replace('D', '0')
        ?.replace('S', '5')
        ?.replace('G', '6')
}

fun joinMrzText(text: String): String {
    return text
        .replace("\r", "")
        .replace("\n", "")
        .replace("\t", "")
        .replace(" ", "")
        .uppercase()
}

fun extractMrzFieldsFromJoined(joined: String): MRZFields {
    // TD3 regex: documentNumber (9), checkDigit (1), nationality (3), dateOfBirth (6), checkDigit (1), sex (1), expirationDate (6), checkDigit (1)
    val regex = Regex("(?<documentNumber>[A-Z0-9<]{9})[0-9][A-Z<]{3}(?<dateOfBirth>[0-9]{6})[0-9][FM<](?<expirationDate>[0-9]{6})[0-9]")
    val match = regex.find(joined)
    return if (match != null) {
        MRZFields(
            documentNumber = match.groups["documentNumber"]?.value,
            dateOfBirth = match.groups["dateOfBirth"]?.value,
            expirationDate = match.groups["expirationDate"]?.value
        )
    } else {
        MRZFields(null, null, null)
    }
}

fun joinMrzFromTextBlocks(result: com.google.mlkit.vision.text.Text): String {
    var fullRead = ""
    for (block in result.textBlocks) {
        var temp = ""
        for (line in block.lines) {
            temp += line.text + "-"
        }
        temp = temp.replace("\r", "")
            .replace("\n", "")
            .replace("\t", "")
            .replace(" ", "")
        fullRead += "$temp-"
    }
    return fullRead.uppercase()
}

object DatasetUtils {
    /**
     * Traverse the dataset directory and return a list of (country, jpgUri, txtUri?) pairs.
     * If a .txt is missing, txtUri will be null. If a .jpg is missing, the pair is skipped.
     */
    fun findImageTextPairs(context: Context, rootUri: Uri): List<Triple<String, Uri, Uri?>> {
        val pairs = mutableListOf<Triple<String, Uri, Uri?>>()
        val contentResolver = context.contentResolver
        val children = DocumentFile.fromTreeUri(context, rootUri)?.listFiles() ?: return pairs
        for (countryDir in children) {
            if (!countryDir.isDirectory) continue
            val jpgs = countryDir.listFiles().filter { it.name?.endsWith(".jpg", true) == true }
            val txts = countryDir.listFiles().filter { it.name?.endsWith(".txt", true) == true }
            val txtMap = txts.associateBy { it.name?.substringBeforeLast('.') }
            for (jpg in jpgs) {
                val baseName = jpg.name?.substringBeforeLast('.') ?: continue
                val txt = txtMap[baseName]
                pairs.add(Triple(countryDir.name ?: "", jpg.uri, txt?.uri))
            }
        }
        return pairs
    }
}

// Helper function to extract MRZ block from recognized text
fun extractMRZ(text: String): String? {
    val lines = text.lines().map { it.trim() }.filter { it.isNotEmpty() }
    // Try to find two consecutive lines of 44 chars (TD3 passport)
    for (i in 0 until lines.size - 1) {
        if (lines[i].length == 44 && lines[i + 1].length == 44) {
            return lines[i] + "\n" + lines[i + 1]
        }
    }
    // Fallback: try regexes
    val regexes = listOf(
        Regex("([A-Z0-9<]{9}[0-9ILDSOG]{1}[A-Z<]{3}[0-9ILDSOG]{6}[0-9ILDSOG]{1}[FM<]{1}[0-9ILDSOG]{6}[0-9ILDSOG]{1})"),
        Regex("\\bIP[A-Z<]{3}[A-Z0-9<]{9}[0-9]{1}"),
        Regex("[0-9]{6}[0-9]{1}[FM<]{1}[0-9]{6}[0-9]{1}[A-Z<]{3}")
    )
    for (regex in regexes) {
        val match = regex.find(text.replace("\n", ""))
        Log.d("MRZ", "Found match: ${match?.value}")
        if (match != null) return match.value
    }
    return null
}

class MainActivity : AppCompatActivity() {
    private lateinit var resultsAdapter: ResultsAdapter
    private val results = mutableListOf<MRZResult>()
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
                exportLauncher.launch("mrz_results.csv")
            }
        }
    }

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
                var expectedMRZ: String? = null
                var recognizedMRZ: String? = null
                var match = false
                var error: String? = null
                try {
                    if (txtUri == null) {
                        error = "Missing .txt file"
                    } else {
                        expectedMRZ = withContext(Dispatchers.IO) {
                            val inputStream = contentResolver.openInputStream(txtUri)
                            inputStream?.bufferedReader()?.use(BufferedReader::readText)
                        }?.trim()
                    }
                    val inputStream = contentResolver.openInputStream(jpgUri)
                    val bitmap = withContext(Dispatchers.IO) {
                        inputStream?.use { BitmapFactory.decodeStream(it) }
                    }
                    if (bitmap == null) {
                        error = "Failed to decode image"
                    } else {
                        // Crop to bottom 25% for MRZ region
                        val cropRatio = 0.25f
                        val cropHeight = (bitmap.height * cropRatio).toInt()
                        val yStart = bitmap.height - cropHeight
                        val mrzBitmap = Bitmap.createBitmap(bitmap, 0, yStart, bitmap.width, cropHeight)
                        val image = InputImage.fromBitmap(mrzBitmap, 0)
                        val result = withContext(Dispatchers.IO) {
                            recognizer.process(image).await()
                        }
                        recognizedMRZ = extractMRZ(result.text)?.trim()
                        Log.d("MRZ", "${jpgUri.lastPathSegment} MLKit raw text: ${result.text}")
                        // OcrUtils-style: join all lines from textBlocks, uppercase, extract fields with regex
                        val joined = joinMrzFromTextBlocks(result)
                        Log.d("MRZ", "${jpgUri.lastPathSegment} joined: $joined")
                        val detectedFields = extractMrzFieldsFromJoined(joined)
                        val expectedFields = extractMRZFields(expectedMRZ)
                        Log.d("MRZ", "${jpgUri.lastPathSegment} expectedFields: $expectedFields detectedFields: $detectedFields")
                        match = expectedFields.documentNumber != null &&
                                expectedFields.documentNumber == detectedFields.documentNumber &&
                                expectedFields.dateOfBirth == detectedFields.dateOfBirth &&
                                expectedFields.expirationDate == detectedFields.expirationDate
                    }
                } catch (e: Exception) {
                    error = e.message
                }
                results.add(
                    MRZResult(
                        country = country,
                        filename = jpgUri.lastPathSegment ?: "",
                        expectedMRZ = expectedMRZ,
                        recognizedMRZ = recognizedMRZ,
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

    private fun exportResultsToCsv(uri: Uri) {
        try {
            val outputStream = contentResolver.openOutputStream(uri)
            val writer = OutputStreamWriter(outputStream)
            writer.write("country,filename,status,expectedMRZ,recognizedMRZ\n")
            for (result in results) {
                val status = when {
                    result.error != null -> "Failed: ${result.error.replace(",", " ")}"
                    result.match -> "Matched"
                    else -> "Mismatched"
                }
                writer.write("${result.country},${result.filename},${status},\"${result.expectedMRZ?.replace("\"", "'") ?: ""}\",\"${result.recognizedMRZ?.replace("\"", "'") ?: ""}\"\n")
            }
            writer.flush()
            writer.close()
            Toast.makeText(this, "Exported successfully", Toast.LENGTH_SHORT).show()
        } catch (e: Exception) {
            Toast.makeText(this, "Export failed: ${e.message}", Toast.LENGTH_LONG).show()
        }
    }

    class ResultsAdapter(private val items: List<MRZResult>) : RecyclerView.Adapter<ResultsAdapter.ViewHolder>() {
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
                val expectedFields = extractMRZFields(item.expectedMRZ)
                val detectedFields = extractMRZFields(item.recognizedMRZ)
                val message = buildString {
                    append("Country: ${item.country}\n")
                    append("Filename: ${item.filename}\n")
                    append("Status: $status\n\n")
                    append("Expected MRZ:\n${item.expectedMRZ ?: "-"}\n\n")
                    append("Detected MRZ:\n${item.recognizedMRZ ?: "-"}\n\n")
                    append("Expected Fields:\n")
                    append("  Document Number: ${expectedFields.documentNumber ?: "-"}\n")
                    append("  Date of Birth: ${expectedFields.dateOfBirth ?: "-"}\n")
                    append("  Expiry: ${expectedFields.expirationDate ?: "-"}\n\n")
                    append("Detected Fields:\n")
                    append("  Document Number: ${detectedFields.documentNumber ?: "-"}\n")
                    append("  Date of Birth: ${detectedFields.dateOfBirth ?: "-"}\n")
                    append("  Expiry: ${detectedFields.expirationDate ?: "-"}\n")
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

// Extension function to await ML Kit Task in coroutines
suspend fun <T> com.google.android.gms.tasks.Task<T>.await(): T =
    suspendCancellableCoroutine { cont ->
        addOnSuccessListener { cont.resume(it) {} }
        addOnFailureListener { cont.resumeWithException(it) }
    }