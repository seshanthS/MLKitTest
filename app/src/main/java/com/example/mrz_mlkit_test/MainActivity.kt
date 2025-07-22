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
import android.graphics.Canvas
import android.graphics.ColorMatrix
import android.graphics.ColorMatrixColorFilter
import android.graphics.Paint

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

/**
 * Extracts key MRZ fields (document number, date of birth, expiration date) from a raw MRZ string.
 * Supports TD3 passport format with two lines, where the second line contains the field data.
 * 
 * @param mrz The raw MRZ string, potentially containing multiple lines
 * @return MRZFields object containing extracted document number, date of birth, and expiration date
 * 
 * Implementation details:
 * - For TD3 format: expects 2+ lines, extracts from second line at specific positions
 * - For single line format: treats as TD3 line 2 if 27+ characters
 * - Uses fixed character positions: docNum(0-8), dateOfBirth(13-18), expirationDate(21-26)
 * - Returns null fields if format doesn't match expected patterns
 */
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

/**
 * Cleans MRZ field data by correcting common OCR recognition errors.
 * Replaces frequently misrecognized characters with their correct numeric equivalents.
 * 
 * @param field The raw field string from OCR recognition
 * @return Cleaned field string with character corrections applied, or null if input is null
 * 
 * Character corrections applied:
 * - 'O' → '0' (letter O to zero)
 * - 'I' → '1' (letter I to one)  
 * - 'L' → '1' (letter L to one)
 * - 'D' → '0' (letter D to zero)
 * - 'S' → '5' (letter S to five)
 * - 'G' → '6' (letter G to six)
 */
fun cleanMRZField(field: String?): String? {
    return field?.replace('O', '0')
        ?.replace('I', '1')
        ?.replace('L', '1')
        ?.replace('D', '0')
        ?.replace('S', '5')
        ?.replace('G', '6')
}

/**
 * Normalizes MRZ text by removing all whitespace, control characters, and converting to uppercase.
 * Prepares text for pattern matching and field extraction by creating a continuous string.
 * 
 * @param text Raw text input containing potential MRZ data
 * @return Normalized uppercase string with all spacing and control characters removed
 * 
 * Removes: carriage returns (\r), newlines (\n), tabs (\t), and spaces
 */
fun joinMrzText(text: String): String {
    return text
        .replace("\r", "")
        .replace("\n", "")
        .replace("\t", "")
        .replace(" ", "")
        .uppercase()
}

/**
 * Extracts MRZ fields from a joined (continuous) MRZ string using regex pattern matching.
 * Specifically designed for TD3 passport format with precise field positioning.
 * 
 * @param joined Continuous MRZ string without spaces or line breaks
 * @return MRZFields object with extracted fields, or null fields if no match found
 * 
 * Regex pattern breakdown:
 * - Document number: 9 characters [A-Z0-9<]
 * - Check digit: 1 digit [0-9]
 * - Nationality: 3 characters [A-Z<]
 * - Date of birth: 6 digits [0-9]
 * - Check digit: 1 digit [0-9] 
 * - Sex: 1 character [FM<]
 * - Expiration date: 6 digits [0-9]
 * - Check digit: 1 digit [0-9]
 */
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

/**
 * Converts MLKit Text recognition result into a joined MRZ string by processing text blocks.
 * Iterates through all text blocks and lines, concatenating with delimiters for parsing.
 * 
 * @param result MLKit Text recognition result containing detected text blocks and lines
 * @return Uppercase string with all text joined, using hyphens as delimiters between lines/blocks
 * 
 * Processing steps:
 * 1. Iterate through each text block in the result
 * 2. For each block, concatenate all line texts with "-" separator
 * 3. Remove whitespace and control characters from each block
 * 4. Join all blocks with "-" separator and convert to uppercase
 */
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
     * Traverses a dataset directory structure to find matching image-text file pairs.
     * Scans country subdirectories for .jpg images and their corresponding .txt ground truth files.
     * 
     * @param context Android context for accessing content resolver
     * @param rootUri URI of the root dataset directory selected by user
     * @return List of triples containing (country_name, jpg_uri, txt_uri_or_null)
     * 
     * Directory structure expected:
     * - Root/
     *   - CountryA/
     *     - image1.jpg
     *     - image1.txt (optional)
     *   - CountryB/
     *     - image2.jpg
     *     - image2.txt (optional)
     * 
     * Behavior:
     * - Skips entries where .jpg file is missing
     * - Includes entries with missing .txt file (txtUri will be null)
     * - Matches files by base filename (before extension)
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

/**
 * Extracts MRZ block from recognized text using multiple detection strategies.
 * Attempts to find valid MRZ patterns using line length matching and regex patterns.
 * 
 * @param text Raw text string from OCR recognition
 * @return Extracted MRZ string if found, null if no valid MRZ pattern detected
 * 
 * Detection strategies (in order):
 * 1. Line-based: Look for two consecutive 44-character lines (TD3 passport standard)
 * 2. Regex patterns: Try multiple patterns for different MRZ components:
 *    - Full TD3 line 2 pattern with document number, nationality, dates
 *    - Document type and issuing country pattern
 *    - Date and nationality pattern
 * 
 * Character set: A-Z, 0-9, <, with specific positions for check digits
 */
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

/**
 * Converts a color bitmap to grayscale for improved OCR performance.
 * Uses ColorMatrix to apply standard grayscale conversion weights.
 * 
 * @param original The original color bitmap to convert
 * @return New bitmap converted to grayscale, preserving original dimensions
 * 
 * Conversion formula:
 * - Red channel: 0.299 weight
 * - Green channel: 0.587 weight  
 * - Blue channel: 0.114 weight
 * - Maintains same pixel dimensions and density as original
 */
fun toGrayscale(original: Bitmap): Bitmap {
    val grayscale = Bitmap.createBitmap(original.width, original.height, Bitmap.Config.ARGB_8888)
    val canvas = Canvas(grayscale)
    val paint = Paint()
    val colorMatrix = ColorMatrix()
    colorMatrix.setSaturation(0f) // Remove all color saturation
    val filter = ColorMatrixColorFilter(colorMatrix)
    paint.colorFilter = filter
    canvas.drawBitmap(original, 0f, 0f, paint)
    return grayscale
}

/**
 * Applies Otsu thresholding to create a binary image with black text on white background.
 * Automatically determines the optimal threshold value to separate foreground from background.
 * 
 * @param grayscaleBitmap Input grayscale bitmap
 * @return Binary bitmap with black text (0) and white background (255)
 * 
 * Algorithm:
 * 1. Calculate histogram of grayscale values
 * 2. Find threshold that minimizes intra-class variance (Otsu's method)
 * 3. Apply threshold: pixels below threshold become black (text), above become white (background)
 */
fun applyOtsuThreshold(grayscaleBitmap: Bitmap): Bitmap {
    val width = grayscaleBitmap.width
    val height = grayscaleBitmap.height
    val pixels = IntArray(width * height)
    grayscaleBitmap.getPixels(pixels, 0, width, 0, 0, width, height)
    
    // Calculate histogram
    val histogram = IntArray(256)
    for (pixel in pixels) {
        val gray = android.graphics.Color.red(pixel) // Since it's grayscale, R=G=B
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
        wB += histogram[i] // Weight background
        if (wB == 0) continue
        
        val wF = total - wB // Weight foreground
        if (wF == 0) break
        
        sumB += i * histogram[i]
        val mB = sumB / wB // Mean background
        val mF = (sum - sumB) / wF // Mean foreground
        
        // Between class variance
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
        // Text (darker) = black, Background (lighter) = white
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
 * Converts color bitmap to grayscale and applies Otsu thresholding in one step.
 * Creates binary image optimized for OCR with black text on white background.
 * 
 * @param original The original color bitmap
 * @return Binary bitmap with optimal text-background separation
 */
fun toGrayscaleWithOtsuThreshold(original: Bitmap): Bitmap {
    val grayscale = toGrayscale(original)
    return applyOtsuThreshold(grayscale)
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

    /**
     * Initializes the main activity UI and sets up event handlers for dataset processing.
     * Configures RecyclerView for results display, directory picker, and CSV export functionality.
     * 
     * @param savedInstanceState Bundle containing activity's previously saved state
     * 
     * UI Components initialized:
     * - Summary TextViews: total, matched, mismatched, failed counts
     * - Directory selection button with document tree picker
     * - Results RecyclerView with custom adapter
     * - CSV export button with create document picker
     * 
     * Event handlers:
     * - Directory picker: launches processDataset() when directory selected
     * - Export button: creates CSV file with results data
     * - Dynamic summary updates: refreshes counts after each processing step
     */
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

    /**
     * Processes an entire MRZ dataset by iterating through image-text pairs and performing OCR analysis.
     * Compares MLKit text recognition results against ground truth MRZ data for accuracy evaluation.
     * 
     * @param rootUri URI of the selected dataset root directory
     * @param updateSummary Callback function to refresh UI summary statistics
     * 
     * Processing pipeline:
     * 1. Discover all image-text pairs in dataset directory structure
     * 2. For each image:
     *    - Load expected MRZ from .txt file (if available)
     *    - Crop image to bottom 25% (typical MRZ region)
     *    - Run MLKit text recognition on cropped region
     *    - Extract MRZ fields using multiple strategies
     *    - Compare detected fields with expected values
     *    - Store results with match status and error information
     * 3. Update UI progressively as each image is processed
     * 
     * Error handling: Catches and logs all exceptions, marking failed cases with error messages
     * Performance: Runs on background thread with UI updates on main thread
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
                        //Maintain MRZ H:W ratio close to standard (e.g., 7:1 or 8:1 width:height).
                        //Resize MRZ region to 880×120 (or similar) so each char box is ≥ 60×20 px.

                        val cropRatio = 0.25f
                        val cropHeight = (bitmap.height * cropRatio).toInt()
                        val yStart = bitmap.height - cropHeight
                        val croppedBitmap = Bitmap.createBitmap(bitmap, 200, yStart, bitmap.width - 200, cropHeight)
                        val mrzBitmap = toGrayscaleWithOtsuThreshold(croppedBitmap)
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
                        //todo: replace extractMRZField
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

    /**
     * Exports processing results to a CSV file for external analysis and record keeping.
     * Creates a structured CSV with columns for country, filename, status, and MRZ data.
     * 
     * @param uri URI of the target CSV file selected by user through document picker
     * 
     * CSV structure:
     * - Headers: country, filename, status, expectedMRZ, recognizedMRZ  
     * - Status values: "Matched", "Mismatched", "Failed: [error_message]"
     * - MRZ fields: wrapped in quotes with escaped quotes converted to apostrophes
     * - Error handling: commas in error messages replaced with spaces to preserve CSV structure
     * 
     * File operations:
     * - Uses content resolver for cross-platform file access
     * - Handles write permissions through Android's storage access framework
     * - Shows toast notifications for success/failure feedback
     */
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
        
        /**
         * Creates new ViewHolder instances for RecyclerView items.
         * Inflates the item layout and initializes TextView references for data binding.
         * 
         * @param parent The parent ViewGroup that will contain the new view
         * @param viewType The view type identifier (unused in this implementation)
         * @return New ViewHolder instance with inflated item layout
         */
        override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
            val v = LayoutInflater.from(parent.context).inflate(R.layout.item_result, parent, false)
            return ViewHolder(v)
        }
        
        /**
         * Binds MRZ processing result data to ViewHolder views and sets up click interactions.
         * Displays country, filename, and color-coded status with detailed popup on click.
         * 
         * @param holder ViewHolder containing the views to populate with data
         * @param position Position of the item in the results list
         * 
         * Status display:
         * - "Failed": Red color for processing errors
         * - "Matched": Green color for successful field matches  
         * - "Mismatched": Orange color for recognition but incorrect field values
         * 
         * Click interaction:
         * - Shows AlertDialog with detailed comparison of expected vs detected MRZ fields
         * - Includes raw MRZ strings, extracted field values, and error messages
         * - Provides comprehensive debugging information for analysis
         */
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
        
        /**
         * Returns the total number of items in the results dataset.
         * Required by RecyclerView.Adapter for proper list management and scrolling.
         * 
         * @return Size of the items list
         */
        override fun getItemCount() = items.size
    }
}

/**
 * Extension function that converts Google Play Services Task to Kotlin coroutine suspend function.
 * Enables await-style asynchronous programming with MLKit APIs that return Task objects.
 * 
 * @param T The type parameter of the Task result
 * @return The result of type T when the task completes successfully
 * @throws Exception if the task fails, the exception is propagated to the coroutine
 * 
 * Implementation:
 * - Uses suspendCancellableCoroutine to bridge callback-based Task API with coroutines
 * - Registers success/failure listeners that resume the coroutine appropriately
 * - Handles cancellation properly for coroutine lifecycle management
 * 
 * Usage: Allows calling `recognizer.process(image).await()` instead of callback-based approach
 */
suspend fun <T> com.google.android.gms.tasks.Task<T>.await(): T =
    suspendCancellableCoroutine { cont ->
        addOnSuccessListener { cont.resume(it) {} }
        addOnFailureListener { cont.resumeWithException(it) }
    }