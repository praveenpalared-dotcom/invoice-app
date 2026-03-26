# 🧾 Automated Invoice Processing & Anomaly Detection

A fully local ML pipeline that:
- Pre-processes invoice images (OpenCV)
- Extracts text via OCR (Tesseract)
- Parses structured fields with Regex
- Detects anomalies using Z-scores + Isolation Forest
- Exports annotated CSV + visualisation charts

---

## 📁 Project Structure

```
invoice_pipeline/
├── generate_sample_invoices.py   # Creates test invoice images
├── preprocessing.py              # Image cleanup (grayscale, denoise, binarize)
├── ocr_engine.py                 # Tesseract OCR wrapper
├── parser.py                     # Regex field extraction → DataFrame
├── anomaly_detector.py           # Z-score + Isolation Forest + duplicate detection
├── main.py                       # Orchestrator + visualisations
├── requirements.txt
├── sample_invoices/              # Generated test images go here
└── output/                       # CSV results + charts saved here
```

---

## ⚙️ Setup (Step-by-Step)

### Step 1 — Install Tesseract OCR (system dependency)

**Ubuntu / Debian / WSL:**
```bash
sudo apt-get update && sudo apt-get install -y tesseract-ocr
```

**macOS:**
```bash
brew install tesseract
```

**Windows:**
1. Download the installer from:
   https://github.com/UB-Mannheim/tesseract/wiki
2. Install to `C:\Program Files\Tesseract-OCR\`
3. Set the environment variable:
   ```
   set TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
   ```
   Or add the Tesseract folder to your system PATH.

Verify Tesseract works:
```bash
tesseract --version
```

---

### Step 2 — Create a Python virtual environment

```bash
# In VS Code terminal, inside the invoice_pipeline/ folder
python -m venv .venv

# Activate it:
# Linux / macOS
source .venv/bin/activate
# Windows
.venv\Scripts\activate
```

---

### Step 3 — Install Python dependencies

```bash
pip install -r requirements.txt
```

---

### Step 4 — Generate sample invoice images

```bash
python generate_sample_invoices.py
```

This creates **36 invoice PNG images** in `sample_invoices/` with injected
anomalies (amount spikes, future dates, old dates, math errors, duplicates).

---

### Step 5 — Run the full pipeline

```bash
python main.py
```

Or point it at your own image folder:
```bash
python main.py path/to/your/invoices/
```

---

## 📊 Outputs (in `output/`)

| File | Description |
|---|---|
| `invoice_results.csv` | All extracted fields + `is_anomaly` + `anomaly_reason` |
| `boxplot_amounts.png` | Per-vendor box plot with anomalies in red |
| `scatter_zscore.png` | Z-score coloured scatter plot |
| `anomaly_reasons.png` | Bar chart of anomaly type counts |

---

## 🔬 Testing Individual Modules

```bash
# Test preprocessing only
python preprocessing.py sample_invoices/INV-20240001.png

# Test OCR only (prints raw text + confidence)
python ocr_engine.py sample_invoices/INV-20240001.png

# Test parsing only (prints extracted fields)
python parser.py sample_invoices/INV-20240001.png
```

---

## 🧠 Anomaly Detection Logic

| Strategy | Method | Triggers when… |
|---|---|---|
| Math error | Rule-based | `grand_total ≠ Σ(line_totals)` |
| Future date | Rule-based | `invoice_date > today` |
| Old date | Rule-based | `invoice_date < today − 90 days` |
| Amount spike | Z-score (per vendor) | `|Z| > 3.0` |
| Multivariate outlier | Isolation Forest | Deviates in feature space |
| Duplicate payment | Fuzzy match | Same amount + similar invoice number |

---

## 🐛 Troubleshooting

| Problem | Fix |
|---|---|
| `TesseractNotFoundError` | Install Tesseract (Step 1) and ensure it is on PATH |
| `No module named 'cv2'` | Run `pip install opencv-python` |
| Low OCR accuracy | Ensure images are high-resolution (≥300 DPI); use `debug=True` in `preprocess()` |
| Windows path issues | Set `TESSERACT_CMD` env var to the full `.exe` path |
