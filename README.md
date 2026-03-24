# VeriSphere — Aadhaar Fraud Detection

A multi-stage AI-powered system for detecting fraudulent Aadhaar cards using computer vision, OCR, QR validation, and digital forensics.

![Python](https://img.shields.io/badge/Python-3.12-blue) ![Flask](https://img.shields.io/badge/Flask-3.x-lightgrey) ![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple)

---

## Features

- **YOLOv8 Detection** — Locates card regions (name, DOB, Aadhaar number, face, QR code) with bounding boxes
- **EasyOCR Text Extraction** — Reads demographic fields from detected regions
- **Verhoeff Checksum** — Mathematically validates the 12-digit Aadhaar number
- **QR Cross-Check** — Decodes embedded QR data and compares against OCR-extracted fields
- **Digital Forensics** — ELA (Error Level Analysis), noise uniformity, and sharpness consistency checks
- **Fraud Scoring** — Weighted scoring engine produces a final verdict: Genuine / Suspicious / Fake

---

## Tech Stack

| Layer | Technology |
|---|---|
| Object Detection | YOLOv8 (Ultralytics) |
| OCR | EasyOCR |
| QR Decoding | pyzbar / opencv |
| Checksum | Verhoeff Algorithm |
| Forensics | OpenCV (ELA, noise, sharpness) |
| Web Server | Flask |
| Frontend | Tailwind CSS (dark theme) |

---

## Project Structure

```
├── aadhaar_pipeline/
│   ├── pipeline.py        # Main orchestration
│   ├── detector.py        # YOLOv8 inference
│   ├── ocr.py             # EasyOCR extraction + preprocessing
│   ├── qr_validation.py   # QR decode + field comparison
│   ├── tampering.py       # Forensics (ELA, noise, sharpness)
│   ├── validator.py       # Verhoeff checksum
│   ├── consistency.py     # Cross-field consistency checks
│   └── decision.py        # Fraud scoring + verdict
├── Templates/
│   ├── stitch 1/          # Upload / idle page
│   └── stitch 2/          # Results page design reference
├── flask_app.py           # Flask server + JS glue
├── app.py                 # Streamlit version (legacy)
└── requirements.txt
```

---

## Setup

```bash
# 1. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate      # Windows

# 2. Install dependencies
pip install -r aadhaar_pipeline/requirements.txt

# 3. Place your YOLOv8 weights
#    Copy aadhaar_best.pt into the project root

# 4. Run the Flask app
python flask_app.py
```

Open `http://localhost:5000` in your browser.

---

## How It Works

1. Upload a front-side Aadhaar card image (JPG/PNG/WEBP)
2. YOLOv8 detects and crops each field region
3. EasyOCR reads text from each crop
4. Verhoeff algorithm validates the Aadhaar number checksum
5. QR code is decoded and compared against OCR values
6. ELA + noise + sharpness forensics flag pixel-level anomalies
7. A weighted fraud score determines the final verdict

---

## Notes

- Model weights (`*.pt`, `*.h5`) are excluded from this repo via `.gitignore` due to file size
- The system works best with clear, well-lit front-side scans
- Secure-format QR codes (post-2018) do not store the full UID — this is handled gracefully
