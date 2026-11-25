# 🛡️ Hardhat & Safety-Vest Compliance Detection for Construction Workers Using YOLOv8

[![Python 3.12.10](https://img.shields.io/badge/python-3.12.10-blue.svg)](https://www.python.org/downloads/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Nano%20(3.01M%20params)-orange.svg)](https://github.com/ultralytics/ultralytics)
[![Streamlit](https://img.shields.io/badge/Streamlit-App%20Live-FF4B4B.svg)](https://streamlit.io/)
[![Unit Tests](https://img.shields.io/badge/pytest-68%20passed-brightgreen.svg)](tests/)
[![License: CC BY 4.0](https://img.shields.io/badge/Dataset-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

> **Course:** AI2002 — Artificial Intelligence (Path A: Software-Based Implementation)  
> **Repository:** [github.com/mibrahimshoaib56/ppe-compliance](https://github.com/mibrahimshoaib56/ppe-compliance)  
> **Main Developer:** Muhammad Ibrahim Shoaib  

---

## 📌 Project Overview

Safety gear like **hardhats** and **high-visibility safety vests** is required on construction sites to prevent head injuries and vehicular accidents. Manual monitoring by safety inspectors has limited coverage and depends on periodic visual inspections.

This project implements a compliance detection system combining deep-learning object detection with spatial geometric rule logic:
1. **Object Detection Model:** Ultralytics **YOLOv8n** (129 layers, 3,012,798 parameters as trained; 73 layers, 3,134,894 parameters when fused) trained on the **Construction Site Safety (v28)** dataset across 10 classes, achieving a validation $\text{mAP}_{50}$ of **0.813** and test $\text{mAP}_{50}$ of **0.754**.
2. **Spatial Rule Engine:** A Python compliance module that partitions workers into anatomical head (top 0%–35%) and widened torso (middle 15%–75%) zones, matching equipment using **Intersection over Gear Area ($\text{IoG} \ge 0.30$)** with greedy disambiguation.
3. **Reliability Safeguards:** Filters out background clutter, distant figures, crouching postures, and overhead perspectives ($\text{aspect} < 1.2$), routing ambiguous cases to `"Needs Review"`.
4. **Video Processing with Temporal Filtering:** Sequential video processing featuring a **7-frame rolling median filter** running in a post-processing pass over surrounding frames to reduce transient 1–2 frame detection flicker.
5. **Web Portal:** Streamlit interface supporting image uploads, video processing with annotated playback, YouTube clip processing, summary metrics, and downloadable CSV inspection records.

> **Note on Model Weights:** The trained model weights (`models/best.pt`) are not stored in the Git repository due to file size considerations; they are included in the project submission zip.

---

## 🏗️ Pipeline Architecture

```
       RAW IMAGE / CCTV FRAME / YOUTUBE CLIP
                        │
                        ▼
       ┌─────────────────────────────────┐
       │     YOLOv8n Object Detector     │  Base Conf = 0.10 (Prevents NMS drops)
       │ (129 Layers, 3.01M Params Trained) 10 Target & Context Classes
       └────────────────┬────────────────┘
                        │
                        ▼
       ┌─────────────────────────────────┐
       │   Two-Stage Confidence Filter   │  Person ≥ 0.30 | Hardhat ≥ 0.30
       │     (Independent Class Gates)   │  Negative Gear & Vests ≥ 0.20
       └────────────────┬────────────────┘
                        │
                        ▼
       ┌─────────────────────────────────┐
       │ Anatomical Zone Decomposition   │  Head Zone: Top 0% - 35%
       │   (Per Detected Person Box)     │  Torso Zone: Widened 15% - 75%
       └────────────────┬────────────────┘
                        │
                        ▼
       ┌─────────────────────────────────┐
       │ Spatial Disambiguation Matching │  IoG = Area(Gear ∩ Zone) / Area(Gear) ≥ 0.30
       │  (Intersection over Gear Area)  │  Greedy 1-to-1 Association
       └────────────────┬────────────────┘
                        │
                        ▼
       ┌─────────────────────────────────┐
       │  Multi-Criteria Quality Guards  │  • Aspect Ratio (h/w ≥ 1.2: Not overhead)
       │  (Reliability & Safeguard Check)│  • Resolution (height ≥ 12% frame height)
       └────────────────┬────────────────┘  • Confidence (conf ≥ 0.40) & Conflict Check
                        │
                        ▼
       ┌─────────────────────────────────┐
       │ Compliance Decision Resolution  │  ==> Compliant / Violation / Needs Review
       └────────────────┬────────────────┘
                        │
                        ▼
       ┌─────────────────────────────────┐
       │ Temporal Smoothing & Reporting  │  • 7-Frame Rolling Median (Post-processing)
       │ (CCTV Video Stream / Dashboard) │  • Site Safety Rate % (Safe/Warn/Crit)
       └─────────────────────────────────┘  • CSV Inspection Log & Risk Moments Gallery
```

An architecture diagram is available in [`eval/results/pipeline_architecture_diagram.png`](eval/results/pipeline_architecture_diagram.png).

---

## 📂 Project Structure

```text
.
├── app.py                      # Streamlit web application
├── requirements.txt            # Python dependencies (pip install -r requirements.txt)
├── README.md                   # Project documentation and quickstart guide
├── REPORT.md                   # Comprehensive final project report (AI2002 deliverable)
├── SLIDES.md                   # 12-slide presentation deck with speaker notes
├── pytest.ini                  # Pytest configuration
│
├── src/                        # Modular source code
│   ├── detector.py             # YOLOv8 inference wrapper & two-stage confidence filtering
│   ├── compliance.py           # Pure-Python spatial compliance rules & reliability guards
│   ├── video.py                # Video pipeline, 7-frame median smoothing & MP4 export
│   └── utils.py                # Color-coded bounding box annotation & visual helpers
│
├── models/
│   └── .gitkeep                # Placeholder; best.pt is provided in submission zip
│
├── notebooks/
│   └── yolov8-safety-helmet-detection.ipynb  # Kaggle GPU training notebook deliverable
│
├── tests/                      # Automated unit test suite (68 tests)
│   ├── test_compliance.py      # Spatial zones, overlap logic, reliability guards
│   ├── test_video.py           # Video summary, resizing, temporal smoothing, MP4 writer
│   └── test_youtube.py         # YouTube URL parsing and timestamp clamping
│
├── scripts/
│   ├── generate_pipeline_diagram.py  # Generates PNG/SVG architecture diagram
│   └── build_submission_zip.py       # Packages deliverables into submission ZIP
│
├── sample_images/              # Curated construction images for quick manual testing
│
└── eval/                       # Evaluation benchmarks, ground truth, and outputs
    ├── ground_truth.csv        # Ground-truth counts for test photos (visual inspection)
    ├── slider_check.py         # Confidence slider diagnostics (0.15, 0.25, 0.40)
    ├── score_images.py         # Pipeline scoring against ground truth
    ├── test_photos/            # 15 test photos from Google Images
    │   └── SOURCES.md          # Image origins, scenes, and labeling rules
    └── results/                # Evaluation outputs, curves, and reports
        ├── results.png         # 100-epoch training and validation loss curves
        ├── confusion_matrix.png# Model confusion matrix
        ├── tuning_log.md       # Complete record of every threshold and constant
        ├── score_images_report.txt  # Quantitative benchmark report
        └── failure_cases/      # Annotated images of overhead, bag, and plain vest cases
```

---

## 🚀 Quickstart & Setup

### 1. Prerequisites
* **Python 3.12.10** (Tested in the project virtual environment)
* Windows, Linux, or macOS

### 2. Clone and Setup Environment
```bash
# Clone the repository
git clone https://github.com/mibrahimshoaib56/ppe-compliance.git
cd ppe-compliance

# Create and activate virtual environment
python -m venv .venv

# Windows (Command Prompt / PowerShell):
.venv\Scripts\activate

# Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Launch the Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🧪 Running Unit Tests

The test suite contains **68 unit tests** verifying spatial math, compliance rules, temporal smoothing, and URL parsers:

```bash
python -m pytest -q
```
*Result:* `68 passed`.

---

## 📊 Running Evaluation & Benchmark Diagnostics

### 1. Confidence Slider Sweep (0.15, 0.25, 0.40)
Runs the detector and compliance engine across three confidence thresholds on the test photos:
```bash
python eval/slider_check.py eval/test_photos
```
*Outputs saved to:* `eval/results/slider_check.csv`.

### 2. Ground-Truth Scoring Against Test Photos
Scores pipeline predictions against visual ground-truth counts (`true_compliant`, `true_violations`):
```bash
python eval/score_images.py eval/test_photos eval/ground_truth.csv
```
*Outputs saved to:* `eval/results/score_images.csv` and `eval/results/score_images_report.txt`.

### 3. Regenerate Architecture Diagram
```bash
python scripts/generate_pipeline_diagram.py
```
*Outputs saved to:* `eval/results/pipeline_architecture_diagram.png` and `.svg`.

---

## 📈 Key Empirical Results

### 1. Model Training Benchmark (Kaggle GPU Tesla T4, 100 Epochs)
* **Training Time:** **0.565 hours**
* **Validation $\text{mAP}_{50}$ at Epoch 100:** **0.814** (Best weights `best.pt`: **0.813**, $\text{mAP}_{50-95} = 0.511$)
* **Held-Out Test $\text{mAP}_{50}$ (82 images):** **0.754** | **Test $\text{mAP}_{50-95}$:** **0.465**
* **Test Precision / Recall:** **0.874** / **0.702**
* **Test Speed on T4 GPU:** 1.8 ms preprocess, 6.8 ms inference, 2.3 ms postprocess per image (No CPU timing measured)
* **Per-Class Test Metrics (`test_metrics.txt`):**
  * Hardhat: $P = 0.959$, $R = 0.809$, $\text{AP}_{50} = 0.905$
  * Safety Vest: $P = 0.896$, $R = 0.849$, $\text{AP}_{50} = 0.902$
  * Person: $P = 0.884$, $R = 0.786$, $\text{AP}_{50} = 0.857$
  * NO-Safety Vest: $P = 0.930$, $R = 0.739$, $\text{AP}_{50} = 0.799$
  * NO-Hardhat: $P = 0.807$, $R = 0.561$, $\text{AP}_{50} = 0.545$

### 2. System-Level Evaluation on 15 Test Photos (88 True Workers)
*Evaluation photos collected from Google Images; labels made by the group by visual inspection.*

* **Exact Image-Level Match (Both Counts):** **13.3%** (2 of 15 images)
* **Within $\pm 1$ Accuracy (Violations):** **93.3%** (14 of 15 images)
* **Within $\pm 1$ Accuracy (Compliant):** **26.7%** (4 of 15 images)
* **Mean Absolute Error:** Violations = **0.60**, Compliant = **3.73**
* **"Predict Zero Violations" Baseline:** **80.0%** (12 of 15 images within $\pm 1$), because the ground truth has only 7 violations among 88 workers.
* **Worker Coverage:** 69 workers detected vs 88 true (counts only; detections were not matched one by one); 37 received a verdict (25 compliant, 12 violations); 32 were sent to "Needs Review".
* **Violations Flagged / Caught:** 12 flagged by pipeline vs 7 true, with at most 5 caught (sum of min(pred, true)).
* **Evaluation Limitation:** The evaluation set is small (15 photos with 88 workers and 7 true violations).

---

## 📦 Project Deliverables Index

* 📄 **Comprehensive Report:** [`REPORT.md`](REPORT.md)
* 📑 **Presentation Slide Deck:** [`SLIDES.md`](SLIDES.md)
* 📓 **Kaggle GPU Notebook:** [`notebooks/yolov8-safety-helmet-detection.ipynb`](notebooks/yolov8-safety-helmet-detection.ipynb)
* 📐 **Architecture Diagram:** [`eval/results/pipeline_architecture_diagram.png`](eval/results/pipeline_architecture_diagram.png)
* 🔍 **Failure Case Studies:** [`eval/results/failure_cases/`](eval/results/failure_cases/)
* ⚙️ **Tuning Log:** [`eval/results/tuning_log.md`](eval/results/tuning_log.md)

---

## 📜 Dataset & License Attribution

* **Dataset:** Construction Site Safety (v28), hosted on Roboflow Universe. Used under the Creative Commons Attribution 4.0 International ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)) license.
* **Model Framework:** [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) (AGPL-3.0 License).
* **Course Context:** Developed for academic evaluation under AI2002.
