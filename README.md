# 🛡️ Hardhat & Safety-Vest Compliance Detection for Construction Workers Using YOLOv8

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Nano%20(3.14M%20params)-orange.svg)](https://github.com/ultralytics/ultralytics)
[![Streamlit](https://img.shields.io/badge/Streamlit-App%20Live-FF4B4B.svg)](https://streamlit.io/)
[![Unit Tests](https://img.shields.io/badge/pytest-68%2F68%20passing-brightgreen.svg)](tests/)
[![License: CC BY 4.0](https://img.shields.io/badge/Dataset-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

> **Course:** AI2002 — Artificial Intelligence (Path A: Software-Based Implementation)  
> **Repository:** [github.com/mibrahimshoaib56/ppe-compliance](https://github.com/mibrahimshoaib56/ppe-compliance)  
> **Main Developer:** Muhammad Ibrahim Shoaib  

---

## 📌 Project Overview

Occupational head and pedestrian-strike hazards account for over 60% of construction site fatalities. While personal protective equipment (PPE) like **industrial hardhats** and **high-visibility safety vests** is legally mandatory (OSHA/HSE), manual safety walkthrough audits suffer from intermittent coverage and human fatigue.

This project delivers an automated, real-time safety surveillance system that bridges the gap between deep-learning perception and deterministic compliance decisions:
1. **Core Deep Learning Detector:** Ultralytics **YOLOv8 Nano** (130 layers, 3.14M parameters) trained on the **Construction Site Safety (v28)** dataset across 10 classes, achieving a validation $\text{mAP}_{50}$ of **0.813** and test $\text{mAP}_{50}$ of **0.754**.
2. **Deterministic Spatial Rule Engine:** A pure-Python compliance module that partitions workers into anatomical head (top 0%–35%) and widened torso (middle 15%–75%) zones, matching equipment using **Intersection over Gear Area ($\text{IoG} \ge 0.30$)** with greedy bipartite disambiguation.
3. **Multi-Criteria Reliability Safeguards:** Defensively filters out background clutter, tiny distant figures, crouching postures, and overhead perspectives ($\text{aspect} < 1.2$), routing ambiguous cases to `"Needs Review"` rather than falsely penalizing workers.
4. **CCTV Video Engine with Temporal Smoothing:** Sequential video decoder featuring a **7-frame rolling median filter** to eliminate transient 1–2 frame detection flicker without tracking drift.
5. **Interactive Industrial Web Portal:** Multi-page Streamlit dashboard supporting static image uploads, live CCTV video playback, YouTube clip ingestion, KPI cards, and downloadable CSV audit trails.

---

## 🏗️ Pipeline Architecture

```
       RAW IMAGE / CCTV FRAME / YOUTUBE CLIP
                        │
                        ▼
       ┌─────────────────────────────────┐
       │     YOLOv8n Object Detector     │  Base Conf = 0.10 (Prevents NMS drops)
       │    (130 Layers, 3.14M Params)   │  10 Target & Context Classes
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
       │ Temporal Smoothing & Reporting  │  • 7-Frame Rolling Median (Anti-flicker)
       │ (CCTV Video Stream / Dashboard) │  • Site Safety Rate % (Safe/Warn/Crit)
       └─────────────────────────────────┘  • CSV Audit Log & Risk Moments Gallery
```

A publication-grade diagram is available in [`eval/results/pipeline_architecture_diagram.png`](eval/results/pipeline_architecture_diagram.png).

---

## 📂 Project Structure

```text
.
├── app.py                      # Multi-page industrial Streamlit web application
├── requirements.txt            # Python dependencies (pip install -r requirements.txt)
├── README.md                   # Project documentation and quickstart guide
├── REPORT.md                   # Comprehensive final project report (AI2002 deliverable)
├── SLIDES.md                   # 12-slide presentation deck with speaker notes
├── pytest.ini                  # Pytest configuration
│
├── src/                        # Modular source code
│   ├── detector.py             # YOLOv8 inference wrapper & two-stage confidence filtering
│   ├── compliance.py           # Pure-Python spatial compliance rules & reliability guards
│   ├── video.py                # CCTV video pipeline, 7-frame median smoothing & MP4 export
│   └── utils.py                # Color-coded bounding box annotation & visual helpers
│
├── models/
│   └── best.pt                 # Trained YOLOv8n weights (3.14M parameters)
│
├── notebooks/
│   └── yolov8-safety-helmet-detection.ipynb  # Kaggle GPU training notebook deliverable
│
├── tests/                      # Automated unit test suite (68 tests)
│   ├── test_compliance.py      # Spatial zones, overlap logic, reliability guards
│   ├── test_video.py           # Video summary, resizing, temporal smoothing, MP4 writer
│   └── test_youtube.py         # YouTube URL parsing and CCTV timestamp clamping
│
├── scripts/
│   ├── generate_pipeline_diagram.py  # Generates high-res PNG/SVG architecture diagram
│   └── build_submission_zip.py       # Packages deliverables into submission ZIP
│
├── sample_images/              # Curated construction images for quick manual UI testing
│
└── eval/                       # Evaluation benchmarks, ground truth, and outputs
    ├── ground_truth.csv        # Human ground-truth counts for test photos
    ├── slider_check.py         # Confidence slider diagnostics (0.15, 0.25, 0.40)
    ├── score_images.py         # Pipeline scoring against ground truth
    ├── test_photos/            # 15 real-world evaluation photos (photo01 - photo15)
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
* **Python 3.10+** (Tested on Python 3.12)
* Windows, Linux, or macOS (Runs entirely on CPU; no GPU required)

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

The test suite contains **68 isolated unit tests** verifying spatial math, compliance rules, temporal smoothing, and URL parsers with **zero external model dependencies**:

```bash
python -m pytest -v
```
*Expected Result:* `68 passed in ~1.5s` (100% pass rate).

---

## 📊 Running Evaluation & Benchmark Diagnostics

### 1. Confidence Slider Sweep (0.15, 0.25, 0.40)
Runs the detector and compliance engine across three confidence thresholds, breaking down workers, compliance rates, and the 7 "Needs Review" diagnostic reasons:
```bash
python eval/slider_check.py eval/test_photos
```
*Outputs saved to:* `eval/results/slider_check.csv`.

### 2. Ground-Truth Scoring Against Real-World Benchmark
Scoring against human-audited ground-truth counts (`true_compliant`, `true_violations`):
```bash
python eval/score_images.py eval/test_photos eval/ground_truth.csv
```
*Outputs saved to:* `eval/results/score_images.csv` and `eval/results/score_images_report.txt`.

### 3. Regenerate Architecture Diagram
```bash
python scripts/generate_pipeline_diagram.py
```
*Outputs saved to:* `eval/results/pipeline_architecture_diagram.png` (300 DPI) and `.svg`.

---

## 📈 Key Empirical Results

### 1. Model Training Benchmark (Kaggle GPU, 100 Epochs)
* **Validation $\text{mAP}_{50}$:** **0.813** (81.3%)
* **Held-Out Test $\text{mAP}_{50}$:** **0.754** | **Test $\text{mAP}_{50-95}$:** **0.465**
* **Precision / Recall:** **0.874** / **0.702**
* **Weakest Class:** `NO-Hardhat` ($\text{AP}_{50} = 0.545$) due to small pixel footprint and variable hair/caps.

### 2. System-Level Evaluation on 15 Real-World Photos (88 True Workers)
* **Within $\pm 1$ Violation Accuracy:** **93.3%** (14 / 15 images)
* **Mean Absolute Error (Violations):** **0.60 workers**
* **Violations Caught (Upper Bound):** **71.4%** (5 / 7 true violations)
* **Needs-Review Composition:**
  * Small / distant background figures ($< 12\%$ height): **53.1%** (17 workers)
  * Conflicting detections: **18.8%** (6 workers)
  * Low confidence: **12.5%** (4 workers)
  * View not upright (overhead / crouching): **12.5%** (4 workers)
  * Vest unconfirmed: **3.1%** (1 worker)

---

## 📦 Project Deliverables Index

* 📄 **Comprehensive Report:** [`REPORT.md`](REPORT.md) (426 lines covering problem justification, algorithm theory, empirical results, and failure modes).
* 📑 **Presentation Slide Deck:** [`SLIDES.md`](SLIDES.md) (12 Marp-compatible presentation slides with complete speaker notes).
* 📓 **Kaggle GPU Notebook:** [`notebooks/yolov8-safety-helmet-detection.ipynb`](notebooks/yolov8-safety-helmet-detection.ipynb).
* 📐 **Architecture Diagram:** [`eval/results/pipeline_architecture_diagram.png`](eval/results/pipeline_architecture_diagram.png).
* 🔍 **Failure Case Studies:** [`eval/results/failure_cases/`](eval/results/failure_cases/).
* ⚙️ **Constants & Tuning Log:** [`eval/results/tuning_log.md`](eval/results/tuning_log.md).

---

## 📜 Dataset & License Attribution

* **Dataset:** Construction Site Safety (v28), hosted on Roboflow Universe. Used under the Creative Commons Attribution 4.0 International ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)) license.
* **Model Framework:** [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) (AGPL-3.0 License).
* **Course Context:** Developed for academic evaluation under AI2002.
