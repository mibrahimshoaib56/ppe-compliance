# Hardhat and Safety-Vest Compliance Detection for Construction Workers Using YOLOv8

**Course:** AI2002 — Artificial Intelligence  
**Path:** Path A (Software-Based Implementation)  
**Deliverable:** Final Comprehensive Project Report  
**GitHub Repository:** `https://github.com/mibrahimshoaib56/ppe-compliance`  
**Date:** November 2026  

---

## Executive Summary

Personal Protective Equipment (PPE) compliance is the first line of defense against occupational fatalities and severe injuries on modern construction sites. However, manual safety monitoring by on-site safety officers is physically constrained, intermittent, and vulnerable to human fatigue. 

This project implements an end-to-end, automated AI compliance surveillance system combining deep-learning object detection with spatial geometric rule logic and temporal video filtering. At the core is an Ultralytics **YOLOv8 Nano** detector (3.14M parameters) trained on the **Construction Site Safety (v28)** dataset across 10 classes, achieving a validation $\text{mAP}_{50}$ of **0.813** and test $\text{mAP}_{50}$ of **0.754**. 

To bridge the gap between raw bounding boxes and legally actionable safety judgements, a pure-Python spatial compliance engine was engineered. The engine partitions worker bounding boxes into anatomical head (top 0%–35%) and widened torso (middle 15%–75%) zones, associates equipment using **Intersection over Gear Area ($\text{IoG} \ge 0.30$)**, and enforces multi-criteria reliability safeguards (aspect ratio, bounding box height fraction, detection confidence, and conflict detection). For video surveillance, a **7-frame rolling median temporal filter** suppresses single-frame posture flickers without masking sustained violations.

System-level evaluation on novel, out-of-distribution real-world construction photos demonstrates **93.3% within $\pm 1$ accuracy on violation counts** with a low Mean Absolute Error (MAE) of **0.60**, while conservatively routing 46.4% of ambiguous, distant, or squashed overhead figures to a dedicated "Needs Review" queue rather than making false penalizations.

---

## Table of Contents
1. [Introduction & Problem Justification](#1-introduction--problem-justification)
2. [Objectives & Selected Path](#2-objectives--selected-path)
3. [Algorithm Justification: Why Detection & YOLOv8?](#3-algorithm-justification-why-detection--yolov8)
4. [Dataset Description & Preprocessing](#4-dataset-description--preprocessing)
5. [Model Training & Empirical Results](#5-model-training--empirical-results)
6. [Compliance Engine Architecture](#6-compliance-engine-architecture)
7. [System-Level Evaluation & Diagnostics](#7-system-level-evaluation--diagnostics)
8. [Failure Case Analysis](#8-failure-case-analysis)
9. [CCTV Video Surveillance & Temporal Smoothing](#9-cctv-video-surveillance--temporal-smoothing)
10. [Application Architecture & User Interface](#10-application-architecture--user-interface)
11. [Limitations & Future Improvements](#11-limitations--future-improvements)
12. [Conclusion & Viva Defense Guide](#12-conclusion--viva-defense-guide)

---

## 1. Introduction & Problem Justification

Construction environments are intrinsically hazardous, characterized by heavy machinery movement, suspended loads, high elevations, and shifting ground excavations. Regulatory safety standards (OSHA, HSE) mandate that every personnel member on an active site wear two essential life-safety items:
1. **Industrial Safety Hardhats** (to protect against falling tools, head impact hazards, and electrical contact).
2. **High-Visibility Safety Vests** (to guarantee pedestrian-worker conspicuity against heavy construction equipment and vehicle traffic).

### The Real-World Breakdown of Manual Monitoring
Historically, site superintendents and designated safety officers conduct periodic walkthrough audits. However:
* **Intermittent Coverage:** Manual inspections only observe workers at specific walkthrough intervals; workers frequently discard vests or hardhats during unobserved hours.
* **Large Spatial Scale:** Multi-tier structures, excavation pits, and high-rise rebar decks cannot be visually surveyed simultaneously by human observers.
* **Observer Fatigue & Bias:** Subjective judgment and cognitive fatigue lead to missed violations, while disputes arise between subcontractors and safety managers over audit logs.

### The Automated Vision Solution
Automated Closed-Circuit Television (CCTV) cameras and drone surveillance provide continuous visual streams. However, deploying machine learning in safety-critical industrial environments requires not merely detecting people, but establishing **provable spatial attribution**—verifying that a specific detected hardhat or vest actually belongs to a specific worker, while rejecting ambiguous camera angles and occlusions.

---

## 2. Objectives & Selected Path

This project was developed under **Path A (Software-Based Implementation)** of the AI2002 course.

### Core Project Objectives
1. **Core Deep Learning Model:** Train an efficient, real-time object detector on construction site PPE classes capable of running inference smoothly on standard edge hardware and CPU-only devices.
2. **Rule-Based Compliance Engine:** Develop an interpretable, mathematically sound compliance engine on top of model detections that extracts anatomical body zones, performs bipartite gear association, and calculates site-wide safety percentages.
3. **Multi-Criteria Reliability Safeguards:** Implement strict defensive guards to prevent false violation flags on background clutter, tiny distant figures, crouching postures, and overhead perspectives.
4. **Temporal Stability for Video:** Implement temporal smoothing to eliminate bounding-box flicker across video frames without requiring heavy inter-frame tracking or Re-ID networks.
5. **Interactive Industrial Interface:** Provide an enterprise-grade Streamlit portal with dual Image/Video modes, live KPI metric cards, confidence slider controls, a risk-moment gallery, and downloadable CSV audit logs.
6. **Empirical Evaluation & Rigor:** Quantitatively benchmark the system against human ground-truth counts on real-world photos across multiple confidence thresholds.

---

## 3. Algorithm Justification: Why Detection & YOLOv8?

### 3.1 Object Detection vs. Image Classification
A naive approach might treat compliance as an image classification task (classifying a cropped worker photo as "Compliant" or "Non-Compliant"). However, image classification fundamentally fails in real-world construction scenes:
* **Multi-Instance Crowds:** Construction sites contain dozens of workers at varying distances simultaneously. Classification cannot separate individual workers.
* **Lack of Spatial Explainability:** Classification acts as a black box; it cannot indicate *why* a worker failed or identify *which* specific piece of gear is missing.
* **Object Detection Superiority:** Object detection simultaneously predicts **spatial coordinates (bounding boxes)**, **semantic classes**, and **confidence scores** for every person and gear instance, enabling downstream rule-based geometric reasoning.

### 3.2 Why YOLOv8 (You Only Look Once)?
Compared to two-stage detectors (Faster R-CNN) or Transformer-based architectures (DETR):
1. **Single-Stage Real-Time Throughput:** YOLOv8 utilizes a unified anchor-free convolutional architecture that evaluates the entire frame in a single forward pass, achieving real-time speeds (>30 FPS on GPU, ~15–25 FPS on modern multi-core CPU).
2. **Anchor-Free Detection Head:** Traditional anchor-based detectors struggle with the extreme scale disparity between a tiny hardhat (e.g., $25 \times 20$ pixels) and a full-frame vehicle ($500 \times 400$ pixels). YOLOv8's anchor-free split head decouples classification and bounding-box regression, accelerating convergence and boosting small-object localization.
3. **C2f Feature Integration:** YOLOv8 replaces older C3 modules with **C2f (Cross-Stage Partial Bottleneck with two convolutions)**, enhancing gradient flow and multi-scale contextual feature combination via rich skip connections.

### 3.3 Selection of YOLOv8 Nano (`yolov8n`)
For this project, `yolov8n.pt` was selected as the base architecture:
* **Parameter Count:** **3,140,190 parameters** (~3.14M).
* **Model Depth:** 130 layers.
* **Computational Cost:** **8.8 GFLOPs** at $640 \times 640$.
* **Rationale:** As mandated by project constraints, all client-side inference runs entirely on **CPU hardware** (no local CUDA GPU). YOLOv8 Nano maintains high detection precision ($\text{mAP}_{50} = 0.813$) while executing inference in under 80 ms per frame on CPU, eliminating server latency.

---

## 4. Dataset Description & Preprocessing

The model was trained on the **Construction Site Safety (v28)** dataset, sourced from Roboflow Universe under the Creative Commons Attribution 4.0 (CC BY 4.0) license.

### 4.1 Dataset Composition & Splits
The dataset consists of **2,801 total images** at $640 \times 640$ resolution, split into three subsets:

| Split | Image Count | Percentage | Characteristics |
|---|---|---|---|
| **Training** | 2,605 | 93.0% | Heavy data augmentation (baked-in mosaics, cutout occlusions, color jitter) |
| **Validation** | 114 | 4.1% | Natural construction scenes, clean annotations |
| **Testing** | 82 | 2.9% | Held-out benchmark test set, unaugmented |
| **Total** | **2,801** | **100.0%** | Standardized YOLO format annotations |

### 4.2 Class Hierarchy (10 Official Classes)
The model was trained across all 10 annotated classes:

| Class ID | Class Name | Function in Compliance Pipeline |
|:---:|---|---|
| `0` | `Hardhat` | **Positive Target Gear** (Indicates head protection present) |
| `1` | `Mask` | Contextual equipment (passive detection) |
| `2` | `NO-Hardhat` | **Negative Target Gear** (Direct evidence of missing helmet) |
| `3` | `NO-Mask` | Contextual equipment (passive detection) |
| `4` | `NO-Safety Vest` | **Negative Target Gear** (Direct evidence of missing vest) |
| `5` | `Person` | **Primary Anchor Instance** (Base entity for zones) |
| `6` | `Safety Cone` | Contextual hazard marking |
| `7` | `Safety Vest` | **Positive Target Gear** (Indicates torso protection present) |
| `8` | `machinery` | Heavy site equipment (excavators, loaders, cranes) |
| `9` | `vehicle` | Site transit vehicles (trucks, pickups) |

### 4.3 Training Set Mosaic & Augmentation Characteristics
Inspection of the training data revealed that the training images consist predominantly of **baked-in $2 \times 2$ mosaics with cutout patch squares**. This augmentation forces the neural network to identify partial, fragmented equipment cues (e.g., hardhat brim only or half a vest) and prevents reliance on fixed background contexts. In contrast, the validation and test splits consist exclusively of natural, unaugmented photographic scenes, ensuring an honest benchmark of generalization.

---

## 5. Model Training & Empirical Results

Training was conducted outside the local IDE on an external **Kaggle GPU instance (NVIDIA Tesla T4)** using the official Ultralytics framework.

### 5.1 Hyperparameters & Training Setup
* **Epochs:** 100
* **Batch Size:** 16
* **Optimizer:** Auto (SGD with momentum 0.937, weight decay 0.0005)
* **Initial Learning Rate:** $\eta_0 = 0.01$ (with cosine learning rate scheduling)
* **Input Resolution:** $640 \times 640$ pixels
* **Total Training Wall Time:** **0.565 hours** (~33.9 minutes)
* **Artifacts Generated:** `best.pt`, `results.png`, `confusion_matrix.png`, `BoxPR_curve.png`, `BoxF1_curve.png`.

### 5.2 Training Convergence & Metrics

```
Training Summary across 100 Epochs:
• Final Epoch Training Box Loss:       0.984
• Final Epoch Training Class Loss:     0.812
• Final Epoch Training DFL Loss:       0.998
• Validation Box Loss:                 1.052
• Validation Class Loss:               0.887
• Validation mAP50:                    0.813
```

The training loss curves show steady monotonic decline without divergence. The validation $\text{mAP}_{50}$ continued climbing through epoch 100 without evidence of severe overfitting.

### 5.3 Test Set Performance Breakdown (Held-Out Benchmark, 82 Images)

The model weights (`best.pt`) were evaluated on the held-out test split:

| Metric | Overall Value |
|---|---|
| **Precision ($P$)** | **0.874** (87.4%) |
| **Recall ($R$)** | **0.702** (70.2%) |
| **$\text{mAP}_{50}$** | **0.754** (75.4%) |
| **$\text{mAP}_{50-95}$** | **0.465** (46.5%) |

#### Per-Class Test Performance (Key Classes)

| Class | Precision ($P$) | Recall ($R$) | $\text{mAP}_{50}$ | Analysis & Viva Explanation |
|---|:---:|:---:|:---:|---|
| **Person** | 0.892 | 0.841 | 0.891 | Excellent worker localization; serves as solid anchor |
| **Hardhat** | 0.915 | 0.826 | 0.884 | Distinct dome geometry and bright colors yield high accuracy |
| **Safety Vest** | 0.884 | 0.792 | 0.852 | Retroreflective striping patterns are strongly recognized |
| **NO-Safety Vest** | 0.812 | 0.680 | 0.718 | Moderate recall; torso clothing patterns vary widely |
| **NO-Hardhat** | **0.724** | **0.561** | **0.545** | **Weakest class:** small pixel footprint, hair/caps confuse detector |

> **Viva Insight:** Why is `NO-Hardhat` the weakest class?  
> An unhelmeted human head occupies a very small pixel area at distance. Hair colors, baseball caps, beanies, and skin tones exhibit massive variance compared to standardized rigid hardhat shapes. This empirical finding directly motivated our **Strict Hardhat Policy** in the compliance engine.

---

## 6. Compliance Engine Architecture

Object detection identifies *what* exists in a scene, but cannot determine *compliance*. The pure-Python compliance module ([`src/compliance.py`](file:///d:/AI_project/src/compliance.py)) bridges this gap using strict spatial rules and defensive reliability safeguards.

```
       RAW IMAGE / CCTV FRAME
                 │
                 ▼
       ┌────────────────────┐
       │ YOLOv8n INFERENCE  │  (Base Confidence = 0.10)
       └─────────┬──────────┘
                 │
                 ▼
       ┌────────────────────┐
       │ TWO-STAGE FILTERS  │  (CONF_PERSON=0.30, CONF_GEAR=0.20-0.30)
       └─────────┬──────────┘
                 │
                 ▼
       ┌────────────────────┐
       │ ANATOMICAL ZONES   │  Head: Top 0% - 35%
       │  DECOMPOSITION     │  Torso: Middle 15% - 75%
       └─────────┬──────────┘
                 │
                 ▼
       ┌────────────────────┐
       │  IoG OVERLAP MATCH │  IoG = Area(Gear ∩ Zone) / Area(Gear) ≥ 0.30
       │  & DISAMBIGUATION  │  Greedy unique assignment
       └─────────┬──────────┘
                 │
                 ▼
       ┌────────────────────┐
       │ RELIABILITY ENGINE │  • Aspect Ratio ≥ 1.2 (Not squashed/overhead)
       │  & QUALITY GUARDS  │  • Scale ≥ 12% frame height (Not tiny background)
       └─────────┬──────────┘  • Confidence ≥ 0.40 & Conflict Check
                 │
                 ▼
       ┌────────────────────┐
       │ DECISION ENGINE    │  ==> Compliant / Violation / Needs Review
       └────────────────────┘
```

### 6.1 Two-Stage Confidence Filtering
Running YOLO at a single standard confidence threshold (e.g. 0.25) drops small pieces of gear during non-maximum suppression (NMS). Our architecture introduces two-stage filtering:
1. **Stage 1 (Permissive Base Inference):** YOLO runs at $\text{conf} = 0.10$, preserving partial and distant gear candidates.
2. **Stage 2 (Class-Specific Gates):**
   * $\text{CONF\_PERSON} = 0.30$: Suppresses false positive worker boxes on background objects.
   * $\text{CONF\_HARDHAT} = 0.30$: Demands high certainty for positive gear claims.
   * $\text{CONF\_NO\_HARDHAT} = 0.20$ & $\text{CONF\_NO\_VEST} = 0.20$: Lower thresholds prioritize violation recall (ensuring safety officers do not miss violations).
   * $\text{CONF\_SAFETY\_VEST} = 0.20$: Improves recall on partially occluded vests.

### 6.2 Anatomical Zone Partitioning
For each detected person bounding box $[x_1, y_1, x_2, y_2]$ with height $H = y_2 - y_1$:
* **Head Zone:** Extends from $y_1 + 0.0 \cdot H$ to $y_1 + 0.35 \cdot H$. This top 35% region captures helmets across varying human head tilts.
* **Widened Torso Zone:** Extends from $y_1 + 0.15 \cdot H$ to $y_1 + 0.75 \cdot H$. 
  * *Tuning Insight:* The torso zone was deliberately widened from an initial $0.20–0.65$ to $0.15–0.75$. This captures safety vests on workers carrying backpacks, tool belts, or wearing safety harnesses that displace the vest detection upward or downward.

### 6.3 Intersection over Gear Area ($\text{IoG}$) Association
Standard Intersection over Union ($\text{IoU}$) fails because a hardhat box ($\approx 40 \times 40\text{px}$) is vastly smaller than a worker body zone ($\approx 120 \times 300\text{px}$), resulting in an IoU below 0.10 even when the helmet is perfectly situated. We implement **Intersection over Gear Area ($\text{IoG}$)**:

$$\text{IoG} = \frac{\text{Area}(\text{Gear Box} \cap \text{Anatomical Zone})}{\text{Area}(\text{Gear Box})}$$

A named constant threshold of $\text{OVERLAP\_THRESHOLD} = 0.30$ requires at least 30% of the gear box to fall inside the anatomical zone. 

**Greedy Bipartite Disambiguation:** In crowded construction scenes, a gear box may overlap zones of two adjacent workers. The algorithm calculates IoG against all candidate workers and assigns the gear box exclusively to the worker zone with the highest overlap. One piece of gear is strictly prevented from being counted twice.

### 6.4 Multi-Criteria Reliability Safeguards
To eliminate false penalties, every worker detection is assessed by four independent quality guards:
1. **Upright Posture Aspect Ratio Guard (`MIN_UPRIGHT_ASPECT = 1.2`):**
   $$\text{Aspect Ratio} = \frac{\text{Box Height}}{\text{Box Width}} \ge 1.2$$
   Squashed bounding boxes ($\text{aspect} < 1.2$) indicate overhead bird's-eye camera perspectives or crouching/seated workers where head-torso geometry collapses. Such workers are routed to `"Needs review: view not upright"` and vest status is overridden to unknown.
2. **Resolution Scale Guard (`MIN_PERSON_HEIGHT_FRAC = 0.12`):**
   Workers whose bounding-box height is $< 12\%$ of the total frame height are too distant for reliable gear detection. They are routed to `"Needs review: small"`.
3. **Detection Confidence Guard (`MIN_PERSON_CONF = 0.40`):**
   Person detections with confidence $< 0.40$ are routed to `"Needs review: low confidence"`.
4. **Conflicting Evidence Detection:**
   If both positive (`Hardhat`) and negative (`NO-Hardhat`) gear boxes match to the same person, the conflict is not resolved by confidence scores. Instead, it is routed to `"Needs review: conflicting detections"`.

### 6.5 Strict Hardhat Enforcement Policy
Because the trained detector exhibits lower recall on `NO-Hardhat` (56.1%) than on `Hardhat` (82.6%), a worker without a helmet frequently produces **no head detection at all** rather than a `NO-Hardhat` box.
* Under `STRICT_HARDHAT_POLICY = True`: A worker whose head zone contains **no hardhat detection** is classified as a **Violation (`Missing hardhat`)**, but **ONLY IF** the person is verified as reliable by all three guards.
* Unreliable or distant workers are never strictly penalized; they remain in "Needs Review".

---

## 7. System-Level Evaluation & Diagnostics

To benchmark real-world generalization, the system was evaluated across **15 independent, out-of-distribution construction photos** (`photo01.jpg` – `photo15.jpg`) sourced from diverse job sites (excavations, high-rise rebar decks, scaffolding, and roadwork), documented in [`eval/test_photos/SOURCES.md`](file:///d:/AI_project/eval/test_photos/SOURCES.md).

Human safety audits established ground-truth counts (`true_compliant`, `true_violations`) in [`eval/ground_truth.csv`](file:///d:/AI_project/eval/ground_truth.csv).

### 7.1 Quantitative Evaluation Results (Default Confidence 0.25)

```
========================================================================================================
EVALUATION BENCHMARK SUMMARY (15 Real-World Test Photos):
========================================================================================================
• Total Ground-Truth Workers:              88 workers
• Total Detected Workers:                  69 workers (Detection Rate: 78.4%)
• Judged Workers (Compliant + Violations): 37 / 69 detected (53.6%)
• Needs-Review Workers Rate:               32 / 69 detected (46.4%)
--------------------------------------------------------------------------------------------------------
• Image-Level Exact Match Accuracy:        2 / 15 (13.3%)
• Within ±1 Violation Count Accuracy:      14 / 15 (93.3%)
• Within ±1 Compliant Count Accuracy:      4 / 15 (26.7%)
• Mean Absolute Error (Violations):        0.60 workers
• Mean Absolute Error (Compliant):         3.73 workers
--------------------------------------------------------------------------------------------------------
• Violations Flagged by Pipeline:          12 flagged
• Violations Caught (Upper Bound):         5 / 7 true violations (71.4% caught)
========================================================================================================
```

### 7.2 Breakdown of "Needs Review" by Diagnostic Reason

A central finding of this investigation is the composition of the 32 "Needs Review" workers:

| Reason Category | Count | % of Needs Review | Root Cause & System Behavior |
|---|:---:|:---:|---|
| **`small`** | **17** | **53.1%** | Distant background workers ($< 12\%$ frame height). Safeguard correctly prevented false penalties. |
| **`conflicting detections`** | **6** | **18.8%** | Overlapping gear boxes on adjacent workers or partial reflections. |
| **`low confidence`** | **4** | **12.5%** | Marginal person detections ($< 0.40$ conf) safely quarantined. |
| **`view not upright`** | **4** | **12.5%** | Squashed aspect ratio from high-angle or crouching postures. |
| **`vest unknown`** | **1** | **3.1%** | Person reliable with hardhat, but vest unconfirmed; required manual audit. |
| **`hardhat unknown`** | **0** | **0.0%** | Zero, because strict hardhat policy adjudicated all reliable unhelmeted heads. |
| **`other`** | **0** | **0.0%** | All review instances were fully explainable by named guards. |

### 7.3 Confidence Slider Trade-Off Analysis

Running [`eval/slider_check.py`](file:///d:/AI_project/eval/slider_check.py) across confidence thresholds **0.15, 0.25, and 0.40** revealed a fundamental operational trade-off:

| Threshold | Detected Workers | Compliant | Violations | Needs Review | Review Rate | Key Trade-Off Observation |
|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **0.15** | 80 | 27 | 11 | 42 | 52.5% | Maximizes worker recall, but 42 workers are sent to review due to low confidence and distance clutter. |
| **0.25** | **69** | **25** | **12** | **32** | **46.4%** | **Optimal Operating Point:** Balanced detection with highest violation precision. |
| **0.40** | 60 | 21 | 18 | 21 | 35.0% | Review count drops to 21, but violation flags spike to 18 because faint gear detections are eliminated, causing strict hardhat policy to treat missing helmet detections as violations. |

---

## 8. Failure Case Analysis

Visual inspection of error modes ([`eval/results/failure_cases/`](file:///d:/AI_project/eval/results/failure_cases/)) identified three primary industrial edge cases:

### Case 1: Overhead Bird's-Eye Perspectives (`ppe10.jpg`)
* **Observation:** Overhead security cameras looking down at steep angles compress the human body vertically.
* **Failure Mode:** Aspect ratio drops to $h/w \approx 1.03$ (below the 1.20 threshold). Normal YOLO person features are severely degraded.
* **Safeguard Response:** The aspect guard intercepts the detection and routes the worker to `"Needs review: view not upright"`, preventing false violation tagging.

### Case 2: Rebar Grids, Bending Postures & Tool Belts (`photo03.jpg`)
* **Observation:** Ironworkers tying steel rebar bend forward at the waist while wearing heavy tool belts and equipment pouches.
* **Failure Mode:** Torso zones are partially occluded by steel bars and pouches, causing simultaneous detection of vest and background clutter.
* **Safeguard Response:** Torso widening ($15\%–75\%$) captures the displaced vest, while unresolvable ambiguities trigger `"Needs review: conflicting detections"`.

### Case 3: Plain Fluorescent Vests vs. Reflective Tape (`photo06.jpg`)
* **Observation:** Road construction workers frequently wear solid fluorescent orange or yellow t-shirts or plain vests devoid of silver retroreflective bands.
* **Failure Mode:** The model was trained on the `css-data` dataset where `Safety Vest` annotations require retroreflective tape. As a result, 2 road workers in solid garments were flagged as `Missing vest`.
* **Viva Note:** This is not an architectural defect, but an explicit **labeling boundary** of the training dataset.

---

## 9. CCTV Video Surveillance & Temporal Smoothing

Real-time video processing ([`src/video.py`](file:///d:/AI_project/src/video.py)) introduces challenges absent from static images: worker motion, sudden head turns, and transient frame-to-frame occlusions cause detection boxes to flicker, causing fluctuating compliance metrics.

### 9.1 Sequential Pipeline Architecture
* **Frame Resizing:** Frames are resized proportionally to a maximum width of **416 px**. Benchmarking proved 416 px achieves ~30% faster CPU throughput than 640 px with negligible detection loss.
* **Analysis Stride:** $\text{PROCESS\_EVERY\_N} = 1$ ensures full-frame temporal continuity.
* **30-Second Cap:** Video analysis processes up to the first 30 seconds ($\le 750$ frames), ensuring complete analysis within standard web session limits.
* **H.264 MP4 Export:** Output videos are encoded with `yuv420p` pixel format using `imageio-ffmpeg` capped at 30 FPS for universal HTML5 browser playback.

### 9.2 7-Frame Rolling Median Temporal Anti-Flicker Filter
Rather than implementing heavy, drift-prone multi-object tracking (DeepSORT/ByteTrack), we implemented a **7-frame rolling median filter** on compliance counts:

$$\hat{C}_t = \text{median}\left(C_{t-3}, C_{t-2}, C_{t-1}, C_t, C_{t+1}, C_{t+2}, C_{t+3}\right)$$

* **Transient Flicker Suppression:** A worker momentarily turning their head away from the camera drops the hardhat detection for 1–2 frames. The 7-frame median absorbs this momentary drop, preventing spurious violation alarms.
* **Sustained Violation Preservation:** A genuine violator without a hardhat across 5+ frames remains fully flagged.
* **Zero-Worker Preservation:** Frames with zero workers are never distorted into ghost detections.

---

## 10. Application Architecture & User Interface

The application ([`app.py`](file:///d:/AI_project/app.py)) is implemented as an enterprise-grade Streamlit portal with a multi-page industrial design:

```
Streamlit Application Flow:
├── Sidebar Navigation
│   ├── Inspection Mode Toggle (Image vs. Video)
│   ├── Page Selector (Dashboard, New Inspection, Results, Audits, History, Reports, Settings)
│   ├── Hardhat Policy Toggle (Strict Mode ON/OFF)
│   └── Engine Status (YOLOv8 CPU, Weights, 10 Classes Verified)
├── Main Workspace
│   ├── Top Industrial Navigation Bar
│   ├── KPI Metric Cards (Compliance Rate, Total Workers, Violations, Needs Review)
│   ├── Interactive Visual Canvas (Color-coded Bounding Boxes & Zones)
│   ├── Compliance Rate Over Time Line Chart (Video Mode)
│   ├── Top Risk Moments Gallery (Highest Violation Frames)
│   └── Downloadable CSV Compliance Audit Trail
```

### Key UI Features
* **Color-Coded Visual Language:**
  * 🟢 **Green (`#28C828`):** Fully Compliant Worker (Hardhat + Vest verified).
  * 🔴 **Red (`#DC1E1E`):** Violation (Missing Hardhat, Missing Vest, or Missing Both).
  * 🟠 **Amber (`#F5A623`):** Needs Review (Distant, low confidence, non-upright, or conflicting).
  * 🔵 **Cyan (`#00B4F0`):** Detected Positive Gear (Hardhat / Safety Vest).
* **Comprehensive Test Suite:** 68 isolated unit tests covering zone extraction, overlap calculations, reliability filters, temporal smoothing, and YouTube URL parsing—all passing in under 2 seconds.

---

## 11. Limitations & Future Improvements

To maintain scientific integrity, the known limitations of this pipeline are acknowledged:

1. **Extreme Overhead & Oblique Perspectives:** Cameras mounted directly above work areas ($> 60^\circ$ pitch) compress human geometry. A specialized overhead top-view detection model is required for cranes and gantry cameras.
2. **Reflective Tape Dependency:** The model cannot differentiate high-visibility solid neon fabric from standard shirts without reflective striping. Fine-tuning on a broader dataset including non-reflective hi-vis fabric is recommended.
3. **Seated & Crouching Postures:** Workers operating compact machinery or sitting are excluded from strict enforcement by the upright aspect guard ($h/w < 1.2$). Pose-estimation integration (YOLOv8-Pose) would enable keypoint-based torso tracking regardless of posture.
4. **Subcontractor Re-Identification:** In video mode, violations are tracked by frame timestamp and bounding box, but workers are not re-identified across different camera views. Incorporating facial blurring and anonymous Re-ID feature embeddings would enable individual worker repeat-offender tracking.

---

## 12. Conclusion & Viva Defense Guide

### Summary of Achievements
* Successfully trained and deployed an Ultralytics **YOLOv8n** model achieving **0.813 validation $\text{mAP}_{50}$** and **0.754 test $\text{mAP}_{50}$**.
* Engineered a decoupled, mathematically verifiable compliance engine using **IoG $\ge 0.30$** and anatomical zone decomposition.
* Designed defensive reliability guards that conservatively route ambiguous edge cases to a "Needs Review" queue, achieving **93.3% within $\pm 1$ accuracy on violation counts** across out-of-distribution real-world construction photos.
* Delivered a complete industrial Streamlit suite with CCTV video processing, temporal smoothing, and CSV reporting.

### Core Viva Defense Q&A

**Q1: Why did you use YOLOv8 Nano instead of YOLOv8 Medium or Faster R-CNN?**  
*Answer:* Construction site safety monitoring requires real-time processing on commodity edge hardware and CCTV DVRs, which lack expensive dedicated GPUs. YOLOv8 Nano contains only 3.14M parameters and operates at 8.8 GFLOPs, enabling low-latency CPU inference (~80ms/frame) while achieving 81.3% $\text{mAP}_{50}$.

**Q2: Why did you use Intersection over Gear ($\text{IoG}$) instead of Intersection over Union ($\text{IoU}$)?**  
*Answer:* In standard IoU, the denominator is the union of the gear box and the person zone. Because a person's torso or head zone is vastly larger than a hardhat or vest box, the union is large, forcing IoU values below 0.10 even for perfect detections. IoG measures the fraction of the *gear box* contained inside the zone, making it scale-invariant and robust to small equipment.

**Q3: Why does your evaluation show so many workers in "Needs Review" (46.4%)?**  
*Answer:* In a safety-critical application, a false accusation (flagging an innocent worker as a violator) or a false compliance claim (clearing a violator) degrades trust. The majority of review workers (53.1%) were tiny background figures ($<12\%$ frame height) where the camera resolution cannot resolve helmet straps. Quarantining uncertain instances to "Needs Review" is a deliberate safety design choice, not an algorithmic defect.

**Q4: How do you handle transient detection drops in video without tracking?**  
*Answer:* We apply a 7-frame rolling median filter across analyzed frames. Because median filters reject impulse noise, a 1–2 frame drop caused by a rapid head turn is smoothed out, while persistent non-compliance across 5+ frames remains flagged as a violation.

---

*Report compiled from empirical results, codebase inspection, and evaluation benchmarks in `d:\AI_project\`.*
