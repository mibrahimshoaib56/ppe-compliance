---
marp: true
theme: gaia
_class: lead
paginate: true
backgroundColor: #f8fafc
color: #0f172a
---

# 🛡️ Hardhat & Safety-Vest Compliance Detection
### Construction Site Surveillance Using YOLOv8 & Spatial Rule Reasoning
**Course:** AI2002 — Artificial Intelligence (Path A: Software-Based)  
**Main Developer:** Muhammad Ibrahim Shoaib  
**Repository:** `github.com/mibrahimshoaib56/ppe-compliance`  
**Date:** November 2026  

---

## Slide 1: Problem Statement & Motivation

### The Construction Safety Challenge
* **Site Hazards:** Moving machinery, overhead loads, scaffolding, and falling materials pose safety risks.
* **Required Safety Gear:**
  1. **Safety Hardhats:** Protect against falling tools, overhead loads, and structural impacts.
  2. **High-Visibility Vests:** Ensure worker visibility around mobile construction equipment.

### Challenges of Manual Monitoring
* **Intermittent Coverage:** Safety personnel inspect periodically; gear may be removed when inspectors are absent.
* **Large Spatial Scale:** Multi-level structures, scaffolding, and excavations cannot be surveyed simultaneously.
* **Observer Fatigue:** Manual tracking across full work shifts can lead to unobserved infractions.

> **Speaker Note:** Emphasize that automated monitoring provides continuous coverage and consistent documentation across shifts.

---

## Slide 2: Proposed Solution: The Two-Tier Architecture

### Structured Pipeline Architecture

```
┌────────────────────────────────────────────────────────┐
│  Tier 1: Deep Learning Perception (Ultralytics YOLOv8)  │
│  Detects bounding boxes: Person, Hardhat, Vest, etc.   │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│  Tier 2: Spatial Compliance & Reliability Engine       │
│  Anatomical Zones + IoG Overlap + Multi-Criteria Guards │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│  Streamlit Portal & Video Post-Processing Filter       │
│  Summary Dashboard, 7-Frame Rolling Median, CSV Log    │
└────────────────────────────────────────────────────────┘
```

* **Separation of Concerns:** Deep learning detects visual objects; geometric spatial rules determine compliance.
* **Conservative Triage:** Ambiguous, distant, or non-upright figures are routed to `"Needs Review"` rather than marked as violations.

> **Speaker Note:** Highlight that the neural network performs object localization, while compliance decisions are handled by deterministic geometric rules.

---

## Slide 3: Algorithm Justification: Why Detection & YOLOv8?

### Detection vs. Image Classification
* **Classification Limitations:** Whole-image classification cannot handle multiple workers or identify *which* specific gear piece is missing.
* **Object Detection Capability:** Predicts bounding boxes $[x_1, y_1, x_2, y_2]$, class labels, and confidence scores for every worker and piece of gear.

### Why YOLOv8 Nano (`yolov8n`)?
* **Single-Stage Throughput:** Anchor-free decoupled head processes full frames in one pass.
* **Multi-Scale Feature Fusion:** **C2f modules** enhance feature gradient flow for varied object scales.
* **Model Configuration:**
  * **3,012,798 Parameters** (129 layers) as trained; 3,134,894 parameters (73 layers) when fused.
  * **8.2 GFLOPs** compute complexity as trained (8.6 GFLOPs fused).
  * Runs on CPU hardware without requiring dedicated local GPU hardware.

> **Speaker Note:** Note that YOLOv8 Nano provides a compact footprint (3.01M parameters) suitable for CPU execution without heavy hardware requirements.

---

## Slide 4: Dataset Description (Construction Site Safety v28)

* **Source:** Roboflow Universe (`roboflow-universe-projects/construction-site-safety`), CC BY 4.0 license.
* **Total Volume:** **2,801 annotated images** at $640 \times 640$ resolution.

| Split | Images | % | Characteristics |
|---|:---:|:---:|---|
| **Training** | **2,605** | 93.0% | Baked-in $2 \times 2$ mosaics with cutout patch squares |
| **Validation** | **114** | 4.1% | 114 images (counts verified; not individually inspected) |
| **Testing** | **82** | 2.9% | Held-out benchmark test set of natural photographic scenes |

### 10 Trained Classes
* **Worker & Target Gear:** `Person` (5), `Hardhat` (0), `NO-Hardhat` (2), `Safety Vest` (7), `NO-Safety Vest` (4)
* **Contextual Classes:** `machinery` (8), `vehicle` (9), `Safety Cone` (6), `Mask` (1), `NO-Mask` (3)

> **Speaker Note:** Note that training images contain mosaic augmentations to teach partial equipment cues, while the held-out test split consists of natural photos.

---

## Slide 5: Model Training & Empirical Metrics

### Training Setup on Kaggle GPU (NVIDIA Tesla T4)
* **Framework:** Ultralytics 8.4.215 | **Epochs:** 100 | **Batch Size:** 16 | **Image Size:** 640
* **Optimizer:** AdamW (auto-selected, lr $0.000714$, momentum $0.9$) | **Time:** **0.565 hours**

### Benchmark Results (Held-Out Test Set, 82 Images, 760 Boxes)
* **Validation at Epoch 100:** $\text{mAP}_{50} = \mathbf{0.814}$, $\text{mAP}_{50-95} = \mathbf{0.512}$ (`best.pt`: 0.813 / 0.511)
* **Test Split Overall:** $\text{mAP}_{50} = \mathbf{0.754}$ | $\text{mAP}_{50-95} = \mathbf{0.465}$ | Precision = **0.874** | Recall = **0.702**
* **Test Speed on T4 GPU:** 1.8 ms preprocess, 6.8 ms inference, 2.3 ms postprocess per image (No CPU timing measured)

### Per-Class Test Performance (`test_metrics.txt`)
* `Hardhat`: $P = 0.959, R = 0.809, \text{AP}_{50} = 0.905$ | `Safety Vest`: $P = 0.896, R = 0.849, \text{AP}_{50} = 0.902$
* `Person`: $P = 0.884, R = 0.786, \text{AP}_{50} = 0.857$ | `NO-Safety Vest`: $P = 0.930, R = 0.739, \text{AP}_{50} = 0.799$
* `NO-Hardhat`: $P = 0.807, R = 0.561, \text{AP}_{50} = 0.545$ (Lowest AP50, likely due to small head scale and visual variation)

> **Speaker Note:** Point out that NO-Hardhat recall was lower than positive hardhat recall, which informed the strict hardhat policy logic.

---

## Slide 6: End-to-End Pipeline Architecture

![Pipeline Architecture](file:///d:/AI_project/eval/results/pipeline_architecture_diagram.png)

1. **Input Ingestion:** Image, video file, or YouTube stream resized $\le 416\text{px}$.
2. **YOLOv8 Inference:** Base confidence 0.10 to prevent early NMS drops.
3. **2-Stage Class Filtering:** Per-class gates (Person $\ge 0.30$, Gear $\ge 0.20–0.30$).
4. **Anatomical Partitioning:** Head zone (0%–35%) & widened torso zone (15%–75%).
5. **Spatial IoG Association:** $\text{IoG} \ge 0.30$ with greedy disambiguation.
6. **Multi-Criteria Reliability:** Aspect ratio $\ge 1.2$, height fraction $\ge 12\%$, conf $\ge 0.40$.
7. **Compliance Decision Engine:** Compliant / Violation / Needs Review.
8. **Post-Processing Smoothing & UI:** 7-frame rolling median filter for aggregate counts/rates & interactive dashboard.

> **Speaker Note:** Walk through the 8 stages from input detection to geometric zone evaluation and final status resolution.

---

## Slide 7: Spatial Compliance & Anatomical Zones

### Intersection over Gear Area ($\text{IoG}$)
* Standard $\text{IoU}$ is small when comparing a small gear box to a large person zone.
* **Our Formulation:** **Intersection over Gear Area ($\text{IoG}$)**:

$$\text{IoG} = \frac{\text{Area}(\text{Gear Box} \cap \text{Anatomical Zone})}{\text{Area}(\text{Gear Box})} \ge 0.30$$

### Anatomical Body Zones
* **Head Zone (top 0% – 35%):** Covers helmets and bare heads across head tilts.
* **Widened Torso Zone (middle 15% – 75%):** Widened from original $20\%–65\%$ to accommodate tool belts, utility pouches, and vest displacement.
* **Greedy Disambiguation:** Each gear box is assigned to the worker zone with the highest IoG, preventing double-counting.

> **Speaker Note:** Explain that IoG divides by the gear box area, making it appropriate for associating small gear with larger body zones.

---

## Slide 8: Multi-Criteria Reliability Safeguards

### Conservative Triage of Uncertain Detections

| Safeguard | Threshold | Reason Assigned | Function |
|---|:---:|---|---|
| **Upright Aspect Ratio** | $h / w \ge 1.2$ | `"Needs review: view not upright"` | Rejects overhead / squashed views where body geometry collapses. |
| **Resolution Scale** | $h \ge 12\%$ frame | `"Needs review: small"` | Excludes tiny distant figures where camera cannot resolve details. |
| **Detection Confidence** | $\text{conf} \ge 0.40$ | `"Needs review: low confidence"` | Suppresses background clutter and weak detections. |
| **Conflict Detector** | Pos & Neg detected | `"Needs review: conflicting detections"` | Prevents arbitrary decisions when both positive and negative gear match. |

### Strict Hardhat Policy Toggle
* When ON, a reliable worker with no hardhat detection is adjudicated as a **Violation (`Missing hardhat`)**.
* Unreliable or distant figures are not strictly penalized; they remain in "Needs Review".

> **Speaker Note:** Explain that routing ambiguous figures to "Needs Review" prevents making false violation claims on low-confidence or distant background figures.

---

## Slide 9: System Evaluation on 15 Test Photos

### Evaluation on 15 Test Photos from Google Images (88 True Workers, Visual Ground Truth)

| Evaluation Metric | Value |
|---|:---:|
| **Exact Match Accuracy (Both Counts)** | **13.3%** (2 of 15 images) |
| **Within $\pm 1$ Accuracy (Violations)** | **93.3%** (14 of 15 images) |
| **Within $\pm 1$ Accuracy (Compliant)** | **26.7%** (4 of 15 images) |
| **Mean Absolute Error (Violations)** | **0.60 workers** |
| **Mean Absolute Error (Compliant)** | **3.73 workers** |
| **"Predict Zero Violations" Baseline** | **80.0%** (12 of 15 within $\pm 1$) |
| **Worker Coverage** | 69 detected vs 88 true (counts only; not matched 1:1) |
| **Verdict Breakdown** | 37 received a verdict (25 compliant, 12 violations); 32 sent to review |

### Breakdown of the 32 "Needs Review" Workers
* **`small` ($< 12\%$ height):** **53.1%** (17 workers) — Distant background figures.
* **`conflicting detections`:** **18.8%** (6 workers) — Overlapping boxes.
* **`low confidence` ($< 0.40$):** **12.5%** (4 workers) — Weak person detections.
* **`view not upright` ($h/w < 1.2$):** **12.5%** (4 workers) — Overhead/crouching views.
* **`vest unknown`:** **3.1%** (1 worker) — Unconfirmed vest evidence.

> **Speaker Note:** Note that the zero-violation baseline achieves 80% because ground truth contains only 7 violations across 88 workers.

---

## Slide 10: Confidence Threshold Trade-Off & Failure Cases

### Confidence Threshold Sweep (0.15, 0.25, 0.40) on 15 Test Photos
* **0.15:** Workers = 83, Compliant = 27, Violations = 9, Needs Review = 47 (Review Rate = 56.6%)
* **0.25:** Workers = 69, Compliant = 25, Violations = 12, Needs Review = 32 (Review Rate = 46.4%)
* **0.40:** Workers = 58, Compliant = 20, Violations = 18, Needs Review = 20 (Review Rate = 34.5%)
* *Observation:* Confidence 0.25 represents the least-bad trade-off on these 15 photos.

### Documented Failure Cases
1. **Overhead Angles (`sample_images/ppe10.jpg`):** Base detections yield Person confidences 0.268 and 0.148, both dropped by the `CONF_PERSON = 0.30` gate $\to$ 0 detected workers.
2. **Material Occlusion (`eval/test_photos/photo03.jpg`):** Workers on rebar deck with bags/scaffolding causing partial torso occlusion. True counts: 10 compliant, 0 violations; predicted: 3 compliant, 0 violations, 3 Needs Review (1 small, 1 upright, 1 conflict).
3. **Clothing without Reflectors (`eval/test_photos/photo06.jpg`):** Roadway workers wearing solid fluorescent tops without silver retroreflective tape. True counts: 3 compliant, 2 violations; predicted: 1 compliant, 3 violations, 1 Needs Review (conflict).

> **Speaker Note:** Explain the slider trade-off: at 0.15 review count is high, while at 0.40 missing gear detections raise violation counts under strict enforcement.

---

## Slide 11: Video Processing & Temporal Smoothing

### Smoothing Aggregate Counts & Compliance Rates
* Transient detection variations across frames can cause noise in aggregate summary metrics.

### 7-Frame Rolling Median Filter Implementation

$$\hat{C}_i = \text{round}\left(\text{median}\left(C_{i-3}, \dots, C_{i+3}\right)\right)$$

* **Smoothed Metrics:** Operates purely on frame-level aggregate counts (`total_workers`, `compliant_count`, `violation_count`, `needs_review_count`) and `compliance_rate`.
* **Bounding Boxes Unaltered:** Bounding boxes, per-worker tracks, and playback video frames remain raw per-frame detections; workers are not tracked across frames.
* **Window Size:** 7 consecutive analyzed frames (`TEMPORAL_SMOOTHING_WINDOW = 7`, from $i - 3$ to $i + 3$).
* **Execution Phase:** Runs in a post-processing pass after all video frames have been decoded and analyzed.

### Video Interface Features
* **Playback:** Displays annotated video rendered with raw per-frame bounding boxes.
* **Dashboard:** Displays smoothed summary metric cards, compliance rate curve over time, and top violation moments.

> **Speaker Note:** Emphasize that smoothing runs in post-processing over aggregate counts and rates; individual bounding boxes and playback frames are not smoothed.

---

## Slide 12: Summary of Deliverables & Limitations

### Project Deliverables
1. **Model:** YOLOv8n (3,012,798 parameters trained, 8.2 GFLOPs), validation $\text{mAP}_{50} = \mathbf{0.813}$, test $\text{mAP}_{50} = \mathbf{0.754}$.
2. **Logic Engine:** Python compliance engine with IoG overlap, anatomical zones, and 4 reliability safeguards.
3. **Evaluation Benchmark:** 93.3% within $\pm 1$ violation accuracy on 15 test photos (MAE 0.60).
4. **Application:** Streamlit web application with image upload, video processing with annotated playback, and CSV export.
5. **Quality Assurance:** 68 unit tests passing (`python -m pytest -q`).

### Limitations
* The evaluation set contains only 15 photos with 88 workers and 7 true violations, limiting statistical conclusions.
* Downward camera perspectives compress human body geometry, triggering the aspect ratio guard.
* The model relies on reflective tape striping to recognize safety vests.

---

# Thank You!
### Questions & Discussion
`https://github.com/mibrahimshoaib56/ppe-compliance`
