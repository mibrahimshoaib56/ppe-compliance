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

## Slide 1: Problem Statement & Industrial Motivation

### The Construction Safety Challenge
* **The Fatal Four Hazards:** Falls from height and struck-by equipment account for over **60% of construction fatalities** worldwide (OSHA / HSE).
* **Mandatory Life-Safety Gear:**
  1. **Industrial Hardhats:** Protect against falling tools, overhead loads, and structural impacts.
  2. **High-Visibility Vests:** Ensure pedestrian-worker visibility around heavy machinery (loaders, excavators).

### Why Manual Safety Audits Fail
* **Intermittent Walkthroughs:** Safety officers inspect periodically; workers often discard gear when unsupervised.
* **Large Spatial Scale:** High-rise rebar decks, multi-tier scaffolds, and excavations cannot be surveyed simultaneously.
* **Observer Fatigue:** Human oversight leads to missed infractions and contentious, unrecorded audit disputes.

> **Speaker Note:** Emphasize that PPE non-compliance is rarely deliberate malice—it happens during hot hours or rushed shifts. Automated visual surveillance provides non-stop, objective audit logs without requiring a human officer at every corner.

---

## Slide 2: Proposed Solution: The Two-Tier Architecture

### Overcoming "Black-Box" Neural Networks in Industrial Safety

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
│  Streamlit Portal & CCTV Temporal Video Anti-Flicker    │
│  Live Dashboard, 7-Frame Rolling Median, CSV Audit Log │
└────────────────────────────────────────────────────────┘
```

* **Separation of Concerns:** Deep learning locates visual entities; deterministic mathematics establishes legal compliance attribution.
* **Defensive Auditing:** Ambiguous or distant figures are quarantined to `"Needs Review"` rather than falsely penalized.

> **Speaker Note:** Highlight that we do NOT let the neural network directly output a compliance verdict. Deep learning is probabilistic; our rule engine provides deterministic, auditable spatial reasoning on top of it.

---

## Slide 3: Algorithm Justification: Why Detection & YOLOv8?

### Detection vs. Image Classification
* **Classification Fails:** Cannot handle multi-worker crowds or explain *which* specific gear piece is missing.
* **Object Detection Succeeds:** Predicts spatial coordinates $[x_1, y_1, x_2, y_2]$, class identity, and confidence for every worker and item simultaneously.

### Why YOLOv8 Nano (`yolov8n`)?
* **Single-Stage Throughput:** Anchor-free decoupled head processes full frames in one pass (>30 FPS on GPU, ~15–20 FPS on CPU).
* **Multi-Scale Feature Fusion:** **C2f modules** enhance gradient flow and small-object detection (helmets at distance).
* **Hardware Efficiency (CPU Deployment):**
  * **3,140,190 Parameters** (~3.14M weights)
  * **8.8 GFLOPs** compute complexity
  * Runs inference in **<80 ms per frame on standard laptop CPU** without requiring dedicated GPU hardware.

> **Speaker Note:** Viva defense point: Examiners love to ask why we chose Nano over Medium/Large. The answer is edge feasibility: CCTV monitoring runs on local DVRs/CPUs, where 3.14M parameters delivers real-time speed with 81.3% mAP.

---

## Slide 4: Dataset Description (Construction Site Safety v28)

* **Source:** Roboflow Universe (`roboflow-universe-projects/construction-site-safety`), CC BY 4.0 license.
* **Total Volume:** **2,801 annotated images** at $640 \times 640$ resolution.

| Split | Images | % | Characteristics |
|---|:---:|:---:|---|
| **Training** | **2,605** | 93.0% | Heavy augmentation: $2 \times 2$ baked-in mosaics & cutout patch squares |
| **Validation** | **114** | 4.1% | Natural photographic construction scenes |
| **Testing** | **82** | 2.9% | Held-out benchmark test set, unaugmented |

### 10 Trained Classes
* **Worker & Target Gear:** `Person` (5), `Hardhat` (0), `NO-Hardhat` (2), `Safety Vest` (7), `NO-Safety Vest` (4)
* **Contextual Classes:** `machinery` (8), `vehicle` (9), `Safety Cone` (6), `Mask` (1), `NO-Mask` (3)

> **Speaker Note:** Note that the training set uses synthetic $2 \times 2$ mosaics with cutouts, forcing the model to learn partial equipment cues, while our test sets are 100% natural photos.

---

## Slide 5: Model Training & Empirical Metrics

### Training Setup on Kaggle GPU (NVIDIA Tesla T4)
* **Epochs:** 100 | **Batch Size:** 16 | **Optimizer:** SGD ($\eta_0 = 0.01$, momentum 0.937)
* **Training Duration:** **0.565 hours** (~34 minutes)

### Benchmark Results (Held-Out Test Set, 82 Images)
* **Validation $\text{mAP}_{50}$:** **0.813** (81.3%)
* **Test $\text{mAP}_{50}$:** **0.754** | **Test $\text{mAP}_{50-95}$:** **0.465**
* **Test Precision:** **0.874** | **Test Recall:** **0.702**

### Per-Class Test Performance
* `Person`: $\text{mAP}_{50} = 0.891$ | `Hardhat`: $\text{mAP}_{50} = 0.884$ | `Safety Vest`: $\text{mAP}_{50} = 0.852$
* **Weakest Class:** `NO-Hardhat` ($\text{AP}_{50} = 0.545$, Recall = $0.561$)  
  * *Reason:* Small pixel scale, varied hair colors, caps, and beanies. Directly motivated our **Strict Hardhat Policy**.

> **Speaker Note:** Point out that mAP continued rising through epoch 100 without overfitting. Explain that the lower recall on NO-Hardhat is why our rule engine checks for missing hardhat detections on reliable workers.

---

## Slide 6: End-to-End Pipeline Architecture

![Pipeline Architecture](file:///d:/AI_project/eval/results/pipeline_architecture_diagram.png)

1. **Input Ingestion:** Image, CCTV video, or YouTube stream resized $\le 416\text{px}$.
2. **YOLOv8 Inference:** Base confidence 0.10 to prevent early NMS drops.
3. **2-Stage Class Filtering:** Per-class gates (Person $\ge 0.30$, Gear $\ge 0.20–0.30$).
4. **Anatomical Partitioning:** Head zone (0%–35%) & widened torso zone (15%–75%).
5. **Spatial IoG Association:** $\text{IoG} \ge 0.30$ with greedy bipartite disambiguation.
6. **Multi-Criteria Reliability:** Aspect ratio $\ge 1.2$, height fraction $\ge 12\%$, conf $\ge 0.40$.
7. **Compliance Decision Engine:** Compliant / Violation / Needs Review.
8. **Temporal Aggregation:** 7-frame rolling median filter & interactive dashboard.

> **Speaker Note:** Walk the audience through the 8 stages from left to right, highlighting how raw pixels become verified compliance records.

---

## Slide 7: Spatial Compliance & Anatomical Zones

### Why Traditional IoU Fails for PPE
* Standard $\text{IoU} = \frac{\text{Intersection}}{\text{Union}}$. A hardhat box ($\approx 40 \times 40\text{px}$) inside a person zone ($\approx 120 \times 300\text{px}$) produces an IoU $< 0.10$ even with 100% containment!
* **Our Solution:** **Intersection over Gear Area ($\text{IoG}$)**:

$$\text{IoG} = \frac{\text{Area}(\text{Gear Box} \cap \text{Anatomical Zone})}{\text{Area}(\text{Gear Box})} \ge 0.30$$

### Anatomical Body Zones
* **Head Zone (top 0% – 35%):** Captures helmets and bare heads across head tilts.
* **Widened Torso Zone (middle 15% – 75%):** Widened from original $20\%–65\%$ to accommodate workers wearing tool bags, heavy utility belts, or safety harnesses.
* **Greedy Disambiguation:** In crowds, gear is assigned to the worker zone with the highest IoG. One vest is never assigned to two workers!

> **Speaker Note:** This slide demonstrates technical depth. Memorize the IoG formula and explain why dividing by gear area is the correct mathematical formulation.

---

## Slide 8: Multi-Criteria Reliability Safeguards

### Defensive Auditing: Never Falsely Penalize an Innocent Worker

| Safeguard Guard | Threshold | Failure Reason Triggered | Why It Matters |
|---|:---:|---|---|
| **Upright Aspect Ratio** | $h / w \ge 1.2$ | `"Needs review: view not upright"` | Rejects overhead / squashed camera views where vest geometry collapses. |
| **Resolution Scale** | $h \ge 12\%$ frame | `"Needs review: small"` | Excludes tiny distant figures where camera cannot resolve helmet straps. |
| **Detection Confidence** | $\text{conf} \ge 0.40$ | `"Needs review: low confidence"` | Suppresses background clutter and ghost worker boxes. |
| **Conflict Detector** | Pos & Neg detected | `"Needs review: conflicting detections"` | Prevents coin-flip decisions when both Vest & NO-Vest match one worker. |

### Strict Hardhat Policy Toggle
* When ON, a reliable worker with no hardhat detection is adjudicated as a **Violation (`Missing hardhat`)**.
* Unreliable or distant figures are **never** strictly penalized; they stay in "Needs Review".

> **Speaker Note:** Emphasize that our system has a 3rd state ("Needs Review"). In real industrial safety, false penalties cause labor strikes, while missed violations cause deaths. A triage queue is the industry standard.

---

## Slide 9: System Evaluation on Real-World Photos

### Independent Benchmark on 15 Diverse Job-Site Photos (88 True Workers)

| Evaluation Metric | Value | Industrial Significance |
|---|:---:|---|
| **Within $\pm 1$ Violation Accuracy** | **93.3%** (14 / 15) | Outstanding operational reliability for safety officers |
| **Mean Absolute Error (Violations)** | **0.60 workers** | On average, violation count is off by less than 1 worker |
| **Exact Match Accuracy (Both Counts)** | 13.3% (2 / 15) | Conservative due to distant/background workers |
| **Violations Flagged / Caught** | 12 flagged / 5 caught | Catches 71.4% of true violations |
| **Needs-Review Worker Rate** | 46.4% (32 / 69) | Quarantines ambiguity to human audit |

### Breakdown of the 32 "Needs Review" Workers
* **`small` ($< 12\%$ height):** **53.1%** (17 workers) — Background figures safely excluded.
* **`conflicting detections`:** **18.8%** (6 workers) — Complex overlaps.
* **`low confidence` ($< 0.40$):** **12.5%** (4 workers) — Uncertain figures.
* **`view not upright` ($h/w < 1.2$):** **12.5%** (4 workers) — Overhead/crouching postures.
* **`vest unknown`:** **3.1%** (1 worker) — Unconfirmed vest evidence.

> **Speaker Note:** 93.3% within +-1 violation count is our strongest headline metric. Be ready to explain that 53% of review workers were distant background figures.

---

## Slide 10: Real-World Failure Cases & Edge Case Studies

Visual analysis from [`eval/results/failure_cases/`](file:///d:/AI_project/eval/results/failure_cases/):

### 1. Overhead Camera Angle (`01_overhead_view_ppe10.jpg`)
* *Symptom:* Steep downward angle squashes person bounding box ($h/w \approx 1.03$).
* *Result:* Aspect guard triggers $\to$ `"Needs review: view not upright"`.

### 2. Rebar Grid & Tool Bag Occlusion (`02_bag_rebar_occlusion_photo03.jpg`)
* *Symptom:* Workers bending over steel deck wearing heavy tool belts and carrying rebar.
* *Result:* Torso zone captures displaced vest; ambiguous cases route to `"conflicting detections"`.

### 3. Plain Vest vs. Reflective Tape (`03_plain_vest_photo06.jpg`)
* *Symptom:* Road workers wearing plain neon shirts without silver retroreflective tape.
* *Result:* Model trained on reflective tape flags them as `Missing vest`.
* *Insight:* Highlights the dataset labeling boundary—strict compliance enforces reflective standards.

> **Speaker Note:** Showing failure cases demonstrates honesty and deep engineering understanding. Academic examiners award high marks for analyzing failure modes.

---

## Slide 11: CCTV Video Surveillance & Temporal Smoothing

### Overcoming Detection Flicker in Video
* Worker motion and head turns cause transient 1–2 frame detection drops. Without tracking, raw bounding boxes flicker, causing fluctuating compliance metrics.

### Our Solution: 7-Frame Rolling Median Filter

$$\hat{C}_t = \text{median}\left(C_{t-3}, C_{t-2}, C_{t-1}, C_t, C_{t+1}, C_{t+2}, C_{t+3}\right)$$

* **Transient Suppression:** A 1-frame head turn does not unfairly degrade the site compliance score.
* **Sustained Violation Preservation:** A worker without a hardhat across 5+ frames remains fully flagged.
* **Zero Ghost Detections:** Frames with zero workers stay strictly at zero.

### Video Engine Performance (CPU Mode)
* **Resolution Optimization:** Resized to **416 px width** (~30% faster on CPU than 640 px).
* **Throughput:** ~12–15 FPS on CPU; encodes universal H.264 MP4 with executive KPI summary.

> **Speaker Note:** Emphasize that the rolling median filter gives the benefits of tracking without the heavy computational cost or ID-switching bugs of DeepSORT.

---

## Slide 12: Conclusion & Summary of Deliverables

### Summary of Key Deliverables
1. **Model:** YOLOv8n (3.14M params, 8.8 GFLOPs), validation $\text{mAP}_{50} = \mathbf{0.813}$, test $\text{mAP}_{50} = \mathbf{0.754}$.
2. **Logic Engine:** Pure-Python rule engine with IoG overlap, anatomical zones, and 4 reliability guards.
3. **Evaluation Benchmark:** 93.3% within $\pm 1$ violation accuracy on real-world test scenes.
4. **Application:** Streamlit industrial portal with live KPIs, CCTV video mode, anti-flicker smoothing, and CSV export.
5. **Quality Assurance:** 68 passing unit tests (100% pass rate in 1.44s).

---

### Questions & Answers (Viva Cheat Sheet)
* **Why YOLOv8n?** Real-time CPU edge execution with minimal compute footprint (8.8 GFLOPs).
* **Why IoG over IoU?** Resolves extreme scale disparity between small gear and large human bodies.
* **Why "Needs Review"?** Protects workers from false automated penalties in ambiguous visual conditions.
* **How does Video Smoothing work?** 7-frame rolling median filters out momentary head-turn drops.

---

# Thank You! 
### Questions & Discussion
`https://github.com/mibrahimshoaib56/ppe-compliance`
