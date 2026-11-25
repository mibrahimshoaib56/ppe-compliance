# Documentation Audit & Changes Log

This log documents every correction, replacement, and deletion made during the documentation audit of `REPORT.md`, `README.md`, and `SLIDES.md`.

**Sources of Truth Used:**
1. `eval/results/test_metrics.txt`
2. `eval/results/results.csv`
3. `eval/results/score_images_report.txt`
4. `eval/results/score_images.csv`
5. `eval/results/slider_check.csv`
6. Kaggle Training Log Facts (provided in prompt)

*(Note: `eval/results/args.yaml` was missing from `eval/results/` and was not used).*

---

## 1. Corrections in `REPORT.md`

| # | Item / Topic | Old Value or Text | New Value or Text | Source File / Status |
|---|---|---|---|---|
| 1 | Model Architecture & Size | "130 layers, 3.14M parameters (3,140,190), 8.8 GFLOPs" | "129 layers, 3,012,798 parameters, 8.2 GFLOPs as trained (73 layers, 3,134,894 parameters when fused)" | Kaggle Training Log Facts |
| 2 | Optimizer & Hyperparameters | "Auto (SGD with momentum 0.937, weight decay 0.0005), lr 0.01 with cosine scheduling" | "AdamW (auto-selected, learning rate 0.000714, momentum 0.9)" | Kaggle Training Log Facts |
| 3 | Training & Validation Losses | Box loss: 0.984, Cls loss: 0.812, DFL loss: 0.998; Val Box: 1.052, Val Cls: 0.887 | Epoch 100 Train losses: Box 0.7619, Cls 0.5279, DFL 1.03; Val losses: Box 1.281, Cls 0.714, DFL 1.339 | Kaggle Training Log Facts & `results.csv` (last row) |
| 4 | Validation mAP at Epoch 100 | "Validation mAP50: 0.813" (as epoch 100 value) | "Validation at epoch 100: mAP50 0.814, mAP50-95 0.512 (best.pt: 0.813 / 0.511)" | Kaggle Training Log Facts & `results.csv` |
| 5 | Speed Metrics | "Inference in under 80 ms per frame on CPU", "15–25 FPS on modern multi-core CPU", ">30 FPS" | "1.8 ms preprocess, 6.8 ms inference, 2.3 ms postprocess per image on Tesla T4 GPU. No CPU timing was measured." | `test_metrics.txt` (CPU claims deleted) |
| 6 | Per-Class Test Split Breakdown | Person (P: 0.892, R: 0.841, AP50: 0.891), Hardhat (P: 0.915, R: 0.826, AP50: 0.884), Vest (P: 0.884, R: 0.792, AP50: 0.852), NO-Vest (P: 0.812, R: 0.680, AP50: 0.718), NO-Hardhat (P: 0.724, R: 0.561, AP50: 0.545) | Person (P: 0.884, R: 0.786, AP50: 0.857), Hardhat (P: 0.959, R: 0.809, AP50: 0.905), Safety Vest (P: 0.896, R: 0.849, AP50: 0.902), NO-Safety Vest (P: 0.930, R: 0.739, AP50: 0.799), NO-Hardhat (P: 0.807, R: 0.561, AP50: 0.545) | `test_metrics.txt` |
| 7 | NO-Hardhat Explanation | Stated as factual root cause | Replaced with "is likely due to..." | Prompt instruction |
| 8 | Evaluation Photos Source & Labels | "15 independent, out-of-distribution construction photos sourced from diverse job sites; Human safety audits established ground-truth counts" | "15 test photos collected from Google Images (original pages not recorded); ground-truth worker counts established by the group by visual inspection" | Prompt instruction & `SOURCES.md` |
| 9 | Detection Rate Wording | "Detection Rate: 78.4%" | "69 workers detected vs 88 in the ground truth (counts only; detections were not matched one by one)" | Prompt instruction |
| 10 | Predict Zero Violations Baseline | None | Added: "A naive baseline that predicts zero violations everywhere would be within +-1 on 12 of 15 images (80.0%), because the ground truth has only 7 violations in 88 workers (photo06=2, photo07=1, photo12=2, photo13=2, and 0 elsewhere)" | Prompt instruction & `score_images_report.txt` |
| 11 | Confidence Slider Table & Text | 0.15: 80 / 27 / 11 / 42; 0.25: 69 / 25 / 12 / 32; 0.40: 60 / 21 / 18 / 21; called 0.25 "Optimal Operating Point: Balanced detection with highest violation precision" | 0.15: 83 / 27 / 9 / 47 (rate 56.6%); 0.25: 69 / 25 / 12 / 32 (rate 46.4%); 0.40: 58 / 20 / 18 / 20 (rate 34.5%); stated: "Confidence 0.25 represents the least-bad trade-off on these 15 photos" | Kaggle Training Log Facts (slider actual values) |
| 12 | Video Resizing & Benchmarking Claim | "Benchmarking proved 416 px achieves ~30% faster CPU throughput than 640 px" | Deleted claim; plainly stated: "Frames are resized proportionally to a maximum width of 416 px." | Deletion (untraceable benchmark claim) |
| 13 | Temporal Smoothing Description | Described vaguely as real-time anti-flicker | Described accurately: 7-frame rolling median running in a post-processing pass over frames before and after current frame ($i-3$ to $i+3$) after video decoding | `src/video.py` code inspection |
| 14 | Photo06 Plain Garments Claim | "2 road workers in solid garments were flagged as Missing vest" | Grounded strictly in data: "In score_images_report.txt, photo06 has 3 true compliant and 2 true violations, whereas the pipeline predicted 1 compliant, 3 violations, and 1 needs review. The model was trained on dataset annotations where safety vests include reflective striping." | `score_images_report.txt` |
| 15 | Validation Split Claims | "natural construction scenes, clean annotations" | Replaced with "114 images (split counts verified; images not individually inspected)" | Kaggle Training Log Facts |
| 16 | Small Dataset Limitation | None | Added limitation: "The evaluation set contains only 15 photos with 88 workers and 7 true violations, which limits statistical significance." | Prompt instruction |
| 17 | Tone & Vocabulary Deletions | Deleted all occurrences of: "real-time", "publication-grade", "enterprise-grade", "industrial", "audit", "60% of construction site fatalities", "Viva Defense Guide", "Viva Insight", and scripted Q&A | Replaced with plain descriptive terms: "inspection", "review", "record", "surveillance" | Prompt instruction |

---

## 2. Corrections in `README.md`

| # | Item / Topic | Old Value or Text | New Value or Text | Source File / Status |
|---|---|---|---|---|
| 1 | Python Version & Badges | "Tested on Python 3.12", `68/68 passing`, `3.14M params` | "Python 3.12.10", `68 passed`, `3.01M params trained` | `python --version`, `pytest -q`, Kaggle Log Facts |
| 2 | Model Weights Storage Note | Listed in tree as if tracked in Git | Added notice: "The trained model weights (models/best.pt) are not stored in the Git repository due to file size considerations; they are included in the project submission zip." | Prompt instruction |
| 3 | Model Architecture Numbers | "130 layers, 3.14M parameters" | "129 layers, 3,012,798 parameters as trained; 73 layers, 3,134,894 parameters when fused" | Kaggle Training Log Facts |
| 4 | Speed & Per-Class Metrics | Omitted per-class test metrics; had unmeasured CPU claims | Added exact test metrics from `test_metrics.txt` (P: 0.874, R: 0.702, mAP50: 0.754) and T4 GPU speeds (1.8 ms / 6.8 ms / 2.3 ms). Explicitly noted: "No CPU timing measured". | `test_metrics.txt` |
| 5 | Benchmark Results & Baseline | Omitted zero-violation baseline and complaint within $\pm 1$ metric | Added: exact match 13.3%, within $\pm 1$ violations 93.3% (14/15), compliant 26.7% (4/15), MAE violations 0.60, compliant 3.73, and zero-violation baseline 80.0% (12/15) | `score_images_report.txt` & Prompt |
| 6 | Evaluation Photos Source | Stated as generic test photos | Stated: "Evaluation photos collected from Google Images; labels made by the group by visual inspection." Added small dataset limitation. | `SOURCES.md` & Prompt |
| 7 | Tone & Vocabulary Deletions | Deleted: "real-time", "publication-grade", "enterprise-grade", "industrial", "audit", "over 60% of construction site fatalities" | Replaced with plain language | Prompt instruction |

---

## 3. Corrections in `SLIDES.md`

| # | Item / Topic | Old Value or Text | New Value or Text | Source File / Status |
|---|---|---|---|---|
| 1 | Model Numbers & Layer Counts | "3,140,190 Parameters (~3.14M weights), 8.8 GFLOPs, 130 layers" | "3,012,798 Parameters (129 layers) as trained; 3,134,894 parameters (73 layers) when fused. 8.2 GFLOPs as trained (8.6 GFLOPs fused)." | Kaggle Training Log Facts |
| 2 | Optimizer | "SGD ($\eta_0 = 0.01$, momentum 0.937)" | "AdamW (auto-selected, lr 0.000714, momentum 0.9)" | Kaggle Training Log Facts |
| 3 | CPU Latency Claims | "<80 ms per frame on standard laptop CPU", ">30 FPS on GPU, ~15–20 FPS on CPU", "~12–15 FPS on CPU" | Deleted CPU timing claims. Added T4 GPU speeds (1.8 ms / 6.8 ms / 2.3 ms). Explicitly noted: "No CPU timing measured." | `test_metrics.txt` |
| 4 | Per-Class Test Table | Old mAP values without exact AP50 from test_metrics | Exact values from `test_metrics.txt`: Hardhat (P 0.959, R 0.809, AP50 0.905), Vest (P 0.896, R 0.849, AP50 0.902), Person (P 0.884, R 0.786, AP50 0.857), NO-Vest (P 0.930, R 0.739, AP50 0.799), NO-Hardhat (P 0.807, R 0.561, AP50 0.545) | `test_metrics.txt` |
| 5 | Validation Split Description | "Natural photographic construction scenes" | "114 images (counts verified; not individually inspected)" | Kaggle Training Log Facts |
| 6 | Evaluation Benchmark Metrics | Omitted compliant $\pm 1$ accuracy and zero-violation baseline | Added: exact match 13.3%, compliant within $\pm 1$ 26.7%, zero-violation baseline 80.0% (12/15 images) | `score_images_report.txt` & Prompt |
| 7 | Slider Sweep Table & Notes | Old values; called 0.25 "optimal" | Exact values: 0.15 (83/27/9/47), 0.25 (69/25/12/32), 0.40 (58/20/18/20); stated: "Confidence 0.25 represents the least-bad trade-off on these 15 photos." | Kaggle Training Log Facts |
| 8 | Temporal Smoothing Description | Described as real-time smoothing | Described as 7-frame rolling median post-processing pass over surrounding frames ($i-3$ to $i+3$) after video decoding | `src/video.py` code inspection |
| 9 | Deleted Sections & Wording | Deleted: "60% of construction fatalities", "Viva Defense Guide", scripted Q&A, and terms: "real-time", "publication-grade", "enterprise", "industrial", "audit" | Replaced with plain descriptive terms | Prompt instruction |
