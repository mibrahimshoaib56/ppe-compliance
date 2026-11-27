# Failure Case Studies

This folder contains visual evidence of key failure modes identified during pipeline evaluation on construction scenes. These cases represent real-world domain challenges for object detection and compliance rule engines.

---

### 1. Overhead Camera Angle (`01_overhead_view_ppe10.jpg`)
* **Source Image:** `sample_images/ppe10.jpg`
* **Visual Content:** High-angle steep overhead perspective of an equipment operator beside industrial machinery.
* **Empirical Execution (`eval/results/ppe10_check.txt`):**
  - Raw YOLO inference at base confidence 0.10 detects two Person candidates (confidences 0.268 and 0.148), one machinery (0.454), one NO-Safety Vest (0.451), and one Safety Vest (0.234).
  - Both Person detections fail the `CONF_PERSON = 0.30` per-class gate and are dropped upstream in `run_detection`.
  - Consequently, zero Person bounding boxes enter `evaluate_site_compliance()`, resulting in 0 detected workers.
  - The `MIN_UPRIGHT_ASPECT = 1.2` guard is not reached because candidate figures are discarded upstream by the confidence gate.

---

### 2. Bag, Harness, and Posture Occlusion (`02_bag_rebar_occlusion_photo03.jpg`)
* **Source Image:** `eval/test_photos/photo03.jpg`
* **Visual Content:** Work crew gathered on a reinforced construction deck with background scaffolding, tool pouches, and rebar structures partially occluding torso regions.
* **Empirical Diagnostic (`eval/results/score_images_report.txt`):**
  - **True Counts:** 10 compliant, 0 violations (10 true workers).
  - **Predicted Counts:** 3 compliant, 0 violations, 6 total workers detected, 3 routed to "Needs Review" (compliance rate: 50.0%).
  - **Needs Review Breakdown (3 workers):** 1 `small` ($< 12\%$ frame height), 1 `view not upright` ($h/w < 1.2$), and 1 `conflicting detections`.
  - **Diagnostic Status:** Compliant Mismatch (3 predicted vs 10 true compliant workers).

---

### 3. Plain Vest vs. Reflective Tape (`03_plain_vest_photo06.jpg`)
* **Source Image:** `eval/test_photos/photo06.jpg`
* **Visual Content:** Five construction workers on an active roadway worksite, two wearing everyday clothing and three wearing fluorescent solid jackets lacking silver retroreflective stripes.
* **Empirical Diagnostic (`eval/results/score_images_report.txt`):**
  - **True Counts:** 3 compliant, 2 violations (5 true workers).
  - **Predicted Counts:** 1 compliant, 3 violations, 5 total workers detected, 1 routed to "Needs Review" (compliance rate: 20.0%).
  - **Needs Review Breakdown (1 worker):** 1 `conflicting detections`.
  - **Diagnostic Status:** Compliant Mismatch, Violation Mismatch.
