# Failure Case Studies

This folder contains visual evidence of key failure modes identified during pipeline evaluation on construction scenes. These cases represent real-world domain challenges for object detection and compliance rule engines.

---

### 1. Overhead Camera Angle (`01_overhead_view_ppe10.jpg`)
* **Source Image:** `sample_images/ppe10.jpg`
* **Phenomenon:** High-angle or bird's-eye camera perspective looking down at worker from above.
* **Failure Mode:**
  - Standard person detectors fail or produce low confidence scores because training data consists predominantly of eye-level upright persons.
  - The bounding box aspect ratio ($h / w$) is squashed to $\approx 1.03$, collapsing the head and torso anatomical zones into the same vertical slice.
* **System Safeguard:**
  - The `MIN_UPRIGHT_ASPECT = 1.2` guard activates.
  - Rather than making an erroneous violation or compliance claim, the worker is safely flagged as:
    `Needs review: view not upright`.

---

### 2. Bag, Harness, and Posture Occlusion (`02_bag_rebar_occlusion_photo03.jpg`)
* **Source Image:** `eval/test_photos/photo03.jpg`
* **Phenomenon:** Workers bending over rebar grids wearing tool belts, equipment harnesses, or carrying materials.
* **Failure Mode:**
  - The torso zone is partially obscured by straps, bags, or construction materials, causing both `Safety Vest` and `NO-Safety Vest` to trigger simultaneously or fail minimum overlap.
* **System Safeguard:**
  - Torso zone was widened to $15\% - 75\%$ of body height.
  - Ambiguous/conflicting detections trigger conflict handling:
    `Needs review: conflicting detections`.
  - Distant workers (< $12\%$ image height) are routed to:
    `Needs review: small`.

---

### 3. Plain Vest vs. Reflective Tape (`03_plain_vest_photo06.jpg`)
* **Source Image:** `eval/test_photos/photo06.jpg`
* **Phenomenon:** Workers wearing solid-color fluorescent vests or shirts without retro-reflective silver stripes.
* **Failure Mode:**
  - The dataset (`css-data`) labels high-visibility garments with silver reflective bands as `Safety Vest`. Solid neon clothing without reflectors is either unannotated or tagged as `NO-Safety Vest`.
  - As a result, 2 road workers in solid garments are flagged as:
    `Missing vest` (Violation).
* **Viva / Report Insight:**
  - Highlights the dataset labeling policy boundary: compliance is enforced strictly on reflective safety gear, not plain fluorescent clothing.
