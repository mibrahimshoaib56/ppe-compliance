"""End-to-end integration and sanity test for SAFETY VISION portal."""

import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from PIL import Image
import src.detector as det
import src.compliance as comp
import src.utils as utils

def run_tests():
    print("--- STARTING END-TO-END PIPELINE AUDIT ---")

    # 1. Test model loading and class verification
    model = det.load_detector_model()
    assert det.verify_model_classes(model) is True, "Model classes verification failed!"
    print("Step 1: Model classes verified (10/10 classes matched).")

    # 2. Test sample image inference & compliance
    sample_path = os.path.join("sample_images", "sample_5.jpg")
    img = Image.open(sample_path)
    detections = det.run_detection(img, model=model, conf_threshold=0.25, device="cpu")
    eval_result = comp.evaluate_site_compliance(detections, required_gear="both")
    workers = eval_result["workers"]
    summary = eval_result["summary"]
    rate = summary["compliance_rate"]
    status = summary["site_status"]
    print(f"Step 2: Sample 5 inference -> {len(workers)} workers, rate={rate}%, status={status}")

    # 3. Test annotation & encoding
    annotated = utils.annotate_image(img, workers, eval_result["unassigned_gear"])
    assert annotated.shape == (img.size[1], img.size[0], 3), "Annotated image shape mismatch"
    img_bytes = utils.encode_image_bytes(annotated)
    assert len(img_bytes) > 1000, "Encoded image too small"
    print(f"Step 3: Annotation generated, encoded {len(img_bytes)} bytes.")

    # 4. Test worker crop ROI
    worker_crop = utils.crop_worker_roi(img, workers[0], show_zones=True)
    assert worker_crop.ndim == 3 and worker_crop.shape[0] > 10 and worker_crop.shape[1] > 10
    print(f"Step 4: Worker ROI crop generated with shape {worker_crop.shape}.")

    # 5. Test CSV report generation
    csv_rep = utils.generate_csv_report(summary, workers, "2026-11-19 00:30:00")
    assert "CONSTRUCTION SITE PPE COMPLIANCE AUDIT REPORT" in csv_rep
    assert "Worker #1" in csv_rep
    print(f"Step 5: CSV report generated with {len(csv_rep.splitlines())} lines.")

    # 6. Test zero worker edge case
    zero_eval = comp.evaluate_site_compliance([], required_gear="both")
    assert zero_eval["summary"]["total_workers"] == 0
    assert zero_eval["summary"]["compliance_rate"] == 100.0
    assert zero_eval["summary"]["site_status"] == "Safe"
    print("Step 6: Zero worker edge case handled safely without division error.")

    print("--- ALL 6 PIPELINE AUDITS PASSED WITH ZERO ERRORS ---")

if __name__ == "__main__":
    run_tests()
