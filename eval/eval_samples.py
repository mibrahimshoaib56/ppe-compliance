"""Evaluation test runner for sample construction site dataset images."""

import os
import sys

# Ensure repository root is on Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from PIL import Image
import src.detector as det
import src.compliance as comp

def main():
    model = det.load_detector_model()
    is_valid = det.verify_model_classes(model)
    print(f"Model Path: {det.resolve_model_path()}")
    print(f"Verified 10 Dataset Classes: {is_valid}")
    print(f"Model Class Names: {model.names}\n")

    sample_dir = "sample_images"
    sample_files = sorted([f for f in os.listdir(sample_dir) if f.endswith(".jpg")])

    for fname in sample_files:
        img_path = os.path.join(sample_dir, fname)
        img = Image.open(img_path)
        detections = det.run_detection(img, model=model, conf_threshold=0.25)
        eval_result = comp.evaluate_site_compliance(detections, required_gear="both")
        summary = eval_result["summary"]

        print(f"==================================================")
        print(f"Image: {fname} ({img.size[0]}x{img.size[1]})")
        print(f"  Raw Detections ({len(detections)}): {[d['class_name'] + ' (' + str(d['confidence']) + ')' for d in detections]}")
        print(f"  Site Summary:")
        print(f"    - Total Workers: {summary['total_workers']}")
        print(f"    - Compliant: {summary['compliant_count']}")
        print(f"    - Violations: {summary['violation_count']}")
        print(f"    - Inconclusive/Unknown: {summary['unknown_count']}")
        print(f"    - Compliance Rate: {summary['compliance_rate']}%")
        print(f"    - Site Status: {summary['site_status']}")

        if eval_result["workers"]:
            print("  Worker Breakdown:")
            for w in eval_result["workers"]:
                assigned = [g["class_name"] for g in w["assigned_gear"]]
                print(
                    f"    - Worker #{w['worker_id']}: Status={w['status']} | "
                    f"Hardhat={w['hardhat_status']} | Vest={w['vest_status']} | "
                    f"Assigned Gear={assigned}"
                )
        if eval_result["unassigned_gear"]:
            unassigned_names = [g["class_name"] for g in eval_result["unassigned_gear"]]
            print(f"  Unassigned Gear: {unassigned_names}")
        print()

if __name__ == "__main__":
    main()
