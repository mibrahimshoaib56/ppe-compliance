"""Evaluate detection and compliance counts across confidence slider thresholds.

Runs the YOLOv8 detector and compliance rules at confidence thresholds
0.15, 0.25, and 0.40 for all images in the specified folder. Prints per-image
and total metrics (workers, compliant, violations, needs review) and saves
the detailed results to eval/results/slider_check.csv.
"""

import argparse
import csv
import os
import sys
from typing import Any, Dict, List, Tuple

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from PIL import Image
import src.detector as det
import src.compliance as comp

# ============================================================================
# NAMED CONSTANTS & THRESHOLDS
# ============================================================================
CONFIDENCE_THRESHOLDS: List[float] = [0.15, 0.25, 0.40]
OUTPUT_CSV_PATH: str = os.path.join("eval", "results", "slider_check.csv")
SUPPORTED_EXTENSIONS: Tuple[str, ...] = (
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"
)


def run_slider_check(image_folder: str, output_csv: str = OUTPUT_CSV_PATH) -> List[Dict[str, Any]]:
    """Run detector and compliance across confidence thresholds for all images in folder.

    Args:
        image_folder: Directory containing test images.
        output_csv: Path to save the resulting CSV.

    Returns:
        List[Dict[str, Any]]: Evaluation records.
    """
    if not os.path.isdir(image_folder):
        print(f"Error: Provided folder does not exist or is not a directory: {image_folder}", file=sys.stderr)
        sys.exit(1)

    image_files = sorted([
        f for f in os.listdir(image_folder)
        if os.path.isfile(os.path.join(image_folder, f))
        and os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
    ])

    if not image_files:
        print(f"Error: No supported images found in folder: {image_folder}", file=sys.stderr)
        sys.exit(1)

    model = det.load_detector_model()
    records: List[Dict[str, Any]] = []

    print("=" * 86)
    print(f"EVALUATION: CONFIDENCE SLIDER CHECK ({len(image_files)} images, folder='{image_folder}')")
    print("=" * 86)
    print(f"{'Image':<26} {'Conf':<6} {'Workers':>8} {'Compliant':>10} {'Violations':>11} {'Needs Review':>13} {'Comp Rate':>10}")
    print("-" * 86)

    # Totals accumulator per confidence threshold
    totals: Dict[float, Dict[str, int]] = {
        c: {"workers": 0, "compliant": 0, "violations": 0, "needs_review": 0}
        for c in CONFIDENCE_THRESHOLDS
    }

    for fname in image_files:
        img_path = os.path.join(image_folder, fname)
        try:
            with Image.open(img_path) as img:
                img_rgb = img.convert("RGB")
                img_h = float(img_rgb.height)

                for conf in CONFIDENCE_THRESHOLDS:
                    detections = det.run_detection(img_rgb, model=model, conf_threshold=conf)
                    eval_result = comp.evaluate_site_compliance(
                        detections=detections,
                        required_gear="both",
                        image_height=img_h,
                    )
                    summary = eval_result["summary"]

                    workers = summary.get("total_workers", 0)
                    compliant = summary.get("compliant_count", 0)
                    violations = summary.get("violation_count", 0)
                    needs_review = summary.get("needs_review_count", summary.get("unknown_count", 0))
                    rate = summary.get("compliance_rate", 0.0)

                    records.append({
                        "image": fname,
                        "confidence": conf,
                        "workers": workers,
                        "compliant": compliant,
                        "violations": violations,
                        "needs_review": needs_review,
                        "compliance_rate": rate,
                    })

                    totals[conf]["workers"] += workers
                    totals[conf]["compliant"] += compliant
                    totals[conf]["violations"] += violations
                    totals[conf]["needs_review"] += needs_review

                    print(f"{fname:<26} {conf:<6.2f} {workers:>8} {compliant:>10} {violations:>11} {needs_review:>13} {rate:>9.1f}%")
        except Exception as e:
            print(f"Warning: Failed to process image '{fname}': {e}", file=sys.stderr)

    print("=" * 86)
    print(f"TOTALS ACROSS ALL {len(image_files)} IMAGES")
    print("=" * 86)
    print(f"{'Confidence':<12} {'Total Workers':>14} {'Total Compliant':>16} {'Total Violations':>17} {'Total Needs Review':>19}")
    print("-" * 86)
    for conf in CONFIDENCE_THRESHOLDS:
        t = totals[conf]
        print(f"{conf:<12.2f} {t['workers']:>14} {t['compliant']:>16} {t['violations']:>17} {t['needs_review']:>19}")
    print("=" * 86)

    # Save to CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["image", "confidence", "workers", "compliant", "violations", "needs_review", "compliance_rate"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"\nSaved results to: {output_csv} ({len(records)} rows)")
    return records


def main():
    parser = argparse.ArgumentParser(
        description="Run detector and compliance rules at confidence thresholds 0.15, 0.25, 0.40."
    )
    parser.add_argument(
        "folder",
        nargs="?",
        default=None,
        help="Path to folder containing images (e.g. sample_images).",
    )
    parser.add_argument(
        "--folder",
        dest="flag_folder",
        default=None,
        help="Optional flag for image folder path.",
    )
    parser.add_argument(
        "--output",
        default=OUTPUT_CSV_PATH,
        help=f"Path to output CSV (default: {OUTPUT_CSV_PATH}).",
    )
    args = parser.parse_args()

    target_folder = args.folder or args.flag_folder
    if not target_folder:
        print("Error: Image folder argument is required.", file=sys.stderr)
        print("Usage: python eval/slider_check.py <image_folder>", file=sys.stderr)
        sys.exit(1)

    run_slider_check(image_folder=target_folder, output_csv=args.output)


if __name__ == "__main__":
    main()
