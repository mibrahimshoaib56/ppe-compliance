"""Evaluate detection and compliance counts across confidence slider thresholds.

Runs the YOLOv8 detector and compliance rules at confidence thresholds
0.15, 0.25, and 0.40 for all images in the specified folder. Prints per-image
and total metrics (workers, compliant, violations, needs review with breakdown
reasons) and saves the detailed results to eval/results/slider_check.csv.
"""

import argparse
import csv
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

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

REASON_CATEGORIES: List[str] = [
    "small",
    "low confidence",
    "view not upright",
    "conflicting detections",
    "hardhat unknown",
    "vest unknown",
    "other",
]


def classify_needs_review_reason(worker: Dict[str, Any]) -> str:
    """Classify a Needs Review worker into one of the 7 diagnostic categories.

    Categories:
        - small
        - low confidence
        - view not upright
        - conflicting detections
        - hardhat unknown
        - vest unknown
        - other
    """
    stat = str(worker.get("status", "")).strip().lower()
    rel = str(worker.get("reliability_reason", "")).strip().lower()
    h_reason = str(worker.get("hardhat_reason", "")).strip().lower()
    v_reason = str(worker.get("vest_reason", "")).strip().lower()
    h_stat = str(worker.get("hardhat_status", "")).strip().lower()
    v_stat = str(worker.get("vest_status", "")).strip().lower()

    if rel == "small" or "small" in stat:
        return "small"
    if rel == "low confidence" or "low confidence" in stat:
        return "low confidence"
    if rel == "view not upright" or "view not upright" in stat:
        return "view not upright"

    if (
        h_reason == "conflicting detections"
        or v_reason == "conflicting detections"
        or "conflicting detections" in stat
    ):
        return "conflicting detections"

    if "vest not visible" in stat or (v_stat == "unknown" and h_stat != "unknown"):
        return "vest unknown"

    if h_stat == "unknown" and v_stat != "unknown":
        return "hardhat unknown"

    if v_stat == "unknown":
        return "vest unknown"
    if h_stat == "unknown":
        return "hardhat unknown"

    return "other"


def is_needs_review(worker: Dict[str, Any]) -> bool:
    """Check if worker status corresponds to Needs Review."""
    stat = str(worker.get("status", ""))
    return (
        stat in ("Vest not visible: manual check", "Unknown")
        or stat.startswith("Needs review:")
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

    print("=" * 132)
    print(f"EVALUATION: CONFIDENCE SLIDER CHECK ({len(image_files)} images, folder='{image_folder}')")
    print("=" * 132)
    print(
        f"{'Image':<16} {'Conf':<5} {'Wk':>4} {'Comp':>5} {'Viol':>5} {'NR':>4} {'Rate':>7} | "
        f"{'small':>6} {'low_conf':>9} {'not_uprt':>9} {'conflict':>9} {'hdhat_unk':>10} {'vest_unk':>9} {'other':>6}"
    )
    print("-" * 132)

    # Totals accumulator per confidence threshold
    totals: Dict[float, Dict[str, int]] = {
        c: {
            "workers": 0,
            "compliant": 0,
            "violations": 0,
            "needs_review": 0,
            **{cat: 0 for cat in REASON_CATEGORIES},
        }
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
                    workers_list = eval_result.get("workers", [])

                    workers = summary.get("total_workers", len(workers_list))
                    compliant = summary.get("compliant_count", 0)
                    violations = summary.get("violation_count", 0)
                    needs_review = summary.get("needs_review_count", summary.get("unknown_count", 0))

                    # Compliance rate: show N/A when 0 workers
                    if workers == 0:
                        rate_display = "N/A"
                        rate_val = "N/A"
                    else:
                        calc_rate = round((compliant / workers) * 100.0, 1)
                        rate_display = f"{calc_rate:>6.1f}%"
                        rate_val = calc_rate

                    # Break down needs review by reason
                    reasons_count = {cat: 0 for cat in REASON_CATEGORIES}
                    for w in workers_list:
                        if is_needs_review(w):
                            cat = classify_needs_review_reason(w)
                            reasons_count[cat] += 1

                    records.append({
                        "image": fname,
                        "confidence": conf,
                        "workers": workers,
                        "compliant": compliant,
                        "violations": violations,
                        "needs_review": needs_review,
                        "compliance_rate": rate_val,
                        "small": reasons_count["small"],
                        "low_confidence": reasons_count["low confidence"],
                        "view_not_upright": reasons_count["view not upright"],
                        "conflicting_detections": reasons_count["conflicting detections"],
                        "hardhat_unknown": reasons_count["hardhat unknown"],
                        "vest_unknown": reasons_count["vest unknown"],
                        "other": reasons_count["other"],
                    })

                    totals[conf]["workers"] += workers
                    totals[conf]["compliant"] += compliant
                    totals[conf]["violations"] += violations
                    totals[conf]["needs_review"] += needs_review
                    for cat in REASON_CATEGORIES:
                        totals[conf][cat] += reasons_count[cat]

                    print(
                        f"{fname:<16} {conf:<5.2f} {workers:>4} {compliant:>5} {violations:>5} {needs_review:>4} {rate_display:>7} | "
                        f"{reasons_count['small']:>6} "
                        f"{reasons_count['low confidence']:>9} "
                        f"{reasons_count['view not upright']:>9} "
                        f"{reasons_count['conflicting detections']:>9} "
                        f"{reasons_count['hardhat unknown']:>10} "
                        f"{reasons_count['vest unknown']:>9} "
                        f"{reasons_count['other']:>6}"
                    )
        except Exception as e:
            print(f"Warning: Failed to process image '{fname}': {e}", file=sys.stderr)

    print("=" * 132)
    print(f"TOTALS ACROSS ALL {len(image_files)} IMAGES BY CONFIDENCE SETTING")
    print("=" * 132)
    print(
        f"{'Conf':<6} {'Workers':>8} {'Compliant':>10} {'Violations':>11} {'Needs Review':>13} | "
        f"{'small':>6} {'low_conf':>9} {'not_uprt':>9} {'conflict':>9} {'hdhat_unk':>10} {'vest_unk':>9} {'other':>6}"
    )
    print("-" * 132)
    for conf in CONFIDENCE_THRESHOLDS:
        t = totals[conf]
        print(
            f"{conf:<6.2f} {t['workers']:>8} {t['compliant']:>10} {t['violations']:>11} {t['needs_review']:>13} | "
            f"{t['small']:>6} "
            f"{t['low confidence']:>9} "
            f"{t['view not upright']:>9} "
            f"{t['conflicting detections']:>9} "
            f"{t['hardhat unknown']:>10} "
            f"{t['vest unknown']:>9} "
            f"{t['other']:>6}"
        )
    print("=" * 132)
    print("Note: Comp Rate = compliant / all detected workers, including needs review")
    print("=" * 132)

    # Save to CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    fieldnames = [
        "image",
        "confidence",
        "workers",
        "compliant",
        "violations",
        "needs_review",
        "small",
        "low_confidence",
        "view_not_upright",
        "conflicting_detections",
        "hardhat_unknown",
        "vest_unknown",
        "other",
        "compliance_rate",
    ]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
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
        help="Path to folder containing images (e.g. sample_images or eval/test_photos).",
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
