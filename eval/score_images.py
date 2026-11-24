"""Score detection and compliance pipeline predictions against ground-truth labels.

Reads ground truth CSV (with columns: image, true_compliant, true_violations)
and runs the pipeline at the default confidence threshold (0.25). Computes and
displays:
1. Image-level accuracy (both compliant and violation counts match exactly)
2. Within ±1 accuracy for both compliant and violation counts
3. Mean Absolute Error (MAE) for compliant count and violation count
4. Total detected workers vs total true workers
5. Judged workers (compliant + violations) vs detected workers
6. Needs-review rate and breakdown of needs-review workers by reason
7. Violations flagged and upper bound of violations caught: sum(min(pred, true))
8. Detailed per-image and totals diagnostic report
9. Structured table of wrong/mismatched images

Saves evaluation report and CSV to eval/results/.
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

REQUIRED_COLUMNS = ["image", "true_compliant", "true_violations"]
DEFAULT_CONFIDENCE: float = 0.25
DEFAULT_GT_PATH: str = os.path.join("eval", "ground_truth.csv")
OUTPUT_CSV_PATH: str = os.path.join("eval", "results", "score_images.csv")
OUTPUT_REPORT_PATH: str = os.path.join("eval", "results", "score_images_report.txt")

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


def score_images(
    image_folder: str,
    ground_truth_path: str,
    conf_threshold: float = DEFAULT_CONFIDENCE,
    output_csv: str = OUTPUT_CSV_PATH,
    output_report: str = OUTPUT_REPORT_PATH,
):
    """Evaluate pipeline predictions against ground truth labels.

    Args:
        image_folder: Directory containing image files.
        ground_truth_path: Path to ground truth CSV.
        conf_threshold: Inference confidence threshold (default: 0.25).
        output_csv: Path to save per-image scoring CSV.
        output_report: Path to save formatted text report.
    """
    # 1. Validate image folder
    if not os.path.isdir(image_folder):
        print(f"Error: Image folder not found or is not a directory: '{image_folder}'", file=sys.stderr)
        sys.exit(1)

    # 2. Validate ground truth CSV existence
    if not os.path.isfile(ground_truth_path):
        print(f"Error: Ground truth CSV file not found: '{ground_truth_path}'", file=sys.stderr)
        sys.exit(1)

    # 3. Read and validate CSV columns
    rows: List[Dict[str, str]] = []
    with open(ground_truth_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        # Check required columns
        missing_cols = [col for col in REQUIRED_COLUMNS if col not in fieldnames]
        if missing_cols:
            print(
                f"Error: Ground truth CSV is missing required column(s): {missing_cols}.\n"
                f"Required columns: {REQUIRED_COLUMNS}\n"
                f"Found columns: {fieldnames}",
                file=sys.stderr,
            )
            sys.exit(1)

        for row in reader:
            rows.append(row)

    if not rows:
        print(f"Error: Ground truth CSV '{ground_truth_path}' contains zero data rows.", file=sys.stderr)
        sys.exit(1)

    # 4. Validate that all listed images exist and have valid integer counts
    validated_data: List[Dict[str, Any]] = []
    for line_num, r in enumerate(rows, start=2):  # line 1 is header
        img_name = r.get("image", "").strip()
        if not img_name:
            print(f"Error: Row {line_num} in '{ground_truth_path}' has an empty 'image' field.", file=sys.stderr)
            sys.exit(1)

        img_path = os.path.join(image_folder, img_name)
        if not os.path.isfile(img_path):
            print(
                f"Error: Image '{img_name}' listed in CSV (row {line_num}) was not found in folder '{image_folder}'.",
                file=sys.stderr,
            )
            sys.exit(1)

        try:
            raw_comp = r["true_compliant"].strip()
            raw_viol = r["true_violations"].strip()
            if raw_comp == "" or raw_viol == "":
                raise ValueError("Count values cannot be empty")
            true_comp = int(raw_comp)
            true_viol = int(raw_viol)
            if true_comp < 0 or true_viol < 0:
                raise ValueError("Counts cannot be negative")
        except (ValueError, KeyError) as e:
            print(
                f"Error: Invalid count on row {line_num} for image '{img_name}': {e}. "
                f"Values must be non-negative integers.",
                file=sys.stderr,
            )
            sys.exit(1)

        validated_data.append({
            "image": img_name,
            "path": img_path,
            "true_compliant": true_comp,
            "true_violations": true_viol,
            "true_workers": true_comp + true_viol,
        })

    # 5. Run pipeline inference at requested confidence
    model = det.load_detector_model()

    out_lines: List[str] = []

    def log(msg: str = ""):
        print(msg)
        out_lines.append(msg)

    log("=" * 140)
    log("SCORING PPE COMPLIANCE PIPELINE AGAINST GROUND TRUTH")
    log("=" * 140)
    log(f"Image Folder:        {image_folder}")
    log(f"Ground Truth CSV:    {ground_truth_path}")
    log(f"Images to Evaluate:  {len(validated_data)}")
    log(f"Confidence Setting:  {conf_threshold}")
    log("-" * 140)

    # Per-image diagnostic table
    log(
        f"{'Image':<16} {'True C':>6} {'Pred C':>6} {'True V':>6} {'Pred V':>6} {'True W':>6} {'Pred W':>6} {'NR':>4} {'Comp Rate':>9} | "
        f"{'small':>5} {'low_cf':>6} {'uprt':>5} {'conf':>5} {'h_unk':>5} {'v_unk':>5} {'oth':>4}"
    )
    log("-" * 140)

    correct_images_count = 0
    within_1_compliant_count = 0
    within_1_violations_count = 0
    total_abs_err_compliant = 0.0
    total_abs_err_violations = 0.0

    total_pred_workers = 0
    total_true_workers = 0
    total_pred_compliant = 0
    total_true_compliant = 0
    total_pred_violations = 0
    total_true_violations = 0
    total_pred_needs_review = 0
    violations_caught_upper_bound = 0

    total_reason_counts: Dict[str, int] = {cat: 0 for cat in REASON_CATEGORIES}
    wrong_images: List[Dict[str, Any]] = []
    per_image_records: List[Dict[str, Any]] = []

    for item in validated_data:
        fname = item["image"]
        img_path = item["path"]
        t_comp = item["true_compliant"]
        t_viol = item["true_violations"]
        t_work = item["true_workers"]

        with Image.open(img_path) as img:
            img_rgb = img.convert("RGB")
            img_h = float(img_rgb.height)

            detections = det.run_detection(img_rgb, model=model, conf_threshold=conf_threshold)
            eval_res = comp.evaluate_site_compliance(
                detections=detections,
                required_gear="both",
                image_height=img_h,
            )
            summary = eval_res["summary"]
            workers_list = eval_res.get("workers", [])

            p_comp = summary.get("compliant_count", 0)
            p_viol = summary.get("violation_count", 0)
            p_nr = summary.get("needs_review_count", summary.get("unknown_count", 0))
            p_work = summary.get("total_workers", len(workers_list))

        err_comp = abs(p_comp - t_comp)
        err_viol = abs(p_viol - t_viol)
        total_abs_err_compliant += err_comp
        total_abs_err_violations += err_viol

        if p_comp == t_comp and p_viol == t_viol:
            correct_images_count += 1
        else:
            wrong_images.append({
                "image": fname,
                "true_compliant": t_comp,
                "pred_compliant": p_comp,
                "true_violations": t_viol,
                "pred_violations": p_viol,
                "true_workers": t_work,
                "pred_workers": p_work,
                "needs_review": p_nr,
            })

        if err_comp <= 1:
            within_1_compliant_count += 1
        if err_viol <= 1:
            within_1_violations_count += 1

        # Upper bound of caught violations for this image: min(pred, true)
        caught_ub = min(p_viol, t_viol)
        violations_caught_upper_bound += caught_ub

        # Compliance rate display: N/A if zero workers
        if p_work == 0:
            rate_display = "N/A"
            rate_val = "N/A"
        else:
            calc_rate = round((p_comp / p_work) * 100.0, 1)
            rate_display = f"{calc_rate:>8.1f}%"
            rate_val = calc_rate

        # Reasons breakdown
        reasons_count = {cat: 0 for cat in REASON_CATEGORIES}
        for w in workers_list:
            if is_needs_review(w):
                cat = classify_needs_review_reason(w)
                reasons_count[cat] += 1
                total_reason_counts[cat] += 1

        # Accumulate totals
        total_pred_workers += p_work
        total_true_workers += t_work
        total_pred_compliant += p_comp
        total_true_compliant += t_comp
        total_pred_violations += p_viol
        total_true_violations += t_viol
        total_pred_needs_review += p_nr

        per_image_records.append({
            "image": fname,
            "true_compliant": t_comp,
            "pred_compliant": p_comp,
            "true_violations": t_viol,
            "pred_violations": p_viol,
            "true_workers": t_work,
            "pred_workers": p_work,
            "pred_needs_review": p_nr,
            "compliance_rate": rate_val,
            "small": reasons_count["small"],
            "low_confidence": reasons_count["low confidence"],
            "view_not_upright": reasons_count["view not upright"],
            "conflicting_detections": reasons_count["conflicting detections"],
            "hardhat_unknown": reasons_count["hardhat unknown"],
            "vest_unknown": reasons_count["vest unknown"],
            "other": reasons_count["other"],
        })

        log(
            f"{fname:<16} {t_comp:>6} {p_comp:>6} {t_viol:>6} {p_viol:>6} {t_work:>6} {p_work:>6} {p_nr:>4} {rate_display:>9} | "
            f"{reasons_count['small']:>5} "
            f"{reasons_count['low confidence']:>6} "
            f"{reasons_count['view not upright']:>5} "
            f"{reasons_count['conflicting detections']:>5} "
            f"{reasons_count['hardhat unknown']:>5} "
            f"{reasons_count['vest unknown']:>5} "
            f"{reasons_count['other']:>4}"
        )

    total_images = len(validated_data)
    exact_accuracy_pct = (correct_images_count / total_images) * 100.0
    within_1_comp_pct = (within_1_compliant_count / total_images) * 100.0
    within_1_viol_pct = (within_1_violations_count / total_images) * 100.0
    mae_compliant = total_abs_err_compliant / total_images
    mae_violations = total_abs_err_violations / total_images

    total_judged_workers = total_pred_compliant + total_pred_violations
    judged_rate_pct = (total_judged_workers / total_pred_workers * 100.0) if total_pred_workers > 0 else 0.0
    needs_review_rate_pct = (total_pred_needs_review / total_pred_workers * 100.0) if total_pred_workers > 0 else 0.0

    log("=" * 140)
    log("DIAGNOSTIC SUMMARY & EVALUATION METRICS")
    log("=" * 140)
    log(f"  • Total Images Evaluated:            {total_images}")
    log(f"  • Exact Match Accuracy (Both Counts): {correct_images_count} / {total_images} ({exact_accuracy_pct:.1f}%)")
    log(f"  • Within ±1 Accuracy (Compliant):    {within_1_compliant_count} / {total_images} ({within_1_comp_pct:.1f}%)")
    log(f"  • Within ±1 Accuracy (Violations):   {within_1_violations_count} / {total_images} ({within_1_viol_pct:.1f}%)")
    log(f"  • MAE (Compliant Count):             {mae_compliant:.2f}")
    log(f"  • MAE (Violations Count):            {mae_violations:.2f}")
    log("-" * 140)
    log(f"  • Total Detected vs True Workers:    {total_pred_workers} detected vs {total_true_workers} true (diff: {total_pred_workers - total_true_workers:+d})")
    log(f"  • Judged Workers (Compliant+Viol):   {total_judged_workers} / {total_pred_workers} detected ({judged_rate_pct:.1f}%)")
    log(f"  • Needs-Review Workers Rate:         {total_pred_needs_review} / {total_pred_workers} detected ({needs_review_rate_pct:.1f}%)")
    log("-" * 140)
    log(f"  • Violations Flagged by Pipeline:    {total_pred_violations}")
    log(f"  • Violations Caught (Upper Bound):   {violations_caught_upper_bound} / {total_true_violations} true violations (sum of min(pred, true))")
    log("-" * 140)
    log("  • Needs-Review Workers Breakdown by Reason (Total across all images):")
    for cat in REASON_CATEGORIES:
        count = total_reason_counts[cat]
        pct = (count / total_pred_needs_review * 100.0) if total_pred_needs_review > 0 else 0.0
        log(f"      - {cat:<24}: {count:>3} ({pct:>5.1f}% of needs-review)")
    log("=" * 140)
    log("Note: Comp Rate = compliant / all detected workers, including needs review")
    log("=" * 140)

    # Table of wrong images
    log("\n" + "=" * 140)
    if wrong_images:
        log(f"TABLE OF WRONG IMAGES ({len(wrong_images)} mismatches):")
        log("=" * 140)
        log(f"{'Image':<18} {'True Comp':>10} {'Pred Comp':>10} {'True Viol':>10} {'Pred Viol':>10} {'Pred NR':>9} {'Mismatch Type':>26}")
        log("-" * 140)
        for w in wrong_images:
            diffs = []
            if w["true_compliant"] != w["pred_compliant"]:
                diffs.append("Compliant Mismatch")
            if w["true_violations"] != w["pred_violations"]:
                diffs.append("Violation Mismatch")
            diff_str = ", ".join(diffs)
            log(
                f"{w['image']:<18} "
                f"{w['true_compliant']:>10} "
                f"{w['pred_compliant']:>10} "
                f"{w['true_violations']:>10} "
                f"{w['pred_violations']:>10} "
                f"{w['needs_review']:>9} "
                f"{diff_str:>26}"
            )
    else:
        log("TABLE OF WRONG IMAGES:")
        log("=" * 140)
        log("🎉 Perfect score! Zero mismatches — all predicted counts matched ground truth.")
    log("=" * 140)

    # Save CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    fieldnames = [
        "image",
        "true_compliant",
        "pred_compliant",
        "true_violations",
        "pred_violations",
        "true_workers",
        "pred_workers",
        "pred_needs_review",
        "compliance_rate",
        "small",
        "low_confidence",
        "view_not_upright",
        "conflicting_detections",
        "hardhat_unknown",
        "vest_unknown",
        "other",
    ]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_image_records)

    # Save Text Report
    os.makedirs(os.path.dirname(output_report), exist_ok=True)
    with open(output_report, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines) + "\n")

    log(f"\nSaved scoring CSV to:    {output_csv} ({len(per_image_records)} rows)")
    log(f"Saved text report to:   {output_report}")


def main():
    parser = argparse.ArgumentParser(
        description="Score pipeline predicted counts against ground truth CSV."
    )
    parser.add_argument(
        "folder",
        nargs="?",
        default=None,
        help="Path to folder containing images (e.g. sample_images or eval/test_photos).",
    )
    parser.add_argument(
        "ground_truth",
        nargs="?",
        default=None,
        help=f"Path to ground truth CSV (default: {DEFAULT_GT_PATH}).",
    )
    parser.add_argument(
        "--folder",
        dest="flag_folder",
        default=None,
        help="Optional flag for image folder.",
    )
    parser.add_argument(
        "--ground_truth",
        "--gt",
        dest="flag_gt",
        default=None,
        help=f"Optional flag for ground truth CSV path.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONFIDENCE,
        help=f"Confidence threshold (default: {DEFAULT_CONFIDENCE}).",
    )
    parser.add_argument(
        "--output-csv",
        default=OUTPUT_CSV_PATH,
        help=f"Output CSV path (default: {OUTPUT_CSV_PATH}).",
    )
    parser.add_argument(
        "--output-report",
        default=OUTPUT_REPORT_PATH,
        help=f"Output text report path (default: {OUTPUT_REPORT_PATH}).",
    )
    args = parser.parse_args()

    image_folder = args.folder or args.flag_folder
    ground_truth_path = args.ground_truth or args.flag_gt or DEFAULT_GT_PATH

    if not image_folder:
        print("Error: Image folder argument is required.", file=sys.stderr)
        print("Usage: python eval/score_images.py <image_folder> [<ground_truth_csv>]", file=sys.stderr)
        sys.exit(1)

    score_images(
        image_folder=image_folder,
        ground_truth_path=ground_truth_path,
        conf_threshold=args.conf,
        output_csv=args.output_csv,
        output_report=args.output_report,
    )


if __name__ == "__main__":
    main()
