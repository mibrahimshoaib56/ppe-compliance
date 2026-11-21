"""Score detection and compliance pipeline predictions against ground-truth labels.

Reads ground truth CSV (with columns: image, true_compliant, true_violations)
and runs the pipeline at the default confidence threshold (0.25). Computes and
displays:
1. Image-level accuracy (both compliant and violation counts match exactly)
2. Mean Absolute Error (MAE) for compliant count
3. Mean Absolute Error (MAE) for violation count
4. A structured table of mismatched (wrong) images

Validates that the ground truth CSV contains the required columns and that
every image listed in the CSV exists in the specified folder, raising clear
errors otherwise.
"""

import argparse
import csv
import os
import sys
from typing import Any, Dict, List

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


def score_images(image_folder: str, ground_truth_path: str, conf_threshold: float = DEFAULT_CONFIDENCE):
    """Evaluate pipeline predictions against ground truth labels.

    Args:
        image_folder: Directory containing image files.
        ground_truth_path: Path to ground truth CSV.
        conf_threshold: Inference confidence threshold (default: 0.25).
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
            true_comp = int(r["true_compliant"])
            true_viol = int(r["true_violations"])
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
        })

    # 5. Run pipeline inference at default confidence
    model = det.load_detector_model()

    print("=" * 78)
    print("SCORING PPE COMPLIANCE PIPELINE AGAINST GROUND TRUTH")
    print("=" * 78)
    print(f"Image Folder:        {image_folder}")
    print(f"Ground Truth CSV:    {ground_truth_path}")
    print(f"Images to Evaluate:  {len(validated_data)}")
    print(f"Confidence Setting:  {conf_threshold} (Default)")
    print("-" * 78)

    correct_images_count = 0
    total_abs_err_compliant = 0.0
    total_abs_err_violations = 0.0
    wrong_images: List[Dict[str, Any]] = []

    for item in validated_data:
        fname = item["image"]
        img_path = item["path"]
        t_comp = item["true_compliant"]
        t_viol = item["true_violations"]

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

            p_comp = summary.get("compliant_count", 0)
            p_viol = summary.get("violation_count", 0)

        err_comp = abs(p_comp - t_comp)
        err_viol = abs(p_viol - t_viol)
        total_abs_err_compliant += err_comp
        total_abs_err_violations += err_viol

        is_match = (p_comp == t_comp) and (p_viol == t_viol)
        if is_match:
            correct_images_count += 1
        else:
            wrong_images.append({
                "image": fname,
                "true_compliant": t_comp,
                "pred_compliant": p_comp,
                "true_violations": t_viol,
                "pred_violations": p_viol,
            })

    total_images = len(validated_data)
    accuracy_pct = (correct_images_count / total_images) * 100.0
    mae_compliant = total_abs_err_compliant / total_images
    mae_violations = total_abs_err_violations / total_images

    # 6. Display evaluation metrics
    print(f"\nEVALUATION RESULTS:")
    print(f"  • Total Images:              {total_images}")
    print(f"  • Exact Matches:             {correct_images_count} / {total_images}")
    print(f"  • Image-Level Accuracy:      {accuracy_pct:.1f}% (both compliant & violation counts matched)")
    print(f"  • MAE (Compliant Count):     {mae_compliant:.2f}")
    print(f"  • MAE (Violations Count):    {mae_violations:.2f}")

    # 7. Display table of wrong images
    print("\n" + "=" * 78)
    if wrong_images:
        print(f"TABLE OF WRONG IMAGES ({len(wrong_images)} mismatches):")
        print("=" * 78)
        print(f"{'Image':<26} {'True Comp':>10} {'Pred Comp':>10} {'True Viol':>10} {'Pred Viol':>10} {'Status':>16}")
        print("-" * 78)
        for w in wrong_images:
            diffs = []
            if w["true_compliant"] != w["pred_compliant"]:
                diffs.append("Comp Mismatch")
            if w["true_violations"] != w["pred_violations"]:
                diffs.append("Viol Mismatch")
            reason_str = ", ".join(diffs)
            print(
                f"{w['image']:<26} "
                f"{w['true_compliant']:>10} "
                f"{w['pred_compliant']:>10} "
                f"{w['true_violations']:>10} "
                f"{w['pred_violations']:>10} "
                f"{reason_str:>16}"
            )
    else:
        print("TABLE OF WRONG IMAGES:")
        print("=" * 78)
        print("🎉 Perfect score! Zero mismatches — all predicted counts matched ground truth.")
    print("=" * 78)


def main():
    parser = argparse.ArgumentParser(
        description="Score pipeline predicted counts against ground truth CSV."
    )
    parser.add_argument(
        "folder",
        nargs="?",
        default=None,
        help="Path to folder containing images (e.g. sample_images).",
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
    )


if __name__ == "__main__":
    main()
