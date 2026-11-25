"""Build the final submission ZIP package for the PPE Compliance Project.

Creates a clean, portable ZIP archive containing all project code, models,
reports, slides, evaluation scripts, failure cases, and test photos, while
excluding virtual environments, git history, and temporary caches.
"""

import os
import sys
import zipfile

ZIP_NAME = "ppe_compliance_submission.zip"

INCLUDED_FILES = [
    "app.py",
    "README.md",
    "REPORT.md",
    "REPORT.pdf",
    "SLIDES.md",
    "SLIDES.pptx",
    "requirements.txt",
    "pytest.ini",
    "check_model.py",
    "PROJECT_CONTEXT.md",
]

INCLUDED_DIRS = [
    "src",
    "models",
    "notebooks",
    "tests",
    "scripts",
    "sample_images",
    os.path.join("eval", "results"),
    os.path.join("eval", "test_photos"),
    os.path.join("eval", "test_samples"),
]

EXCLUDED_DIRS = {
    ".venv",
    ".git",
    "__pycache__",
    ".pytest_cache",
    "roboflow_test",
}

EXCLUDED_EXTS = {
    ".pyc",
    ".tmp",
    ".log",
    ".swp",
}


def build_zip(output_path: str = ZIP_NAME):
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_file = os.path.join(repo_root, output_path)

    # Remove existing zip if present
    if os.path.exists(out_file):
        os.remove(out_file)

    print(f"Packaging submission ZIP: {out_file}")
    file_count = 0

    with zipfile.ZipFile(out_file, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # 1. Add root files
        for fname in INCLUDED_FILES:
            full_path = os.path.join(repo_root, fname)
            if os.path.isfile(full_path):
                zf.write(full_path, arcname=fname)
                file_count += 1
                print(f"  + Added: {fname}")

        # 2. Add eval root scripts & ground truth
        eval_root = os.path.join(repo_root, "eval")
        for f in ["ground_truth.csv", "ground_truth_example.csv", "slider_check.py", "score_images.py"]:
            full_f = os.path.join(eval_root, f)
            if os.path.isfile(full_f):
                arc = os.path.join("eval", f)
                zf.write(full_f, arcname=arc)
                file_count += 1
                print(f"  + Added: {arc}")

        # 3. Add included directories
        for d in INCLUDED_DIRS:
            full_dir = os.path.join(repo_root, d)
            if not os.path.isdir(full_dir):
                continue

            for root, dirs, files in os.walk(full_dir):
                # Filter out excluded directories in-place
                dirs[:] = [sub for sub in dirs if sub not in EXCLUDED_DIRS]

                for file in files:
                    ext = os.path.splitext(file)[1].lower()
                    if ext in EXCLUDED_EXTS:
                        continue

                    file_abs = os.path.join(root, file)
                    rel_path = os.path.relpath(file_abs, repo_root)
                    zf.write(file_abs, arcname=rel_path)
                    file_count += 1

    size_mb = os.path.getsize(out_file) / (1024 * 1024)
    print("\n" + "=" * 60)
    print(f"SUCCESS: Created submission package!")
    print(f"Archive: {out_file}")
    print(f"Files:   {file_count} items")
    print(f"Size:    {size_mb:.2f} MB")
    print("=" * 60)


if __name__ == "__main__":
    build_zip()
