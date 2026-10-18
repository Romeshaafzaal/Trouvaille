"""Runs the whole Phase 2 pipeline from the repo root:
    python data_scripts/run_pipeline.py              # scrape + clean + features + EDA
    python data_scripts/run_pipeline.py --skip-scrape  # reuse existing data/raw/products_raw.csv
"""
import subprocess
import sys

STEPS = ["scrape_products.py", "clean_data.py", "preprocess_features.py", "eda.py"]

if __name__ == "__main__":
    steps = STEPS[1:] if "--skip-scrape" in sys.argv else STEPS
    for step in steps:
        print(f"\n=== Running {step} ===")
        result = subprocess.run([sys.executable, f"data_scripts/{step}"])
        if result.returncode != 0:
            sys.exit(f"{step} failed (exit code {result.returncode})")
    print("\nPipeline finished.")
