"""Reproduce every result from the raw CSV:  python run_all.py"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = ["01_inspect_data.py", "02_make_splits.py", "03_tune.py", "04_train_final.py", "05_evaluate.py",
         "06_error_analysis.py", "07_robustness.py"]

for step in STEPS:
    print(f"\n{'=' * 20} {step} {'=' * 20}")
    subprocess.run([sys.executable, str(ROOT / "scripts" / step)], cwd=ROOT, check=True)
print("\nAll steps finished.")
