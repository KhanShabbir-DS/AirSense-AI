"""
Train all AirSense AI models and generate evaluation outputs.

Usage (from project root):
    python scripts/train_all.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

TRAINING_SCRIPTS = [
    "train_regression.py",
    "train_classification.py",
    "regression_analysis.py",
]


def main() -> int:
    for script in TRAINING_SCRIPTS:
        script_path = SRC / script
        if not script_path.exists():
            print(f"Missing script: {script_path}")
            return 1

        print("\n" + "=" * 70)
        print(f"Running {script}")
        print("=" * 70)

        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=SRC,
            check=False,
        )
        if result.returncode != 0:
            print(f"Failed: {script}")
            return result.returncode

    print("\nAll training scripts completed successfully.")
    print("Models saved to: models/")
    print("Reports saved to: outputs/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
