"""
run_pipeline.py
---------------
Runs every step in order:
    1. generate synthetic data  2. preprocess  3. EDA charts  4. train models

Run:
    python run_pipeline.py
    python run_pipeline.py --skip-generate   # keep your existing raw CSV
"""

import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"

steps = ["generate_data.py", "preprocess.py", "eda.py", "train.py"]
if "--skip-generate" in sys.argv:
    steps.remove("generate_data.py")

for step in steps:
    print(f"\n===== Running {step} =====")
    subprocess.run([sys.executable, str(SRC / step)], check=True, cwd=SRC)

print("\nAll done! Start the dashboard with:  streamlit run app.py")
