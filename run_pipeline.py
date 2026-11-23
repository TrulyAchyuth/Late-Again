"""
run_pipeline.py
---------------
Runs every step in order:

    1. download the real data   (src/download_data.py)  - needs internet, about 8 MB
    2. build the clean tables   (src/build_dataset.py)
    3. train + evaluate models  (src/train.py)

Run:
    python run_pipeline.py              # everything
    python run_pipeline.py --offline    # skip steps 1-2 and reuse data/processed/ (already in the repo)
"""

import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"

steps = ["download_data.py", "build_dataset.py", "train.py"]
if "--offline" in sys.argv:
    steps = ["train.py"]

for step in steps:
    print(f"\n===== {step} =====", flush=True)
    subprocess.run([sys.executable, str(SRC / step)], check=True, cwd=SRC)

print("\nDone. Start the dashboard with:  streamlit run app.py")
