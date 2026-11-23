"""
Runs download -> build -> train.

python run_pipeline.py            full run (needs internet)
python run_pipeline.py --offline  retrain from data/processed/ only
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
