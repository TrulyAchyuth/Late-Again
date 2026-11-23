import sys
from pathlib import Path

# Make "import features", "import analysis" ... work like they do inside src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
