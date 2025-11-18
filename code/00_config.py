from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
for d in [DATA_RAW, DATA_PROCESSED, MODELS, RESULTS, FIGURES]:
    d.mkdir(parents=True, exist_ok=True)
