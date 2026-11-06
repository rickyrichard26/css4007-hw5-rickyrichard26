import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

def load_helpdesk():
    with open(DATA_DIR / "helpdesk.json", encoding="utf-8") as f:
        return json.load(f)