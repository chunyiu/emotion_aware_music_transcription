"""Centralized configuration for all pipelines."""

from pathlib import Path

BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "data" / "GTSinger_English"
OUTPUT_DIR = BASE_DIR / "output"
MODEL_DIR = BASE_DIR / "results" / "emotion_model"
MUSESCORE_EXE = r"C:\Program Files\MuseScore 4\bin\musescore4.exe"

# Default BPM for score construction
DEFAULT_BPM = 120
