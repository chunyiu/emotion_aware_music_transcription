# Emotion Pipelines

This folder contains **12 standalone pipelines** that process singing recordings into sheet music (MusicXML/MIDI/JSON) and run emotion classification on the original vs. reconstructed audio.

## Pipeline Reference

### Pipelines with Harmony (1–4)

These combine pitch detection with harmony generation:

| Pipeline       | Pitch Method       | Harmony Method           | Output          |
| -------------- | ------------------ | ------------------------ | --------------- |
| **Pipeline 1** | pYIN + HMM         | Music21 (Roman numerals) | MusicXML        |
| **Pipeline 2** | pYIN + HMM         | Transformer + mingus     | MusicXML + MIDI |
| **Pipeline 3** | TorchCrepe         | Music21 (Roman numerals) | MusicXML        |
| **Pipeline 4** | CREPE (TensorFlow) | Music21 (Roman numerals) | MusicXML        |

### Standalone Pipelines (A–H)

These focus on pitch detection + emotion classification without harmony:

| Pipeline       | Pitch Method         | Output Format    | Notes                                         |
| -------------- | -------------------- | ---------------- | --------------------------------------------- |
| **Pipeline A** | Simple pYIN          | JSON (roundtrip) | Exports notes as JSON, re-synthesizes to MIDI |
| **Pipeline B** | Simple pYIN          | MusicXML + PDF   | MuseScore renders WAV for emotion comparison  |
| **Pipeline C** | pYIN + HMM + Viterbi | JSON (roundtrip) | Viterbi-smoothed pitch, JSON roundtrip        |
| **Pipeline D** | pYIN + HMM + Viterbi | MusicXML         | Viterbi-smoothed pitch, MusicXML output       |
| **Pipeline E** | CREPE (TensorFlow)   | MusicXML         | Standard CREPE, no harmony                    |
| **Pipeline F** | CREPE (TensorFlow)   | JSON             | Same as E but JSON output                     |
| **Pipeline G** | TorchCrepe (PyTorch) | MusicXML + PDF   | TorchCrepe pitch detection                    |
| **Pipeline H** | TorchCrepe (PyTorch) | JSON (roundtrip) | TorchCrepe with JSON roundtrip                |

## Folder Structure

```
emotion_pipelines/
├── Pipeline 1.py ... Pipeline H.py   # Standalone pipelines (see table above)
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── musicxml_to_wav.py                 # MusicXML → WAV converter (FluidSynth)
├── vocal-to-score-demo/               # Demo with 50 sample audio files
│   ├── Vocal_to_Score_Pipeline.ipynb  # Vocal-to-MIDI notebook
│   ├── Input/GTSinger_sample_50/      # Sample audio files
│   └── outputs/                       # Generated MIDI/MusicXML/WAV
├── src/                               # Shared source code
│   ├── download_ravdess.py            # Download RAVDESS emotion dataset
│   ├── musicxml_to_wav.py             # MusicXML → WAV converter
│   ├── run_emotion_pipeline.py        # Master script: train → predict → analyze
│   └── emotion_classification/        # Emotion model training & prediction
│       ├── train_emotion_classifier.py
│       ├── predict_emotion.py
│       └── analyze_best_files.py
├── data/raw/archive/                  # RAVDESS dataset (download required)
└── results/
    ├── analysis/                      # Top files, listening test candidates
    ├── emotion_model/                 # Trained model (.pkl files)
    └── predictions/                   # Emotion predictions (JSON/CSV)
```

## Prerequisites

1. **Python 3.9 or 3.10**
   - On Windows, during install, tick **"Add Python to PATH"**.
2. **MuseScore 4** (for MusicXML/MIDI → WAV conversion)
   - Default path: `C:\Program Files\MuseScore 4\bin\musescore4.exe`
3. **FluidSynth** (optional, for WAV synthesis)
   - Set environment variables `FLUIDSYNTH_DIR` and `FLUIDSYNTH_BIN` to point to your FluidSynth installation.

## Setup

### 1. Create a virtual environment

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the RAVDESS dataset (for emotion model training)

```bash
python src/download_ravdess.py
```

## Running a Pipeline

Each pipeline is a standalone script. Run any of them directly:

```bash
python "Pipeline 1.py"
python "Pipeline A.py"
```

Input audio files should be placed in `vocal-to-score-demo/Input/GTSinger_sample_50/`.

## Running the Full Emotion Classification Workflow

```bash
python src/run_emotion_pipeline.py
```

This runs three steps:

1. **Train** the emotion classifier on RAVDESS (80/20 split)
2. **Predict** emotions on pipeline outputs
3. **Analyze** results and select top candidates for listening tests

Results are saved under `results/`.
