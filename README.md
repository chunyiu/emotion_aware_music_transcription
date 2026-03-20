# Tracking Emotional Drift through Multi-Stage Pipeline Analysis of Transcription and Harmonization

This project transcribes singing recordings into sheet music (MusicXML) and evaluates whether the emotional character of the audio is preserved through the transcription and harmonization process. It integrates pitch detection algorithms (pYIN, CREPE, HMM) with deep learning-based emotion classification trained on the RAVDESS dataset.

The system is organized into two stages that combine to form **32 pipeline combinations** (8 Stage 1 x 4 Stage 2), evaluated on the full [GTSinger](https://huggingface.co/datasets/GTSinger/GTSinger) English dataset.

## Architecture

### Stage 1: Pitch Detection & Transcription (Pipelines A-H)

Each pipeline detects pitch from audio, segments notes, and compares against ground truth from the GTSinger dataset.

| Pipeline | Pitch Method         | Ground Truth Format |
| -------- | -------------------- | ------------------- |
| **A**    | Simple pYIN          | JSON                |
| **B**    | Simple pYIN          | MusicXML            |
| **C**    | pYIN + HMM/Viterbi   | JSON                |
| **D**    | pYIN + HMM/Viterbi   | MusicXML            |
| **E**    | CREPE (TensorFlow)   | JSON                |
| **F**    | CREPE (TensorFlow)   | MusicXML            |
| **G**    | TorchCrepe (PyTorch) | JSON                |
| **H**    | TorchCrepe (PyTorch) | MusicXML            |

Output: `emotion_pipelines/output/stage1_transcription/pipeline_X/{unique_id}_notes.json`

### Stage 2: Harmony Generation (Pipelines 1-4)

Each pipeline reads Stage 1 intermediate JSON, generates harmony, exports MusicXML, converts to WAV, and classifies emotion before vs. after.

| Pipeline | Harmony Method                                       |
| -------- | ---------------------------------------------------- |
| **1**    | Music21 transpose (-M3 interval)                     |
| **2**    | Mingus algorithmic chord analysis                    |
| **3**    | Diatonic Roman numeral chords (Krumhansl-Schmuckler) |
| **4**    | Circle-of-fifths progressions + secondary dominants  |

Output: `emotion_pipelines/output/stage2_harmony/{combo_id}/` (e.g., `A1/`, `B3/`, `H4/`)

### 32 Combinations

Each Stage 2 pipeline runs against each Stage 1 output:

```
A1  A2  A3  A4
B1  B2  B3  B4
C1  C2  C3  C4
D1  D2  D3  D4
E1  E2  E3  E4
F1  F2  F3  F4
G1  G2  G3  G4
H1  H2  H3  H4
```

## Project Structure

```
.
├── README.md
├── LICENSE
├── .gitignore
├── emotion_pipelines/              # Main pipeline codebase
│   ├── config.py                   # Centralized paths and settings
│   ├── run_all.py                  # Orchestrator for all 32 combinations
│   ├── generate_report.py          # Comparison CSV/JSON report generator
│   ├── requirements.txt            # Python dependencies
│   ├── common/                     # Shared modules
│   │   ├── emotion_classifier.py   # RAVDESS-trained emotion classifier (339 features)
│   │   ├── pitch_detectors.py      # pYIN, CREPE, TorchCrepe detectors
│   │   ├── note_segmentation.py    # Note segmentation strategies
│   │   ├── note_schema.py          # TranscribedNote dataclass + JSON I/O
│   │   ├── ground_truth.py         # GT loading (JSON/MusicXML) + mir_eval comparison
│   │   ├── harmony_methods.py      # Four harmony generation methods
│   │   ├── score_utils.py          # Music21 score construction utilities
│   │   ├── musicxml_to_wav.py      # MusicXML/MIDI → WAV conversion (MuseScore + fallback)
│   │   └── file_discovery.py       # GTSinger dataset file discovery
│   ├── stage1/                     # Stage 1 pipelines (A-H)
│   │   └── pipeline_a.py ... pipeline_h.py
│   ├── stage2/                     # Stage 2 pipelines (1-4)
│   │   └── pipeline_1.py ... pipeline_4.py
│   ├── scripts/
│   │   └── download_gtsinger.py    # Download full GTSinger English dataset
│   ├── src/                        # Emotion model training pipeline
│   │   ├── download_ravdess.py     # Download RAVDESS emotion dataset
│   │   ├── run_emotion_pipeline.py # Train → predict → analyze orchestrator
│   │   └── emotion_classification/ # Training, prediction, and analysis scripts
│   ├── results/                    # Trained models and analysis outputs
│   │   ├── emotion_model/          # Trained classifier (.pkl files)
│   │   ├── analysis/               # Top files, listening test candidates
│   │   └── predictions/            # Emotion predictions (CSV/JSON)
│   ├── vocal-to-score-demo/        # Demo notebook with sample audio
│   │   ├── Vocal_to_Score_Pipeline.ipynb
│   │   └── Input/                  # Sample audio files
│   ├── data/
│   │   └── GTSinger_English/       # Full dataset (downloaded, not in repo)
│   └── output/                     # Pipeline outputs (generated, not in repo)
│       ├── stage1_transcription/
│       ├── stage2_harmony/
│       └── summaries/
├── eda/                            # Exploratory Data Analysis
│   ├── eda.py                      # EDA script for GTSinger + RAVDESS datasets
│   ├── eda_plots/                  # Generated plots
│   └── requirements.txt
└── vocal-transcription-with-harmony/   # Prior experimental transcription results
    ├── notebooks/                      # Transcription scripts + results (~54k files)
    └── scripts/                        # Harmony generation scripts + results
```

## Prerequisites

- **Python 3.9+**
- **MuseScore 4** (for MusicXML → WAV conversion)
  - Default path: `C:\Program Files\MuseScore 4\bin\musescore4.exe`

## Setup

### 1. Create a virtual environment

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
```

### 2. Install dependencies

```bash
cd emotion_pipelines
pip install -r requirements.txt
```

### 3. Download the GTSinger English dataset

```bash
python scripts/download_gtsinger.py
```

### 4. Train the emotion classifier

Download the RAVDESS dataset and train the model:

```bash
python src/download_ravdess.py
python src/run_emotion_pipeline.py
```

This trains on RAVDESS (80/20 split) and saves the model to `results/emotion_model/`.

## Usage

All commands run from the `emotion_pipelines/` directory.

### Run all 32 combinations

```bash
python run_all.py
```

### Run with a small subset first

```bash
python run_all.py --max-files 5
```

### Run only Stage 1 or Stage 2

```bash
python run_all.py --stage 1
python run_all.py --stage 2
```

### Run specific pipelines

```bash
python run_all.py --stage1-pipelines A B --stage2-pipelines 1 3
```

### Run individual pipelines

```bash
# Stage 1
python -m stage1.pipeline_a --max-files 10

# Stage 2 (specify which Stage 1 output to consume)
python -m stage2.pipeline_1 --stage1 A
```

### Generate comparison report

```bash
python generate_report.py
```

Outputs `output/summaries/full_matrix_32.csv` and `full_matrix_32.json` with pitch accuracy (F1) and emotion preservation rates across all 32 combinations.

## Exploratory Data Analysis

```bash
cd eda
pip install -r requirements.txt
python eda.py
```

Generates summary statistics and plots for the GTSinger and RAVDESS datasets in `eda_plots/`.

## Vocal-to-Score Demo

A standalone demo notebook is available at `emotion_pipelines/vocal-to-score-demo/Vocal_to_Score_Pipeline.ipynb` for quick testing with sample audio files.
