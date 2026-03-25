# Tracking Emotional Drift through Multi-Stage Music Transcription and Harmonization

This project measures how much the emotional character of a singing recording is preserved (or lost) as it passes through automated transcription and harmonization. Audio is transcribed into sheet music using several pitch detection algorithms, harmony is then generated on top, and an emotion classifier evaluates the audio at each stage — before transcription, after transcription, and after harmonization.

The system is organized into **two stages** that combine into **32 pipeline variants** (8 Stage 1 × 4 Stage 2), evaluated on the full [GTSinger](https://huggingface.co/datasets/GTSinger/GTSinger) English dataset (~2,982 audio files).

---

## How It Works

### Stage 1 — Pitch Detection & Transcription (Pipelines A–H)

Each pipeline takes raw vocal audio, detects pitch frame-by-frame, segments it into discrete notes, and exports a structured JSON file with the detected notes and metrics. The transcribed notes are compared against ground truth data from GTSinger.

**What varies between pipelines:**

| Pipeline | Pitch Detection Method | Ground Truth Format |
|----------|------------------------|---------------------|
| A | Simple pYIN (librosa) | JSON |
| B | Simple pYIN (librosa) | MusicXML |
| C | pYIN + HMM/Viterbi smoothing | JSON |
| D | pYIN + HMM/Viterbi smoothing | MusicXML |
| E | CREPE (TensorFlow) | JSON |
| F | CREPE (TensorFlow) | MusicXML |
| G | TorchCrepe (PyTorch) | JSON |
| H | TorchCrepe (PyTorch) | MusicXML |

Each pipeline also runs the emotion classifier on the original source audio (`emotion_before`) and stores it alongside the transcribed notes. A separate post-processing step (`add_emotion_after.py`) synthesizes audio from the transcribed notes and classifies `emotion_after`, allowing a direct comparison of how transcription affects perceived emotion.

**Stage 1 output per file** (`{unique_id}_notes.json`):
```json
{
  "pipeline": "A",
  "pitch_method": "simple_pyin",
  "gt_format": "json",
  "source_audio": "/path/to/audio.wav",
  "unique_id": "English_EN_Alto_1_Breathy_song_Group_0000",
  "notes": [
    { "start": 0.0, "end": 0.5, "pitch_midi": 60, "pitch_hz": 261.6, "confidence": 1.0 }
  ],
  "emotion_before": { "top_emotion": "happy", "top_confidence": 0.87, ... },
  "emotion_after":  { "top_emotion": "sad",   "top_confidence": 0.72, ... },
  "gt_emotion": "happy",
  "ground_truth_metrics": {
    "f1": 0.25, "precision": 0.30, "recall": 0.22,
    "rpa": 0.95, "rca": 0.95, "oa": 0.78,
    "ref_count": 50, "pred_count": 42
  }
}
```

**Stage 1 metrics explained:**
- **F1 / Precision / Recall** — note-level overlap between detected and ground-truth notes (onset, offset, pitch must all match within tolerance)
- **RPA** (Raw Pitch Accuracy) — percentage of voiced frames where estimated pitch is within 50 cents of ground truth
- **RCA** (Raw Chroma Accuracy) — same as RPA but pitch class only (octave errors are forgiven)
- **OA** (Overall Accuracy) — combines pitch accuracy with voicing (silence detection) accuracy
- **EPR** (Emotion Preservation Rate) — percentage of files where the classifier's top emotion matches the GTSinger ground truth label

---

### Stage 2 — Harmony Generation & Emotion Analysis (Pipelines 1–4)

Each Stage 2 pipeline reads Stage 1 note data, generates a harmony part, combines melody + harmony into a MusicXML score, converts it to audio (via MuseScore 4 or a fallback sine-wave renderer), and re-classifies emotion.

| Pipeline | Harmony Method |
|----------|----------------|
| 1 | Music21 transposition (interval down a major third) |
| 2 | Mingus algorithmic chord analysis per measure |
| 3 | Diatonic Roman numeral chords (Krumhansl-Schmuckler key detection) |
| 4 | Circle-of-fifths progressions with secondary dominants |

Running Stage 2 against all 8 Stage 1 outputs gives **32 combinations** (A1, A2 … H4). The emotion classifier runs on the final harmonized audio, and the result is compared against `emotion_before` to measure how much harmony generation shifts the perceived emotion.

---

### Emotion Classifier

A Random Forest classifier trained on the [RAVDESS](https://zenodo.org/record/1188976) dataset (24 actors, 8 emotions). It extracts 339 features from the first 3 seconds of audio:
- 40 MFCCs (mean + std)
- 128 mel-spectrogram bands (mean + std)
- Spectral centroid, rolloff, and zero-crossing rate (mean + std)

For Stage 1 EPR, the classifier output is compared against GTSinger's ground-truth emotion labels (happy / sad). For Stage 2, it measures whether the emotion predicted before harmonization matches the emotion predicted after.

---

## Project Structure

```
.
├── README.md
├── LICENSE
├── .gitignore
├── emotion_pipelines/                  # Main codebase
│   ├── config.py                       # Dataset, output, and model paths
│   ├── run_all.py                      # Orchestrator: runs any/all of the 32 combinations
│   ├── generate_report.py              # Produces full_matrix_32.csv/.json summary
│   ├── requirements.txt
│   ├── gtsinger_english_emotions.csv   # GTSinger emotion labels (needed by B/D/F/H pipelines)
│   ├── common/                         # Shared modules used by all pipelines
│   │   ├── emotion_classifier.py       # RAVDESS-trained Random Forest classifier
│   │   ├── pitch_detectors.py          # pYIN, pYIN+HMM, CREPE, TorchCrepe
│   │   ├── note_segmentation.py        # Converts frame-level F0 to note events
│   │   ├── note_schema.py              # TranscribedNote dataclass + JSON I/O
│   │   ├── ground_truth.py             # Load JSON/MusicXML GT; compute mir_eval metrics
│   │   ├── csv_emotions.py             # Load and match GTSinger CSV emotion labels
│   │   ├── harmony_methods.py          # Four harmony generation algorithms
│   │   ├── score_utils.py              # Build music21 scores from note data
│   │   ├── musicxml_to_wav.py          # MusicXML/MIDI -> WAV (MuseScore + sine fallback)
│   │   └── file_discovery.py           # Discover GTSinger wav/json/musicxml file pairs
│   ├── stage1/                         # Stage 1 pipeline scripts
│   │   ├── pipeline_a.py               # Simple pYIN + JSON GT
│   │   ├── pipeline_b.py               # Simple pYIN + MusicXML GT
│   │   ├── pipeline_c.py               # pYIN+HMM + JSON GT
│   │   ├── pipeline_d.py               # pYIN+HMM + MusicXML GT
│   │   ├── pipeline_e.py               # CREPE + JSON GT
│   │   ├── pipeline_f.py               # CREPE + MusicXML GT
│   │   ├── pipeline_g.py               # TorchCrepe + JSON GT
│   │   └── pipeline_h.py               # TorchCrepe + MusicXML GT
│   ├── stage2/                         # Stage 2 pipeline scripts
│   │   ├── pipeline_1.py               # Music21 transposition harmony
│   │   ├── pipeline_2.py               # Mingus chord harmony
│   │   ├── pipeline_3.py               # Diatonic Roman numeral harmony
│   │   └── pipeline_4.py               # Circle-of-fifths harmony
│   ├── scripts/
│   │   └── download_gtsinger.py        # Download the GTSinger English dataset
│   ├── src/                            # Emotion model training
│   │   ├── download_ravdess.py
│   │   ├── run_emotion_pipeline.py     # Train -> predict -> evaluate
│   │   └── emotion_classification/
│   ├── results/
│   │   ├── emotion_model/              # Trained model files (.pkl) — not in repo
│   │   ├── analysis/
│   │   └── predictions/
│   ├── vocal-to-score-demo/            # Standalone demo notebook
│   │   ├── Vocal_to_Score_Pipeline.ipynb
│   │   └── Input/
│   ├── data/
│   │   └── GTSinger_English/           # Dataset (downloaded separately, not in repo)
│   └── output/                         # Generated pipeline outputs (not in repo)
│       ├── stage1_transcription/       # *_notes.json per pipeline (2,982 files each)
│       ├── stage2_harmony/             # MusicXML, WAV, emotion results per combo
│       └── summaries/                  # full_matrix_32.csv/.json
├── stage1_transcription/               # Post-processing scripts for Stage 1 analysis
│   ├── add_emotion_after.py            # Synthesize audio from notes, classify emotion_after
│   ├── compute_transcription_metrics.py   # Average F1/RPA/RCA/OA across all files per pipeline
│   ├── compute_emotion_preservation.py    # Compare emotion_after vs gt_emotion per pipeline
│   ├── patch_musicxml_gt_emotion.py    # Backfill gt_emotion for MusicXML pipelines (B/D/F/H)
│   └── patch_pipeline_b_frame_metrics.py # Patch missing frame metrics for pipeline B
├── eda/                                # Exploratory data analysis
│   ├── eda.py
│   ├── eda_plots/
│   └── requirements.txt
└── vocal-transcription-with-harmony/   # Earlier experimental work
    ├── notebooks/
    └── scripts/
```

> **Note:** The `stage1_transcription/pipeline_*/` result JSON files (2,982 × 8 = ~23,856 files) and `emotion_pipelines/output/` are not committed to the repository. Run the pipelines locally to generate them (see Usage below).

---

## Prerequisites

- **Python 3.9+** (recommended: conda env with numpy < 2.3 to avoid numba compatibility issues)
- **MuseScore 4** — used for MusicXML-to-WAV conversion in Stage 2
  - Default path: `C:\Program Files\MuseScore 4\bin\musescore4.exe` (configurable in `config.py`)

---

## Setup

### 1. Create environment

```bash
conda create -n music-ai python=3.10
conda activate music-ai
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

This places the dataset at `emotion_pipelines/data/GTSinger_English/`.

### 4. Train the emotion classifier

```bash
python src/download_ravdess.py          # Download RAVDESS dataset
python src/run_emotion_pipeline.py      # Train and save model to results/emotion_model/
```

---

## Usage

All commands run from the `emotion_pipelines/` directory unless stated otherwise.

### Run all 32 pipeline combinations

```bash
python run_all.py
```

### Run only Stage 1 or Stage 2

```bash
python run_all.py --stage 1       # Transcription only (pipelines A-H)
python run_all.py --stage 2       # Harmony only (requires Stage 1 output)
```

### Run a subset of pipelines

```bash
python run_all.py --stage1-pipelines A C E G         # JSON-GT pipelines only
python run_all.py --stage2-pipelines 1 3             # Specific harmony methods
python run_all.py --stage1-pipelines A --stage2-pipelines 1 2  # A1 and A2 only
```

### Test with a small number of files first

```bash
python run_all.py --max-files 10
```

### Force a specific compute device

```bash
python run_all.py --device cuda    # NVIDIA GPU
python run_all.py --device cpu     # CPU only
```

Device is auto-selected (CUDA > MPS > CPU) if not specified.

### Generate the 32-combination summary report

```bash
python generate_report.py
```

Outputs `output/summaries/full_matrix_32.csv` and `full_matrix_32.json` — one row per pipeline combination with Stage 1 transcription accuracy and Stage 2 emotion preservation rate.

---

## Stage 1 Post-Processing

After running Stage 1, use the scripts in `stage1_transcription/` to compute metrics and analyze emotion preservation. These scripts operate on the `*_notes.json` files in `stage1_transcription/pipeline_*/`.

### Add `emotion_after` to all Stage 1 files

```bash
cd stage1_transcription
python add_emotion_after.py                  # All pipelines
python add_emotion_after.py --pipeline A     # Single pipeline
python add_emotion_after.py --overwrite      # Re-classify even if already set
```

Synthesizes short sine-wave audio from the transcribed notes for each file, runs the emotion classifier, and writes `emotion_after` back into the JSON.

### Compute transcription metrics (F1, RPA, RCA, OA)

```bash
python compute_transcription_metrics.py
```

Reads `ground_truth_metrics` from each `*_notes.json` and saves per-pipeline averages to `transcription_metrics_summary.json`.

### Compute emotion preservation rates

```bash
python compute_emotion_preservation.py
```

Compares `emotion_after` against GTSinger ground truth (`gt_emotion`) for each file and saves per-pipeline rates to `emotion_preservation_summary.json`. Reports both:
- **Before rate** — original source audio emotion vs. ground truth (baseline, same for all pipelines)
- **After rate** — synthesized transcription emotion vs. ground truth (the meaningful per-pipeline metric)

---

## Exploratory Data Analysis

```bash
cd eda
pip install -r requirements.txt
python eda.py
```

Generates summary statistics and plots for the GTSinger and RAVDESS datasets in `eda_plots/`.

---

## Demo Notebook

A self-contained walkthrough is available at `emotion_pipelines/vocal-to-score-demo/Vocal_to_Score_Pipeline.ipynb`. It runs the full pipeline on sample audio files in `Input/` without needing the full GTSinger dataset.
