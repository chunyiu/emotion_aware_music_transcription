# Automated Transcription of Vocal Performances to Sheet Music Through Emotion-Aware Deep Learning and LLMs

This project provides a comprehensive pipeline to transcribe vocal performances into sheet music (MusicXML/MIDI) while analyzing and preserving the emotional content of the original performance. It integrates pitch detection algorithms (pYIN, CREPE, HMM) with deep learning-based emotion classification.

## Project Structure

| Folder                              | Description                                                                                                                                                                                 |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `emotion_pipelines/`                | 12 standalone pipelines for vocal-to-score transcription with emotion classification. See [emotion_pipelines/README.md](emotion_pipelines/README.md) for the full pipeline reference table. |
| `vocal-transcription-with-harmony/` | Notebooks and scripts for vocal transcription using Librosa, pYIN, CREPE, and TorchCrepe.                                                                                                   |
| `eda/`                              | Exploratory Data Analysis on the GTSinger (vocal) and RAVDESS (emotion) datasets.                                                                                                           |

## Prerequisites

- **Python 3.9 or 3.10**
- **MuseScore 4**: Required for rendering MusicXML/MIDI to WAV for audio evaluation.
  - Default path: `C:\Program Files\MuseScore 4\bin\musescore4.exe`
- **FluidSynth** (optional): For WAV synthesis. Set environment variables `FLUIDSYNTH_DIR` and `FLUIDSYNTH_BIN`.

## Installation

1. Create and activate venv

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
```

2. Install dependencies

```bash
pip install -r requirements.txt
```
