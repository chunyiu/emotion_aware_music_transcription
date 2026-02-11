# Vocal Transcription with Harmony

Notebooks and scripts for vocal pitch transcription using various algorithms, with optional harmony generation.

## Notebooks

Located in `notebooks/`:

| File                               | Pitch Method            | Output         | Description                                         |
| ---------------------------------- | ----------------------- | -------------- | --------------------------------------------------- |
| `crepe_json_transcription.py`      | CREPE (TensorFlow, CPU) | JSON           | CREPE pitch detection → JSON note output            |
| `crepe_xml_transcription.py`       | CREPE (TensorFlow, CPU) | MusicXML       | CREPE pitch detection → MusicXML output             |
| `torchcrepe_json_transcription.py` | TorchCrepe (PyTorch)    | JSON           | TorchCrepe pitch detection → JSON output            |
| `torchcrepe_xml_transcription.py`  | TorchCrepe (PyTorch)    | MusicXML       | TorchCrepe pitch detection → MusicXML + harmony     |
| `pyin_hmm_transcription.py`        | Librosa pYIN + HMM      | MIDI           | pYIN pitch detection with HMM smoothing             |
| `pyin_evaluation.py`               | Librosa pYIN            | MIDI + metrics | Evaluation pipeline for pYIN transcription accuracy |

### Notebook Results

- `crepe_json_results/` — Output from CREPE JSON transcription
- `crepe_xml_results/` — Output from CREPE XML transcription
- `torchcrepe_json_results/` — Output from TorchCrepe JSON transcription
- `torchcrepe_xml_results/` — Output from TorchCrepe XML transcription
- `pyin_hmm_results/` — Output from pYIN + HMM transcription
- `pyin_evaluation_results/` — Evaluation metrics and results
- `experiment_log.csv` — Log of all experiments run

## Scripts

Located in `scripts/`:

| File                          | Pitch Method         | Harmony                  | Output          |
| ----------------------------- | -------------------- | ------------------------ | --------------- |
| `pyin_hmm_music21_harmony.py` | pYIN + HMM           | Music21 (Roman numerals) | MusicXML + MIDI |
| `pyin_hmm_mingus_harmony.py`  | pYIN + HMM           | Transformer + mingus     | MusicXML + MIDI |
| `pyin_hmm_viterbi.py`         | pYIN + HMM + Viterbi | None                     | MIDI            |

### Script Results

- `pyin_hmm_music21_output/` — Output from Music21 harmony script
- `pyin_hmm_mingus_output/` — Output from mingus harmony script
- `pyin_hmm_viterbi_output/` — Output from Viterbi script

## Setup

```bash
pip install -r requirements.txt
```

## Data

Place the GTSinger sample audio files in:

- `notebooks/GTSinger_sample_50/`
- `scripts/GTSinger_sample_50/`
