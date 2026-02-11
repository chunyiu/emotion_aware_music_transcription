# Vocal-to-Score Demo

This folder contains a demo of the vocal-to-score pipeline with sample audio files and generated outputs.

## Folder Structure

```
vocal-to-score-demo/
├── Vocal_to_Score_Pipeline.ipynb   # Main vocal-to-MIDI pipeline notebook
├── Input/
│   └── GTSinger_sample_50/         # 50 sample audio files (.wav)
└── outputs/
    ├── melody.mid                   # Generated melody MIDI
    ├── melody.musicxml              # Melody sheet music
    ├── score.mid                    # Harmonized score MIDI
    ├── score.musicxml               # Full score sheet music
    ├── score_render.wav             # Rendered playback (if generated)
    └── evaluation_report.json       # Transcription metrics
```

## What Gets Analyzed

The emotion classifier processes the **50 sample audio files** in the `Input/` folder.

### Input Audio

- `Input/` folder — 50 sample audio files
  - **Purpose**: Predict emotion for each of the 50 candidate files
  - **Goal**: Rank them and select top 3 best-performing ones for manual verification

### Generated Outputs (if available)

- `outputs/score_render.wav` — Synthesized playback of generated score
  - **Purpose**: Measure emotion of final generated music
  - **Note**: Created by evaluation cell in notebook (MIDI → WAV via MuseScore)

### Pipeline Flow

```
50 Sample Audio Files (Input folder)
    ↓ For each file:
    ↓ - Extract audio features (MFCC, mel-spectrogram)
    ↓ - Predict emotion with trained model
    ↓ - Calculate confidence score
Emotion Predictions for all 50 files
    ↓ Ranking:
    ↓ - Sort by confidence (highest first)
    ↓ - Select top 50 overall
    ↓ - Select top 3 per emotion
Top 3 Files for Manual Listening Test
```

## Expected Workflow

1. **Run vocal-to-score pipeline** (`Vocal_to_Score_Pipeline.ipynb`)
   - Generates melody.mid, score.mid, score.musicxml
   - Evaluation cell creates score_render.wav

2. **Run emotion classification**:

   ```cmd
   python src\run_emotion_pipeline.py
   ```

   - Trains on RAVDESS dataset
   - Predicts emotions on all audio in this folder
   - Ranks outputs by confidence

3. **Analyze results**:
   - Compare original input emotion vs generated output emotion
   - Review top 3 candidates
   - Manual listening test to verify predictions

## Key Comparisons

### Before Training (Baseline)

- How well does the model perform on RAVDESS test set (20%)?
- Metrics: accuracy, precision, recall, F1 per emotion

### After Training (Unseen Data)

- What emotion does the original input convey?
- What emotion does the generated output (`score_render.wav`) convey?
- Does the pipeline preserve the emotional content?

## Customization

Edit `src/emotion_classification/predict_emotion.py`:

```python
# Process only outputs
main(input_dir=SAMPLE_FOLDER / "outputs")

# Process only Input folder
main(input_dir=SAMPLE_FOLDER / "Input")

# Process specific files
specific_files = [
    SAMPLE_FOLDER / "Input" / "test_melody.mp3",
    SAMPLE_FOLDER / "outputs" / "score_render.wav"
]
main(input_files=specific_files)
```

## Troubleshooting

### "No audio files found"

- Ensure MIDI files have been rendered to WAV
- Run the evaluation cell in `Vocal_to_Score_Pipeline.ipynb` to generate `score_render.wav`
- Check that MuseScore is installed and configured

### "score_render.wav missing"

- The evaluation notebook cell creates this file via MIDI rendering
- Requires MuseScore CLI or fluidsynth + soundfont
- Run notebook evaluation cell before emotion classification
