"""
Synthesize audio from each pipeline's transcribed notes, classify emotion,
and write 'emotion_after' back into each *_notes.json file.

Synthesis: direct sine-wave rendering from (start, end, pitch_hz) tuples.
The EmotionClassifier only reads the first 3 seconds, so we only synthesize
notes that fall within the first 4 seconds.

Usage:
    python add_emotion_after.py                  # all pipelines
    python add_emotion_after.py --pipeline A     # single pipeline
    python add_emotion_after.py --overwrite      # re-classify even if already set
"""

import argparse
import glob
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent                               # stage1_transcription/
MODEL_DIR = BASE_DIR.parent / "emotion_pipelines" / "results" / "emotion_model"

sys.path.insert(0, str(BASE_DIR.parent / "emotion_pipelines"))
from common.emotion_classifier import EmotionClassifier

SR             = 22050
SYNTH_DURATION = 4.0   # seconds to synthesize (classifier uses first 3 s)
AMPLITUDE      = 0.6   # peak amplitude of synthesized signal


# ── Synthesis ────────────────────────────────────────────────────────────────

def synthesize_notes(notes: list, sr: int = SR, max_dur: float = SYNTH_DURATION) -> np.ndarray:
    """Render note dicts (start, end, pitch_hz) as overlapping sine waves, float32."""
    if not notes:
        return np.zeros(int(sr * max_dur), dtype=np.float32)

    relevant = [n for n in notes if n.get('start', 0) < max_dur and n.get('pitch_hz', 0) > 0]
    if not relevant:
        relevant = [n for n in notes if n.get('pitch_hz', 0) > 0][:5]
    if not relevant:
        return np.zeros(int(sr * max_dur), dtype=np.float32)

    total_samples = int(max_dur * sr)
    audio = np.zeros(total_samples, dtype=np.float32)

    for note in relevant:
        start_s  = int(note['start'] * sr)
        end_s    = min(int(note['end'] * sr), total_samples)
        duration = end_s - start_s
        if duration <= 0:
            continue

        t    = np.arange(duration) / sr
        sine = np.sin(2 * np.pi * note['pitch_hz'] * t).astype(np.float32)

        # Attack/release to avoid clicks at note boundaries
        attack  = min(int(0.010 * sr), duration // 4)
        release = min(int(0.050 * sr), duration // 4)
        env = np.ones(duration, dtype=np.float32)
        if attack  > 0: env[:attack]   = np.linspace(0, 1, attack, dtype=np.float32)
        if release > 0: env[-release:] = np.linspace(1, 0, release, dtype=np.float32)

        audio[start_s:end_s] += sine * env

    peak = np.max(np.abs(audio))
    if peak > 0:
        audio *= AMPLITUDE / peak

    return audio


# ── Main ─────────────────────────────────────────────────────────────────────

def process_pipeline(pipeline_dir: Path, classifier: EmotionClassifier,
                     tmp_wav: str, overwrite: bool):
    all_files = sorted(glob.glob(str(pipeline_dir / "*_notes.json")))
    if not all_files:
        print(f"  No *_notes.json files found in {pipeline_dir.name}")
        return

    # Single pass: load JSON once, decide whether to process, cache data for reuse
    to_process = []
    already = 0
    for fp in all_files:
        with open(fp, encoding='utf-8') as f:
            data = json.load(f)
        if not overwrite and data.get('emotion_after'):
            already += 1
        else:
            to_process.append((fp, data))

    print(f"\n{pipeline_dir.name}: {len(to_process)} to process"
          + (f", {already} already done (skipping)" if already else ""))

    if not to_process:
        return

    errors = 0
    for fp, data in tqdm(to_process, desc=pipeline_dir.name, unit="file", ncols=100):
        try:
            audio = synthesize_notes(data.get('notes', []))
            sf.write(tmp_wav, audio, SR, subtype='PCM_16')

            emotion_after = classifier.predict(tmp_wav)
            if emotion_after:
                data['emotion_after'] = emotion_after
                with open(fp, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, default=str)

        except Exception as e:
            errors += 1
            tqdm.write(f"  Error on {Path(fp).name}: {e}")

    if errors:
        print(f"  {errors} error(s) in {pipeline_dir.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pipeline', type=str, default=None,
                        help="Process only this pipeline letter (e.g. A)")
    parser.add_argument('--overwrite', action='store_true',
                        help="Re-classify even if emotion_after already exists")
    args = parser.parse_args()

    print("Loading emotion classifier...")
    classifier = EmotionClassifier(str(MODEL_DIR))
    print("Classifier loaded.\n")

    if args.pipeline:
        pipelines = [BASE_DIR / f"pipeline_{args.pipeline.upper()}"]
    else:
        pipelines = sorted(
            [d for d in BASE_DIR.iterdir()
             if d.is_dir() and d.name.startswith("pipeline_")],
            key=lambda d: d.name
        )

    # Single shared temp WAV file reused across all files
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp_wav = tmp.name

    try:
        for pipeline_dir in pipelines:
            process_pipeline(pipeline_dir, classifier, tmp_wav, args.overwrite)
    finally:
        if os.path.exists(tmp_wav):
            os.unlink(tmp_wav)

    print("\nDone. Re-run compute_emotion_preservation.py to see updated rates.")


if __name__ == "__main__":
    main()
