"""
Pipeline A: Simple pYIN pitch detection + JSON ground truth comparison.
Modified to support device selection with priority: CUDA > MPS > CPU.
Device is auto-detected if not provided (works with current orchestrator).
"""

import sys
import json
import os
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from common.emotion_classifier import EmotionClassifier
from common.pitch_detectors import detect_pitch_simple_pyin
from common.note_segmentation import segment_notes_simple
from common.note_schema import TranscribedNote, save_transcription
from common.ground_truth import load_ground_truth_json, compare_notes, compute_melody_frame_metrics
from common.file_discovery import discover_gtsinger_files, make_unique_id
from config import DATASET_DIR, OUTPUT_DIR, MODEL_DIR

PIPELINE_ID = "A"
PITCH_METHOD = "simple_pyin"
GT_FORMAT = "json"


def get_best_device() -> str:
    """Auto-detect best device with priority: cuda > mps > cpu"""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run_pipeline(input_dir=None, output_dir=None, model_dir=None, max_files=None, device=None):
    input_dir = Path(input_dir or DATASET_DIR)
    output_dir = Path(output_dir or OUTPUT_DIR) / "stage1_transcription" / f"pipeline_{PIPELINE_ID}"
    model_dir = Path(model_dir or MODEL_DIR)

    # === DEVICE SELECTION (CUDA > MPS > CPU) ===
    if device is None:
        device = get_best_device()
    if device == "mps":
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    print(f"[Pipeline {PIPELINE_ID}] Using device: {device.type.upper()}")

    output_dir.mkdir(parents=True, exist_ok=True)
    # Pass device to EmotionClassifier (update the common class to accept this parameter)
    classifier = EmotionClassifier(str(model_dir), device=device)

    files = discover_gtsinger_files(input_dir, max_files=max_files)
    print(f"\n[Pipeline {PIPELINE_ID}] Found {len(files)} audio files")

    if not files:
        print("No files to process.")
        return []

    results = []

    # ── Main processing loop with progress bar ────────────────────────────────
    for file_info in tqdm(
        files,
        desc=f"Pipeline {PIPELINE_ID} – processing files",
        unit="file",
        ncols=100,
        dynamic_ncols=True
    ):
        wav_path = file_info['wav']

        try:
            unique_id = make_unique_id(wav_path, input_dir)

            # Emotion BEFORE
            emotion_before = classifier.predict(str(wav_path))
            if not emotion_before:
                print(f"  {wav_path.name} -> Emotion classification failed, skipping")
                continue

            # Pitch detection (no device needed - uses librosa/pyin)
            times, f0, voiced_flag, voiced_probs, y, sr = detect_pitch_simple_pyin(str(wav_path))

            # Note segmentation
            detected_notes = segment_notes_simple(f0, voiced_flag, times, sr=sr)

            # Ground truth comparison (JSON)
            gt_path = file_info['json']
            gt_metrics = {}
            gt_emotion = None
            if gt_path:
                gt = load_ground_truth_json(gt_path)
                if gt:
                    # Note-level metrics
                    gt_metrics = compare_notes(detected_notes, gt)
                    # Frame-level melody metrics (RPA/RCA/OA)
                    frame_metrics = compute_melody_frame_metrics(times, f0, gt)
                    if frame_metrics:
                        gt_metrics.update(frame_metrics)
                    gt_emotion = gt.get('emotion')

            # Convert to TranscribedNote objects
            notes = [
                TranscribedNote(
                    start=n['start'], end=n['end'],
                    pitch_midi=n['pitch_midi'], pitch_hz=n['pitch_hz'],
                    confidence=n['confidence']
                )
                for n in detected_notes
            ]

            # Save intermediate JSON
            out_path = output_dir / f"{unique_id}_notes.json"
            save_transcription(notes, {
                'pipeline': PIPELINE_ID,
                'pitch_method': PITCH_METHOD,
                'gt_format': GT_FORMAT,
                'source_audio': str(wav_path),
                'unique_id': unique_id,
                'bpm': 120,
                'emotion_before': emotion_before,
                'ground_truth_metrics': gt_metrics,
                'gt_emotion': gt_emotion,
                'device_used': device
            }, out_path)

            results.append({
                'file': wav_path.name,
                'unique_id': unique_id,
                'num_notes': len(notes),
                'emotion_before': emotion_before,
                'gt_metrics': gt_metrics,
                'gt_emotion': gt_emotion,
                'output': str(out_path.relative_to(output_dir)),
            })

        except Exception as e:
            print(f"  Error processing {wav_path.name}: {e}")
            # Optional: uncomment for full trace in development
            # import traceback
            # traceback.print_exc()

    # ── Save summary ──────────────────────────────────────────────────────────
    summary = {
        'pipeline': f'Pipeline {PIPELINE_ID}: {PITCH_METHOD} + {GT_FORMAT} GT',
        'total_files': len(files),
        'successful': len(results),
        'failed': len(files) - len(results),
        'device': device,
        'results': results,
    }
    summary_path = output_dir / f"pipeline_{PIPELINE_ID.lower()}_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nSummary saved to {summary_path}")
    print(f"Successfully processed {len(results)} / {len(files)} files")

    return results


if __name__ == "__main__":
    run_pipeline()