import sys
import json
from pathlib import Path
import traceback

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from common.emotion_classifier import EmotionClassifier
from common.pitch_detectors import detect_pitch_pyin_hmm
from common.note_segmentation import segment_notes_hmm
from common.note_schema import TranscribedNote, save_transcription
from common.ground_truth import load_ground_truth_json, compare_notes, compute_melody_frame_metrics
from common.file_discovery import discover_gtsinger_files, make_unique_id
from config import DATASET_DIR, OUTPUT_DIR, MODEL_DIR


PIPELINE_ID = "C"
PITCH_METHOD = "pyin_hmm_viterbi"
GT_FORMAT = "json"


def get_device() -> torch.device:
    """Determine best available device with priority: CUDA > MPS > CPU"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using device: CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: MPS (Apple Silicon)")
        # Enable fallback for unsupported operations
        import os
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    else:
        device = torch.device("cpu")
        print("Using device: CPU (no GPU acceleration available)")
    return device


def run_pipeline(
    input_dir=None,
    output_dir=None,
    model_dir=None,
    max_files=None,
    device=None,           # ← added: can be passed from orchestrator later
):
    # Determine device (allow override via argument)
    if device is None:
        device = get_device()
    else:
        device = torch.device(device)
        print(f"Using explicitly requested device: {device}")

    input_dir = Path(input_dir or DATASET_DIR)
    output_dir = Path(output_dir or OUTPUT_DIR) / "stage1_transcription" / f"pipeline_{PIPELINE_ID}"
    model_dir = Path(model_dir or MODEL_DIR)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Pass device to models that support it
    classifier = EmotionClassifier(model_dir=str(model_dir), device=device)

    files = discover_gtsinger_files(input_dir, max_files=max_files)
    print(f"\n[Pipeline {PIPELINE_ID}] Found {len(files)} audio files")

    if not files:
        print("No files to process.")
        return []

    results = []

    # ────────────────────────────────────────────────
    #           Main processing loop with tqdm
    # ────────────────────────────────────────────────
    for i, file_info in enumerate(tqdm(files, desc=f"Pipeline {PIPELINE_ID}", unit="file"), 1):
        wav_path = file_info['wav']

        try:
            unique_id = make_unique_id(wav_path, input_dir)

            # Emotion classification
            emotion_before = classifier.predict(str(wav_path))
            if not emotion_before:
                print(f"  {wav_path.name} → Emotion classification failed, skipping")
                continue

            # Pitch detection with HMM / Viterbi
            times, smoothed_midi, f0, voiced_probs, y, sr = detect_pitch_pyin_hmm(
                str(wav_path),
                device=device
            )

            # Note segmentation
            detected_notes = segment_notes_hmm(times, smoothed_midi)

            # Ground truth comparison (JSON)
            gt_path = file_info.get('json')
            gt_metrics = {}
            gt_emotion = None
            if gt_path and gt_path.exists():
                gt = load_ground_truth_json(gt_path)
                if gt:
                    # Note-level metrics
                    gt_metrics = compare_notes(detected_notes, gt)
                    # Frame-level melody metrics: use smoothed MIDI as our
                    # best estimate of the melody contour.
                    import librosa
                    est_freqs = librosa.midi_to_hz(smoothed_midi)
                    frame_metrics = compute_melody_frame_metrics(times, est_freqs, gt)
                    if frame_metrics:
                        gt_metrics.update(frame_metrics)
                    gt_emotion = gt.get('emotion')

            # Convert to standard TranscribedNote objects
            notes = [
                TranscribedNote(
                    start=n['start'],
                    end=n['end'],
                    pitch_midi=n['pitch_midi'],
                    pitch_hz=n['pitch_hz'],
                    confidence=n.get('confidence', 1.0)
                )
                for n in detected_notes
            ]

            # Save result
            out_path = output_dir / f"{unique_id}_notes.json"
            save_transcription(
                notes,
                metadata={
                    'pipeline': PIPELINE_ID,
                    'pitch_method': PITCH_METHOD,
                    'gt_format': GT_FORMAT,
                    'source_audio': str(wav_path),
                    'unique_id': unique_id,
                    'bpm': 120,  # ← consider making this configurable / estimated
                    'emotion_before': emotion_before,
                    'ground_truth_metrics': gt_metrics,
                    'gt_emotion': gt_emotion,
                    'device_used': str(device),
                },
                output_path=out_path
            )

            results.append({
                'file': wav_path.name,
                'unique_id': unique_id,
                'num_notes': len(notes),
                'emotion_before': emotion_before,
                'gt_metrics': gt_metrics,
                'gt_emotion': gt_emotion,
                'output': str(out_path.relative_to(output_dir.parent.parent)),
            })

        except Exception as e:
            print(f"  Error processing {wav_path.name}: {e}")
            traceback.print_exc()

    # Final summary
    summary = {
        'pipeline': f'Pipeline {PIPELINE_ID}: {PITCH_METHOD} + {GT_FORMAT} GT',
        'device': str(device),
        'total_files': len(files),
        'successful': len(results),
        'failed': len(files) - len(results),
        'results': results,
    }

    summary_path = output_dir / f"pipeline_{PIPELINE_ID.lower()}_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nSummary saved to {summary_path}")
    print(f"[Pipeline {PIPELINE_ID}] Completed – {len(results)}/{len(files)} files processed successfully")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Pipeline C")
    parser.add_argument('--device', type=str, default=None,
                        help="Force device: cuda, mps, cpu (default = auto)")
    parser.add_argument('--max-files', type=int, default=None,
                        help="Limit number of files to process")
    parser.add_argument('--input-dir', type=str, default=None,
                        help="Input directory (overrides DATASET_DIR)")
    parser.add_argument('--model-dir', type=str, default=None,
                        help="Model directory (overrides MODEL_DIR)")

    args = parser.parse_args()

    run_pipeline(
        input_dir=args.input_dir,
        model_dir=args.model_dir,
        max_files=args.max_files,
        device=args.device,
    )