import sys
import json
import traceback
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from common.emotion_classifier import EmotionClassifier
from common.pitch_detectors import detect_pitch_pyin_hmm
from common.note_segmentation import segment_notes_hmm
from common.note_schema import TranscribedNote, save_transcription
from common.ground_truth import load_ground_truth_musicxml, compare_notes
from common.file_discovery import discover_gtsinger_files, make_unique_id
from config import DATASET_DIR, OUTPUT_DIR, MODEL_DIR


PIPELINE_ID = "D"
PITCH_METHOD = "pyin_hmm_viterbi"
GT_FORMAT = "musicxml"


def get_best_device() -> torch.device:
    """Determine best available device with priority: cuda > mps > cpu"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using device: CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: MPS (Apple Silicon)")
        # Enable fallback for unsupported PyTorch ops on MPS
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
    device=None,           # new optional argument
):
    # ──────────────────────────────────────────────────────────────
    # Device setup
    # ──────────────────────────────────────────────────────────────
    if device is None:
        device = get_best_device()
    elif isinstance(device, str):
        if device == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
        elif device == "mps" and torch.backends.mps.is_available():
            device = torch.device("mps")
            import os
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        else:
            device = torch.device("cpu")
        print(f"Requested device: {device}")
    else:
        # already a torch.device
        pass

    # ──────────────────────────────────────────────────────────────
    # Paths & initialization
    # ──────────────────────────────────────────────────────────────
    input_dir = Path(input_dir or DATASET_DIR)
    output_dir = Path(output_dir or OUTPUT_DIR) / "stage1_transcription" / f"pipeline_{PIPELINE_ID}"
    model_dir = Path(model_dir or MODEL_DIR)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Pipeline {PIPELINE_ID}] Loading emotion classifier on {device} ...")
    classifier = EmotionClassifier(str(model_dir), device=device)

    files = discover_gtsinger_files(input_dir, max_files=max_files)
    print(f"\n[Pipeline {PIPELINE_ID}] Found {len(files)} audio files\n")

    results = []

    # ──── Main processing loop with progress bar ───────────────────
    for file_info in tqdm(
        files,
        desc=f"Processing files (Pipeline {PIPELINE_ID})",
        unit="file",
        ncols=100,
        dynamic_ncols=True
    ):
        wav_path = file_info['wav']

        try:
            unique_id = make_unique_id(wav_path, input_dir)

            # Emotion prediction
            emotion_before = classifier.predict(str(wav_path))
            if not emotion_before:
                tqdm.write(f"    → Emotion classification failed for {wav_path.name}, skipping")
                continue

            # Pitch detection
            times, smoothed_midi, f0, voiced_probs, y, sr = detect_pitch_pyin_hmm(
                str(wav_path),
                device=device
            )

            detected_notes = segment_notes_hmm(times, smoothed_midi)

            # Ground truth comparison (if available)
            gt_path = file_info.get('musicxml')
            gt_metrics = {}
            if gt_path and gt_path.exists():
                gt = load_ground_truth_musicxml(gt_path)
                if gt:
                    gt_metrics = compare_notes(detected_notes, gt)

            # Prepare notes for saving
            notes = [
                TranscribedNote(
                    start=n['start'], end=n['end'],
                    pitch_midi=n['pitch_midi'], pitch_hz=n['pitch_hz'],
                    confidence=n['confidence']
                )
                for n in detected_notes
            ]

            out_path = output_dir / f"{unique_id}_notes.json"
            save_transcription(
                notes,
                metadata={
                    'pipeline': PIPELINE_ID,
                    'pitch_method': PITCH_METHOD,
                    'gt_format': GT_FORMAT,
                    'source_audio': str(wav_path),
                    'unique_id': unique_id,
                    'bpm': 120,
                    'emotion_before': emotion_before,
                    'ground_truth_metrics': gt_metrics,
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
                'output': str(out_path.relative_to(output_dir.parent.parent)),
            })

        except Exception as e:
            tqdm.write(f"    Error processing {wav_path.name}: {e}")
            # traceback.print_exc()   # uncomment during debugging

    # ──────────────────────────────────────────────────────────────
    # Summary
    # ──────────────────────────────────────────────────────────────
    summary = {
        'pipeline': f'Pipeline {PIPELINE_ID}: {PITCH_METHOD} + {GT_FORMAT} GT',
        'device': str(device),
        'total_files': len(files),
        'successful': len(results),
        'failed': len(files) - len(results),
        'results': results,
    }

    summary_path = output_dir / f"pipeline_{PIPELINE_ID.lower()}_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n[Pipeline {PIPELINE_ID}] Summary saved to {summary_path}")
    print(f"Completed {len(results)} / {len(files)} files successfully.\n")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Pipeline D: pYIN+HMM/Viterbi + MusicXML comparison")
    parser.add_argument('--max-files', type=int, default=None,
                        help="Limit number of files to process (useful for testing)")
    parser.add_argument('--device', type=str, default=None,
                        help="Device to use: cuda, mps, cpu (default: auto-detect)")
    args = parser.parse_args()

    run_pipeline(
        max_files=args.max_files,
        device=args.device
    )