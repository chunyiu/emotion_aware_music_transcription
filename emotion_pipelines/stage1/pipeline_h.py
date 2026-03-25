import sys
import json
from pathlib import Path
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from common.emotion_classifier import EmotionClassifier
from common.pitch_detectors import detect_pitch_torchcrepe
from common.note_segmentation import segment_notes_crepe
from common.note_schema import TranscribedNote, save_transcription
from common.ground_truth import load_ground_truth_musicxml, compare_notes, compute_melody_frame_metrics
from common.file_discovery import discover_gtsinger_files, make_unique_id
from common.csv_emotions import load_csv_emotions, wav_to_csv_key
from config import DATASET_DIR, OUTPUT_DIR, MODEL_DIR

CSV_PATH = Path(__file__).parent.parent / "gtsinger_english_emotions.csv"

PIPELINE_ID = "H"
PITCH_METHOD = "torchcrepe"
GT_FORMAT = "musicxml"


def get_best_device() -> torch.device:
    """Determine best available device with priority: CUDA > MPS > CPU"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using device: CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: MPS (Apple Silicon)")
        # Enable fallback for unsupported ops on MPS
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
    device=None,           # ← new optional argument
):
    # Determine device if not provided
    if device is None:
        device = get_best_device()
    else:
        # If passed as string (e.g. from orchestrator), convert to torch.device
        if isinstance(device, str):
            if device == "cuda" and torch.cuda.is_available():
                device = torch.device("cuda")
            elif device == "mps" and torch.backends.mps.is_available():
                device = torch.device("mps")
                import os
                os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            else:
                device = torch.device("cpu")
        print(f"Using provided device: {device}")

    input_dir = Path(input_dir or DATASET_DIR)
    output_dir = Path(output_dir or OUTPUT_DIR) / "stage1_transcription" / f"pipeline_{PIPELINE_ID}"
    model_dir = Path(model_dir or MODEL_DIR)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Pass device to EmotionClassifier (assuming it accepts device)
    classifier = EmotionClassifier(str(model_dir), device=device)

    csv_emotions = load_csv_emotions(CSV_PATH)

    files = discover_gtsinger_files(input_dir, max_files=max_files)
    print(f"\n[Pipeline {PIPELINE_ID}] Found {len(files)} audio files")

    results = []

    # ── Add tqdm progress bar ────────────────────────────────────────
    file_iterator = tqdm(
        files,
        desc=f"Pipeline {PIPELINE_ID} ({PITCH_METHOD})",
        unit="file",
        ncols=100,
        leave=True,
    )
    # ─────────────────────────────────────────────────────────────────

    for i, file_info in enumerate(file_iterator, 1):
        wav_path = file_info['wav']

        # Optional: show current file name in progress bar (can be removed if too noisy)
        file_iterator.set_postfix(file=wav_path.name, refresh=True)

        try:
            unique_id = make_unique_id(wav_path, input_dir)

            # Emotion classification
            emotion_before = classifier.predict(str(wav_path))
            if not emotion_before:
                print(f"  Skipping {wav_path.name}: emotion classification failed")
                continue

            # Pitch detection
            times, f0, conf = detect_pitch_torchcrepe(
                str(wav_path),
                device=device
            )
            detected_notes = segment_notes_crepe(times, f0, conf)

            # gt_emotion from CSV (MusicXML has no emotion field)
            csv_key = wav_to_csv_key(wav_path, input_dir)
            gt_emotion = csv_emotions.get(csv_key, "")

            gt_path = file_info['musicxml']
            gt_metrics = {}
            if gt_path:
                gt = load_ground_truth_musicxml(gt_path)
                if gt:
                    # Note-level metrics
                    gt_metrics = compare_notes(detected_notes, gt)
                    # Frame-level melody metrics from TorchCrepe F0
                    frame_metrics = compute_melody_frame_metrics(times, f0, gt)
                    if frame_metrics:
                        gt_metrics.update(frame_metrics)

            notes = [
                TranscribedNote(
                    start=n['start'], end=n['end'],
                    pitch_midi=n['pitch_midi'], pitch_hz=n['pitch_hz'],
                    confidence=n['confidence']
                )
                for n in detected_notes
            ]

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
                'device_used': str(device),
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
            import traceback
            traceback.print_exc()

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

    print(f"\nSummary saved to {summary_path}")
    print(f"Processed {len(results)} / {len(files)} files successfully")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Pipeline H")
    parser.add_argument('--device', type=str, default=None,
                        help="Device to use: cuda, mps, cpu (auto-detected if not set)")
    parser.add_argument('--max-files', type=int, default=None,
                        help="Limit number of files to process")
    args = parser.parse_args()

    run_pipeline(device=args.device, max_files=args.max_files)