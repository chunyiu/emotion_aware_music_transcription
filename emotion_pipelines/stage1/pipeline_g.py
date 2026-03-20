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
from common.ground_truth import load_ground_truth_json, compare_notes
from common.file_discovery import discover_gtsinger_files, make_unique_id
from config import DATASET_DIR, OUTPUT_DIR, MODEL_DIR


PIPELINE_ID = "G"
PITCH_METHOD = "torchcrepe"
GT_FORMAT = "json"


def get_best_device() -> torch.device:
    """Select best available device with priority: cuda > mps > cpu"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using device: CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        # Enable fallback for ops not supported on MPS
        import os
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        print("Using device: MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("Using device: CPU (no GPU acceleration available)")
    return device


def run_pipeline(
    input_dir=None,
    output_dir=None,
    model_dir=None,
    max_files=None,
    device=None,           # now accepted from orchestrator or auto-detected
):
    # Auto-detect device if not provided
    if device is None:
        device = get_best_device()
    else:
        # If string was passed (e.g. from orchestrator), convert to torch.device
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

    classifier = EmotionClassifier(str(model_dir), device=device)

    files = discover_gtsinger_files(input_dir, max_files=max_files)
    print(f"\n[Pipeline {PIPELINE_ID}] Found {len(files)} audio files (device: {device})")

    if not files:
        print("No files to process.")
        return []

    results = []

    # ── Main processing loop with progress bar ───────────────────────────────
    for file_info in tqdm(
        files,
        desc=f"Pipeline {PIPELINE_ID} – transcribing",
        unit="file",
        dynamic_ncols=True,
        leave=True,
    ):
        wav_path = file_info['wav']

        try:
            unique_id = make_unique_id(wav_path, input_dir)

            # Emotion classification (before transcription)
            emotion_before = classifier.predict(str(wav_path))
            if not emotion_before:
                tqdm.write(f"  Skipping {wav_path.name}: emotion classification failed")
                continue

            # TorchCrepe pitch detection — pass device
            times, f0, conf = detect_pitch_torchcrepe(
                str(wav_path),
                device=device
            )
            detected_notes = segment_notes_crepe(times, f0, conf)

            gt_path = file_info['json']
            gt_metrics = {}
            gt_emotion = None
            if gt_path:
                gt = load_ground_truth_json(gt_path)
                if gt:
                    gt_metrics = compare_notes(detected_notes, gt)
                    gt_emotion = gt.get('emotion')

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
                'output': str(out_path.relative_to(output_dir)),
            })

        except Exception as e:
            tqdm.write(f"  Error processing {wav_path.name}: {e}")
            import traceback
            traceback.print_exc()

    # ── Summary ───────────────────────────────────────────────────────────────
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

    parser = argparse.ArgumentParser(description="Pipeline G: TorchCrepe pitch detection + JSON ground truth comparison")
    parser.add_argument('--max-files', type=int, default=None, help="Limit number of files to process")
    parser.add_argument('--input-dir', type=str, default=None, help="Input dataset directory")
    parser.add_argument('--model-dir', type=str, default=None, help="Directory containing emotion model")
    parser.add_argument('--device', type=str, default=None,
                        help="Device to use: cuda, mps, cpu (auto-detected if not set)")

    args = parser.parse_args()

    run_pipeline(
        max_files=args.max_files,
        input_dir=args.input_dir,
        model_dir=args.model_dir,
        device=args.device,
    )