"""
Pipeline H: TorchCrepe (PyTorch) pitch detection + MusicXML ground truth comparison.
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from common.emotion_classifier import EmotionClassifier
from common.pitch_detectors import detect_pitch_torchcrepe
from common.note_segmentation import segment_notes_crepe
from common.note_schema import TranscribedNote, save_transcription
from common.ground_truth import load_ground_truth_musicxml, compare_notes
from common.file_discovery import discover_gtsinger_files, make_unique_id
from config import DATASET_DIR, OUTPUT_DIR, MODEL_DIR

PIPELINE_ID = "H"
PITCH_METHOD = "torchcrepe"
GT_FORMAT = "musicxml"


def run_pipeline(input_dir=None, output_dir=None, model_dir=None, max_files=None):
    input_dir = Path(input_dir or DATASET_DIR)
    output_dir = Path(output_dir or OUTPUT_DIR) / "stage1_transcription" / f"pipeline_{PIPELINE_ID}"
    model_dir = Path(model_dir or MODEL_DIR)

    output_dir.mkdir(parents=True, exist_ok=True)
    classifier = EmotionClassifier(str(model_dir))

    files = discover_gtsinger_files(input_dir, max_files=max_files)
    print(f"\n[Pipeline {PIPELINE_ID}] Found {len(files)} audio files")

    results = []
    for i, file_info in enumerate(files, 1):
        wav_path = file_info['wav']
        print(f"\n[{i}/{len(files)}] {wav_path.name}")

        try:
            unique_id = make_unique_id(wav_path, input_dir)

            emotion_before = classifier.predict(str(wav_path))
            if not emotion_before:
                continue

            print(f"  Emotion: {emotion_before['top_emotion']} ({emotion_before['top_confidence']:.1%})")

            times, f0, conf = detect_pitch_torchcrepe(str(wav_path))
            detected_notes = segment_notes_crepe(times, f0, conf)
            print(f"  Detected {len(detected_notes)} notes")

            gt_path = file_info['musicxml']
            gt_metrics = {}
            if gt_path:
                gt = load_ground_truth_musicxml(gt_path)
                if gt:
                    gt_metrics = compare_notes(detected_notes, gt)
                    print(f"  GT F1: {gt_metrics.get('f1', 0):.3f}")

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
                'pipeline': PIPELINE_ID, 'pitch_method': PITCH_METHOD,
                'gt_format': GT_FORMAT, 'source_audio': str(wav_path),
                'unique_id': unique_id, 'bpm': 120,
                'emotion_before': emotion_before,
                'ground_truth_metrics': gt_metrics,
            }, out_path)

            results.append({
                'file': wav_path.name, 'unique_id': unique_id,
                'num_notes': len(notes), 'emotion_before': emotion_before,
                'gt_metrics': gt_metrics,
                'output': str(out_path.relative_to(output_dir)),
            })

        except Exception as e:
            print(f"  Error: {e}")
            import traceback; traceback.print_exc()

    summary = {
        'pipeline': f'Pipeline {PIPELINE_ID}: {PITCH_METHOD} + {GT_FORMAT} GT',
        'total_files': len(files), 'successful': len(results),
        'failed': len(files) - len(results), 'results': results,
    }
    summary_path = output_dir / f"pipeline_{PIPELINE_ID.lower()}_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved to {summary_path}")
    return results


if __name__ == "__main__":
    run_pipeline()
