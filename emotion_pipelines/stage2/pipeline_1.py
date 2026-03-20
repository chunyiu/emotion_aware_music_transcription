"""
Pipeline 1: Music21 transpose harmony.
Reads Stage 1 intermediate JSON, adds harmony by transposing melody down a major third.
Outputs MusicXML with melody + harmony + emotion classification.
"""

import sys
import json
import argparse
import os
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from common.note_schema import load_transcription
from common.score_utils import notes_to_music21_part, create_score, export_musicxml_safely
from common.harmony_methods import harmony_transpose
from common.emotion_classifier import EmotionClassifier
from common.musicxml_to_wav import convert_musicxml_to_wav
from config import OUTPUT_DIR, MODEL_DIR

HARMONY_ID = "1"
HARMONY_METHOD = "music21_transpose"


def get_device() -> torch.device:
    """Determine best available device with priority: cuda > mps > cpu"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using device: CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: MPS (Apple Silicon)")
        # Enable fallback for ops not supported on MPS
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    else:
        device = torch.device("cpu")
        print("Using device: CPU (no GPU acceleration available)")
    return device


def run_harmony(
    stage1_pipeline_id: str,
    stage1_output_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    model_dir: str | Path | None = None,
    device: str | torch.device | None = None,
):
    """
    Run harmony generation on all Stage 1 outputs from a given pipeline.

    Args:
        stage1_pipeline_id: e.g., "A", "B", ..., "H"
        stage1_output_dir: Directory containing Stage 1 output JSONs
        output_dir: Output directory for this combination
        model_dir: Path to emotion model
        device: Device string or torch.device ('cuda', 'mps', 'cpu')
    """
    combo_id = f"{stage1_pipeline_id}{HARMONY_ID}"

    stage1_dir = Path(stage1_output_dir or (
        OUTPUT_DIR / "stage1_transcription" / f"pipeline_{stage1_pipeline_id}"
    ))
    output_dir = Path(output_dir or (
        OUTPUT_DIR / "stage2_harmony" / combo_id
    ))
    model_dir = Path(model_dir or MODEL_DIR)

    # Resolve device
    if isinstance(device, str):
        if device == "auto":
            selected_device = get_device()
        elif device in ["cuda", "mps", "cpu"]:
            selected_device = torch.device(device)
            print(f"Using forced device: {selected_device}")
        else:
            raise ValueError(f"Unsupported device: {device}")
    elif isinstance(device, torch.device):
        selected_device = device
    else:
        # default fallback
        selected_device = get_device()

    output_dir.mkdir(parents=True, exist_ok=True)
    musicxml_dir = output_dir / "musicxml"
    wav_dir = output_dir / "wav_after"
    emotion_dir = output_dir / "emotion_results"
    musicxml_dir.mkdir(exist_ok=True)
    wav_dir.mkdir(exist_ok=True)
    emotion_dir.mkdir(exist_ok=True)

    # Initialize emotion classifier with selected device
    classifier = EmotionClassifier(
        model_dir=str(model_dir),
        device=selected_device
    )

    # Find all Stage 1 transcription JSONs
    json_files = sorted(stage1_dir.glob("*_notes.json"))
    print(f"\n[{combo_id}] Found {len(json_files)} transcription files from Pipeline {stage1_pipeline_id}")

    results = []
    for i, json_path in enumerate(json_files, 1):
        # print(f"\n[{i}/{len(json_files)}] {json_path.name}")

        try:
            notes, metadata = load_transcription(json_path)
            if not notes:
                print("  No notes, skipping")
                continue

            unique_id = metadata.get('unique_id', json_path.stem)
            bpm = metadata.get('bpm', 120)

            # Build melody part
            note_dicts = [{'pitch_midi': n.pitch_midi, 'start': n.start, 'end': n.end}
                          for n in notes]
            melody_part = notes_to_music21_part(note_dicts, bpm=bpm, part_id="Melody")

            # Generate harmony (music21 operations — usually CPU only)
            harmony_part = harmony_transpose(melody_part, interval_str='-M3')

            # Create score
            score = create_score(
                melody_part, harmony_part,
                title=unique_id,
                composer=f"{combo_id}: {HARMONY_METHOD}"
            )

            # Export MusicXML
            musicxml_path = musicxml_dir / f"{unique_id}_harmony.musicxml"
            export_musicxml_safely(score, str(musicxml_path))
            print(f"  MusicXML: {musicxml_path.name}")

            # Convert to WAV and classify emotion
            wav_path = wav_dir / f"{unique_id}_harmony.wav"
            success = convert_musicxml_to_wav(musicxml_path, wav_path)

            emotion_after = None
            if success and wav_path.exists():
                emotion_after = classifier.predict(str(wav_path))
                if emotion_after:
                    top_em = emotion_after.get('top_emotion', 'unknown')
                    top_conf = emotion_after.get('top_confidence', 0.0)
                    # print(f"  Emotion after: {top_em} ({top_conf:.1%})")

            # Save emotion comparison
            emotion_before = metadata.get('emotion_before', {})
            result = {
                'unique_id': unique_id,
                'combo': combo_id,
                'stage1_pipeline': stage1_pipeline_id,
                'harmony_method': HARMONY_METHOD,
                'num_notes': len(notes),
                'musicxml': str(musicxml_path.relative_to(output_dir)),
                'wav_after': str(wav_path.relative_to(output_dir)) if success else None,
                'emotion_before': emotion_before,
                'emotion_after': emotion_after,
                'emotion_preserved': (
                    emotion_before.get('top_emotion') == emotion_after.get('top_emotion')
                    if emotion_before and emotion_after else None
                ),
            }

            emotion_path = emotion_dir / f"{unique_id}_emotion.json"
            with open(emotion_path, 'w') as f:
                json.dump(result, f, indent=2, default=str)

            results.append(result)

        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

    # Save summary
    summary = {
        'combo': combo_id,
        'stage1_pipeline': stage1_pipeline_id,
        'harmony_method': HARMONY_METHOD,
        'total': len(json_files),
        'successful': len(results),
        'failed': len(json_files) - len(results),
        'results': results,
    }
    summary_path = output_dir / f"{combo_id}_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved to {summary_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run harmony pipeline 1 (transpose -M3)")
    parser.add_argument('--stage1', default='A', help='Stage 1 pipeline ID (A-H)')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'mps', 'cpu'],
                        help='Device to use for emotion classification model')
    args = parser.parse_args()

    run_harmony(
        stage1_pipeline_id=args.stage1,
        device=args.device
    )