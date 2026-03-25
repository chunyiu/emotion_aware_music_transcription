"""
Pipeline 3: Diatonic Roman numeral harmony (Krumhansl-Schmuckler key detection).
Reads Stage 1 intermediate JSON, adds harmony using diatonic chord analysis.
Outputs MusicXML with melody + chord harmony + post-harmony emotion analysis.
"""

import sys
import json
import argparse
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from common.note_schema import load_transcription
from common.score_utils import notes_to_music21_part, create_score, export_musicxml_safely
from common.harmony_methods import harmony_diatonic_roman
from common.emotion_classifier import EmotionClassifier
from common.musicxml_to_wav import convert_musicxml_to_wav
from config import OUTPUT_DIR, MODEL_DIR


HARMONY_ID = "3"
HARMONY_METHOD = "diatonic_roman"


def get_best_device(preferred: str = None) -> torch.device:
    """
    Select best available device with priority:
    1. Explicitly requested device (if available)
    2. cuda > mps > cpu
    """
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preferred == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if preferred == "cpu":
        return torch.device("cpu")

    # Auto fallback order
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def run_harmony(
    stage1_pipeline_id: str,
    stage1_output_dir: str = None,
    output_dir: str = None,
    model_dir: str = None,
    device: str = None,
):
    combo_id = f"{stage1_pipeline_id}{HARMONY_ID}"

    stage1_dir = Path(stage1_output_dir or (
        OUTPUT_DIR / "stage1_transcription" / f"pipeline_{stage1_pipeline_id}"
    ))
    output_dir = Path(output_dir or (
        OUTPUT_DIR / "stage2_harmony" / combo_id
    ))
    model_dir = Path(model_dir or MODEL_DIR)

    output_dir.mkdir(parents=True, exist_ok=True)
    musicxml_dir = output_dir / "musicxml"
    wav_dir = output_dir / "wav_after"
    emotion_dir = output_dir / "emotion_results"

    for d in [musicxml_dir, wav_dir, emotion_dir]:
        d.mkdir(exist_ok=True)

    # Determine device
    selected_device = get_best_device(device)
    print(f"[{combo_id}] Using device: {selected_device}")

    # Enable MPS fallback if using MPS
    if selected_device.type == "mps":
        import os
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    classifier = EmotionClassifier(
        model_dir=str(model_dir),
        device=selected_device   # ← pass device to classifier (must be supported)
    )

    json_files = sorted(stage1_dir.glob("*_notes.json"))
    print(f"[{combo_id}] Found {len(json_files)} transcription files from Pipeline {stage1_pipeline_id}")

    results = []

    for i, json_path in enumerate(json_files, 1):
        # print(f"\n[{i}/{len(json_files)}] Processing: {json_path.name}")

        try:
            notes, metadata = load_transcription(json_path)
            if not notes:
                print("  -> No notes found, skipping")
                continue

            unique_id = metadata.get('unique_id', json_path.stem)
            bpm = metadata.get('bpm', 120)

            note_dicts = [
                {'pitch_midi': n.pitch_midi, 'start': n.start, 'end': n.end}
                for n in notes
            ]

            melody_part = notes_to_music21_part(note_dicts, bpm=bpm, part_id="Melody")

            # Generate diatonic Roman numeral harmony
            harmony_part, est_key = harmony_diatonic_roman(melody_part)

            score = create_score(
                melody_part, harmony_part,
                title=unique_id,
                composer=f"{combo_id}: {HARMONY_METHOD} | Key: {est_key}"
            )

            musicxml_path = musicxml_dir / f"{unique_id}_harmony.musicxml"
            export_musicxml_safely(score, str(musicxml_path))
            # print(f"  -> MusicXML saved: {musicxml_path.name} (Key: {est_key})")

            wav_path = wav_dir / f"{unique_id}_harmony.wav"
            success = convert_musicxml_to_wav(musicxml_path, wav_path)

            emotion_after = None
            if success and wav_path.exists():
                emotion_after = classifier.predict(str(wav_path))
                if emotion_after:
                    top_emotion = emotion_after.get('top_emotion', 'unknown')
                    top_conf = emotion_after.get('top_confidence', 0.0)
                    # print(f"  -> Emotion after harmony: {top_emotion} ({top_conf:.1%})")

            emotion_before = metadata.get('emotion_before', {})

            result = {
                'unique_id': unique_id,
                'combo': combo_id,
                'stage1_pipeline': stage1_pipeline_id,
                'harmony_method': HARMONY_METHOD,
                'estimated_key': str(est_key),
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
            with open(emotion_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, default=str)

            results.append(result)

        except Exception as e:
            print(f"  -> Error processing {json_path.name}: {e}")
            import traceback
            traceback.print_exc()

    # Final summary
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
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n[{combo_id}] Summary saved -> {summary_path}")
    print(f"[{combo_id}] Completed: {len(results)} / {len(json_files)} files processed successfully")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run diatonic Roman numeral harmony (Pipeline 3)")
    parser.add_argument('--stage1', default='A', help='Stage 1 pipeline ID (A-H)')
    parser.add_argument('--device', type=str, choices=['auto', 'cuda', 'mps', 'cpu'],
                        default='auto', help='Device to use (default: auto = cuda > mps > cpu)')
    args = parser.parse_args()

    run_harmony(
        stage1_pipeline_id=args.stage1,
        device=args.device,
    )