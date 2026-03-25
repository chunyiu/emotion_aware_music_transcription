"""
Pipeline 2: Mingus algorithmic chord analysis harmony.
Reads Stage 1 intermediate JSON, adds harmony using mingus chord determination.
Outputs MusicXML with melody + bass harmony.
"""

import sys
import json
import argparse
from pathlib import Path
import traceback

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from common.note_schema import load_transcription
from common.score_utils import notes_to_music21_part, create_score, export_musicxml_safely
from common.harmony_methods import harmony_mingus_chord
from common.emotion_classifier import EmotionClassifier
from common.musicxml_to_wav import convert_musicxml_to_wav
from config import OUTPUT_DIR, MODEL_DIR


HARMONY_ID = "2"
HARMONY_METHOD = "mingus_chord"


def get_best_device(preferred: str = None) -> torch.device:
    """
    Determine best available device with priority: cuda > mps > cpu
    If preferred is given and available, use it; otherwise fall back.
    """
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preferred == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if preferred == "cpu":
        return torch.device("cpu")

    # Auto mode
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def run_harmony(
    stage1_pipeline_id: str,
    stage1_output_dir=None,
    output_dir=None,
    model_dir=None,
    device: str = "auto",
    preferred_device: str = None,   # allow forcing via CLI
):
    combo_id = f"{stage1_pipeline_id}{HARMONY_ID}"

    # ────────────────────────────────────────────────
    # Device setup
    # ────────────────────────────────────────────────
    selected_device = get_best_device(preferred=preferred_device or device)
    print(f"[{combo_id}] Using device: {selected_device}")

    # Enable MPS fallback if needed
    if selected_device.type == "mps":
        import os
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    stage1_dir = Path(stage1_output_dir or (
        OUTPUT_DIR / "stage1_transcription" / f"pipeline_{stage1_pipeline_id}"
    ))
    output_dir = Path(output_dir or (
        OUTPUT_DIR / "stage2_harmony" / combo_id
    ))
    model_dir = Path(model_dir or MODEL_DIR)

    output_dir.mkdir(parents=True, exist_ok=True)
    musicxml_dir = output_dir / "musicxml"
    wav_dir      = output_dir / "wav_after"
    emotion_dir  = output_dir / "emotion_results"

    for d in [musicxml_dir, wav_dir, emotion_dir]:
        d.mkdir(exist_ok=True)

    # Initialize emotion classifier on selected device
    try:
        classifier = EmotionClassifier(
            model_dir=str(model_dir),
            device=selected_device
        )
        print(f"[{combo_id}] Emotion classifier loaded on {selected_device}")
    except Exception as e:
        print(f"[{combo_id}] Failed to load emotion classifier: {e}")
        classifier = None

    json_files = sorted(stage1_dir.glob("*_notes.json"))
    print(f"[{combo_id}] Found {len(json_files)} transcription files from Pipeline {stage1_pipeline_id}")

    results = []
    for i, json_path in enumerate(json_files, 1):
        # print(f"\n[{combo_id}] [{i}/{len(json_files)}] {json_path.name}")

        try:
            notes, metadata = load_transcription(json_path)
            if not notes:
                print("  -> No notes, skipping")
                continue

            unique_id = metadata.get('unique_id', json_path.stem)
            bpm = metadata.get('bpm', 120)

            note_dicts = [
                {'pitch_midi': n.pitch_midi, 'start': n.start, 'end': n.end}
                for n in notes
            ]

            melody_part = notes_to_music21_part(note_dicts, bpm=bpm, part_id="Melody")

            # Mingus harmony (currently CPU-only logic — no device needed)
            harmony_part = harmony_mingus_chord(melody_part)

            score = create_score(
                melody_part, harmony_part,
                title=unique_id,
                composer=f"{combo_id}: {HARMONY_METHOD}"
            )

            musicxml_path = musicxml_dir / f"{unique_id}_harmony.musicxml"
            export_musicxml_safely(score, str(musicxml_path))
            # print(f"  -> MusicXML: {musicxml_path.name}")

            wav_path = wav_dir / f"{unique_id}_harmony.wav"
            success = convert_musicxml_to_wav(musicxml_path, wav_path)

            emotion_after = None
            if success and wav_path.exists() and classifier is not None:
                emotion_after = classifier.predict(str(wav_path))
                if emotion_after:
                    top_emo = emotion_after.get('top_emotion', 'unknown')
                    top_conf = emotion_after.get('top_confidence', 0.0)
                    # print(f"  -> Emotion after: {top_emo} ({top_conf:.1%})")

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
                'device_used': str(selected_device),
            }

            emotion_path = emotion_dir / f"{unique_id}_emotion.json"
            with open(emotion_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, default=str)

            results.append(result)

        except Exception as e:
            print(f"  -> Error: {e}")
            traceback.print_exc()

    # ────────────────────────────────────────────────
    # Summary
    # ────────────────────────────────────────────────
    summary = {
        'combo': combo_id,
        'stage1_pipeline': stage1_pipeline_id,
        'harmony_method': HARMONY_METHOD,
        'device': str(selected_device),
        'total': len(json_files),
        'successful': len(results),
        'failed': len(json_files) - len(results),
        'results': results,
    }

    summary_path = output_dir / f"{combo_id}_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n[{combo_id}] Summary saved to {summary_path}")
    print(f"[{combo_id}] Completed – {len(results)} / {len(json_files)} files processed successfully")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Pipeline 2: Mingus chord harmony")
    parser.add_argument('--stage1', default='A', help='Stage 1 pipeline ID (A-H)')
    parser.add_argument('--device', default='auto',
                        choices=['auto', 'cuda', 'mps', 'cpu'],
                        help='Device to use (default: auto = cuda > mps > cpu)')
    parser.add_argument('--model-dir', type=str, default=None,
                        help='Override emotion model directory')
    args = parser.parse_args()

    run_harmony(
        stage1_pipeline_id=args.stage1,
        model_dir=args.model_dir,
        device=args.device,
    )