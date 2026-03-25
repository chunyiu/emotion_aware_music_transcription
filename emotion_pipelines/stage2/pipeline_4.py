"""
Pipeline 4: Circle-of-fifths voice leading with secondary dominants.
Reads Stage 1 intermediate JSON, adds harmony using circle-of-fifths progressions
with secondary dominants for chromatic color.
Outputs MusicXML with melody + chord harmony + emotion analysis.
"""

import argparse
import json
import sys
import traceback
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # adjust if needed

from common.note_schema import load_transcription
from common.score_utils import notes_to_music21_part, create_score, export_musicxml_safely
from common.harmony_methods import harmony_circle_of_fifths
from common.emotion_classifier import EmotionClassifier
from common.musicxml_to_wav import convert_musicxml_to_wav
from config import OUTPUT_DIR, MODEL_DIR

HARMONY_ID = "4"
HARMONY_METHOD = "circle_of_fifths"


def get_best_device(requested: str = "auto") -> torch.device:
    """Select best available device with priority: cuda > mps > cpu"""
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        # Force requested device (will raise error if not available)
        dev = torch.device(requested)
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available")
        return dev


def run_harmony(
    stage1_pipeline_id: str,
    stage1_output_dir=None,
    output_dir=None,
    model_dir=None,
    device: str = "auto",
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
    musicxml_dir.mkdir(exist_ok=True)
    wav_dir.mkdir(exist_ok=True)
    emotion_dir.mkdir(exist_ok=True)

    # Determine device
    torch_device = get_best_device(device)
    print(f"[{combo_id}] Using device: {torch_device}")

    # Enable MPS fallback if using MPS
    if torch_device.type == "mps":
        import os
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    classifier = EmotionClassifier(model_dir=str(model_dir), device=torch_device)

    json_files = sorted(stage1_dir.glob("*_notes.json"))
    if not json_files:
        print(f"[{combo_id}] No transcription files found in {stage1_dir}")
        return []

    # print(f"[{combo_id}] Processing {len(json_files)} files from Pipeline {stage1_pipeline_id}")

    results = []

    pbar = tqdm(json_files, desc=f"Harmony {combo_id}", unit="file")
    for json_path in pbar:
        pbar.set_postfix_str(json_path.name)

        try:
            notes, metadata = load_transcription(json_path)
            if not notes:
                continue

            unique_id = metadata.get('unique_id', json_path.stem)
            bpm = metadata.get('bpm', 120)

            note_dicts = [{'pitch_midi': n.pitch_midi, 'start': n.start, 'end': n.end}
                          for n in notes]
            melody_part = notes_to_music21_part(note_dicts, bpm=bpm, part_id="Melody")

            # Generate circle-of-fifths harmony with secondary dominants
            harmony_part, est_key = harmony_circle_of_fifths(melody_part)

            score = create_score(
                melody_part, harmony_part,
                title=unique_id,
                composer=f"{combo_id}: {HARMONY_METHOD} | Key: {est_key}"
            )

            musicxml_path = musicxml_dir / f"{unique_id}_harmony.musicxml"
            export_musicxml_safely(score, str(musicxml_path))

            wav_path = wav_dir / f"{unique_id}_harmony.wav"
            success = convert_musicxml_to_wav(musicxml_path, wav_path)

            emotion_after = None
            if success and wav_path.exists():
                emotion_after = classifier.predict(str(wav_path))

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
            with open(emotion_path, 'w') as f:
                json.dump(result, f, indent=2, default=str)

            results.append(result)

        except Exception as e:
            print(f"  Error processing {json_path.name}: {e}")
            traceback.print_exc()

    # Summary
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

    print(f"[{combo_id}] Summary saved -> {summary_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run harmony pipeline 4 (circle-of-fifths + secondary dominants)")
    parser.add_argument('--stage1', default='A', help='Stage 1 pipeline ID (A-H)')
    parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'mps', 'cpu'],
                        help='Device to use for emotion classification (default: auto)')
    args = parser.parse_args()

    run_harmony(
        stage1_pipeline_id=args.stage1,
        device=args.device
    )