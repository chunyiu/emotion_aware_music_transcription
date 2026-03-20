"""
Standardized note schema for intermediate data exchange between Stage 1 and Stage 2.
"""

import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any


@dataclass
class TranscribedNote:
    """A single detected note."""
    start: float       # onset time in seconds
    end: float         # offset time in seconds
    pitch_midi: int    # MIDI note number (rounded)
    pitch_hz: float    # frequency in Hz
    confidence: float  # detection confidence (0-1)


def notes_to_dicts(notes: List[TranscribedNote]) -> List[Dict]:
    """Convert TranscribedNote list to list of dicts."""
    return [asdict(n) for n in notes]


def dicts_to_notes(dicts: List[Dict]) -> List[TranscribedNote]:
    """Convert list of dicts back to TranscribedNote list."""
    return [TranscribedNote(**d) for d in dicts]


def save_transcription(notes: List[TranscribedNote], metadata: Dict[str, Any],
                       output_path):
    """
    Save transcription result as intermediate JSON.

    Args:
        notes: List of TranscribedNote objects.
        metadata: Dict with keys like pipeline, pitch_method, gt_format,
                  source_audio, bpm, estimated_key, emotion_before,
                  ground_truth_metrics, etc.
        output_path: Path to write the JSON file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        **metadata,
        'notes': notes_to_dicts(notes),
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, default=str)

    return output_path


def load_transcription(json_path):
    """
    Load a transcription JSON.

    Returns:
        (notes: List[TranscribedNote], metadata: dict)
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    notes_data = data.pop('notes', [])
    notes = dicts_to_notes(notes_data)
    metadata = data

    return notes, metadata
