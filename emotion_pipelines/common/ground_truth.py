"""
Ground truth loading and comparison for GTSinger dataset.
Supports both JSON and MusicXML ground truth formats.
"""

import json
import numpy as np
from pathlib import Path


def load_ground_truth_json(json_path):
    """
    Load ground truth from GTSinger JSON format.

    GTSinger JSON is a list of word entries, each with:
        note_start: list of floats (onset times)
        note_end: list of floats (offset times)
        note: list of ints (MIDI pitch, 0 = silence)
        emotion: string

    Returns:
        dict with keys:
            intervals: np.array of shape (N, 2) - [start, end] pairs
            pitches: np.array of shape (N,) - MIDI pitch values
            emotion: str or None
    """
    json_path = Path(json_path)
    if not json_path.exists():
        return None

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Handle GTSinger format: may be a list or dict with 'value' key
        entries = data.get('value', []) if isinstance(data, dict) else data

        intervals = []
        pitches = []
        emotion = None

        for entry in entries:
            if not isinstance(entry, dict):
                continue

            note_starts = entry.get('note_start', [])
            note_ends = entry.get('note_end', [])
            notes = entry.get('note', [])

            for start, end, midi_note in zip(note_starts, note_ends, notes):
                if midi_note > 0:
                    intervals.append([float(start), float(end)])
                    pitches.append(float(midi_note))

            if not emotion and entry.get('emotion'):
                emotion = entry['emotion']

        return {
            'intervals': np.array(intervals) if intervals else np.array([]).reshape(0, 2),
            'pitches': np.array(pitches),
            'emotion': emotion
        }

    except Exception as e:
        print(f"Error loading JSON ground truth: {e}")
        return None


def load_ground_truth_musicxml(musicxml_path):
    """
    Load ground truth from MusicXML file using music21.

    Returns:
        dict with keys:
            intervals: np.array of shape (N, 2) - [start, end] pairs (in seconds)
            pitches: np.array of shape (N,) - MIDI pitch values
            emotion: None (MusicXML doesn't store emotion labels)
    """
    from music21 import converter

    musicxml_path = Path(musicxml_path)
    if not musicxml_path.exists():
        return None

    try:
        score = converter.parse(str(musicxml_path))

        # Try to get tempo for time conversion
        tempos = list(score.flatten().getElementsByClass('MetronomeMark'))
        bpm = tempos[0].number if tempos else 120.0
        sec_per_beat = 60.0 / bpm

        intervals = []
        pitches = []

        for element in score.flatten().notesAndRests:
            if hasattr(element, 'pitch'):
                onset_beats = float(element.offset)
                duration_beats = float(element.quarterLength)
                onset_sec = onset_beats * sec_per_beat
                offset_sec = (onset_beats + duration_beats) * sec_per_beat

                intervals.append([onset_sec, offset_sec])
                pitches.append(float(element.pitch.midi))

        return {
            'intervals': np.array(intervals) if intervals else np.array([]).reshape(0, 2),
            'pitches': np.array(pitches),
            'emotion': None
        }

    except Exception as e:
        print(f"Error loading MusicXML ground truth: {e}")
        return None


def compare_notes(detected_notes, ground_truth, onset_tolerance=0.1,
                  pitch_tolerance=0.5):
    """
    Compare detected notes against ground truth using mir_eval.

    Args:
        detected_notes: list of note dicts with start, end, pitch_midi keys
        ground_truth: dict from load_ground_truth_json/musicxml

    Returns:
        dict with precision, recall, f1, ref_count, pred_count
    """
    import mir_eval

    if ground_truth is None or len(ground_truth['pitches']) == 0:
        return _empty_metrics()

    ref_intervals = ground_truth['intervals']
    ref_pitches = ground_truth['pitches']

    if not detected_notes:
        return _empty_metrics(ref_count=len(ref_pitches))

    pred_intervals = np.array([[n['start'], n['end']] for n in detected_notes])
    pred_pitches = np.array([n['pitch_midi'] for n in detected_notes])

    try:
        precision, recall, f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
            ref_intervals=ref_intervals,
            ref_pitches=ref_pitches,
            est_intervals=pred_intervals,
            est_pitches=pred_pitches,
            onset_tolerance=onset_tolerance,
            pitch_tolerance=pitch_tolerance
        )
    except Exception as e:
        print(f"  mir_eval error: {e}")
        precision = recall = f1 = 0.0

    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'ref_count': int(len(ref_pitches)),
        'pred_count': int(len(pred_pitches))
    }


def _empty_metrics(ref_count=0, pred_count=0):
    return {
        'precision': 0.0,
        'recall': 0.0,
        'f1': 0.0,
        'ref_count': ref_count,
        'pred_count': pred_count
    }
