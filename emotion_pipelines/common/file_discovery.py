"""
GTSinger dataset file discovery.
Finds WAV files and their associated JSON, MusicXML, and TextGrid files.
"""

import random
from pathlib import Path
from typing import List, Dict, Optional


def discover_gtsinger_files(dataset_dir, max_files=None, shuffle=False,
                            exclude_folders=("Paired_Speech_Group",)):
    """
    Discover GTSinger audio files and their associated annotation files.

    Args:
        dataset_dir: Root directory of the GTSinger dataset.
        max_files: If set, limit to this many files.
        shuffle: If True, randomly sample when max_files is set.
        exclude_folders: Folder names to exclude from search.

    Returns:
        List of dicts with keys: wav, json, musicxml, textgrid (Paths or None).
    """
    dataset_dir = Path(dataset_dir)
    audio_files = sorted(dataset_dir.rglob("*.wav"))

    # Filter out excluded folders
    if exclude_folders:
        audio_files = [
            f for f in audio_files
            if not any(ex in f.parts for ex in exclude_folders)
        ]

    if max_files and len(audio_files) > max_files:
        if shuffle:
            audio_files = random.sample(audio_files, max_files)
        else:
            audio_files = audio_files[:max_files]

    results = []
    for wav_path in audio_files:
        entry = {
            'wav': wav_path,
            'json': _find_sibling(wav_path, '.json'),
            'musicxml': _find_sibling_musicxml(wav_path),
            'textgrid': _find_sibling(wav_path, '.TextGrid'),
        }
        results.append(entry)

    return results


def _find_sibling(wav_path, suffix):
    """Find a sibling file with the given suffix."""
    candidate = wav_path.with_suffix(suffix)
    return candidate if candidate.exists() else None


def _find_sibling_musicxml(wav_path):
    """Find a sibling MusicXML file (.musicxml or .xml)."""
    for suffix in ('.musicxml', '.xml', '.mxl'):
        candidate = wav_path.with_suffix(suffix)
        if candidate.exists():
            return candidate
    return None


def make_unique_id(wav_path, base_dir):
    """Create a unique ID from the relative path structure."""
    try:
        rel_path = wav_path.relative_to(base_dir)
        folder_parts = rel_path.parts[:-1]
        unique_id = "_".join(folder_parts) + "_" + wav_path.stem
        return unique_id.replace(" ", "_").replace("-", "_")
    except ValueError:
        return wav_path.stem
