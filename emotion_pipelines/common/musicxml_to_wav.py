"""
Consolidated MusicXML/MIDI to WAV conversion.
Supports MuseScore, FluidSynth, and simple sine-wave fallback.
"""

import subprocess
import tempfile
import os
from pathlib import Path


def musicxml_to_wav_musescore(musicxml_path, output_wav_path,
                              musescore_exe=r"C:\Program Files\MuseScore 4\bin\musescore4.exe"):
    """Convert MusicXML to WAV using MuseScore's command-line interface."""
    try:
        cmd = [musescore_exe, '-o', str(output_wav_path), str(musicxml_path)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[ERROR] MuseScore failed (rc={result.returncode}): {result.stderr[:200]}")
            return False
        return True
    except Exception as e:
        print(f"[ERROR] MuseScore conversion exception: {e}")
        return False


def midi_to_wav_musescore(midi_path, output_wav_path,
                          musescore_exe=r"C:\Program Files\MuseScore 4\bin\musescore4.exe"):
    """Convert MIDI to WAV using MuseScore."""
    try:
        cmd = [musescore_exe, '-o', str(output_wav_path), str(midi_path)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[ERROR] MuseScore MIDI failed (rc={result.returncode}): {result.stderr[:200]}")
            return False
        return True
    except Exception as e:
        print(f"[ERROR] MuseScore MIDI conversion exception: {e}")
        return False


def musicxml_to_wav_simple(musicxml_path, output_wav_path, sample_rate=22050):
    """Simple MusicXML to WAV converter using sine waves (no external deps)."""
    try:
        import numpy as np
        import librosa
        import soundfile as sf
        from music21 import converter

        score = converter.parse(str(musicxml_path))
        notes_list = []
        for element in score.flatten().notesAndRests:
            if hasattr(element, 'pitch'):
                notes_list.append({
                    'pitch': element.pitch.midi,
                    'start': element.offset,
                    'duration': element.quarterLength,
                })

        if not notes_list:
            return False

        bpm = 120
        quarter_duration = 60.0 / bpm
        max_end = max(n['start'] + n['duration'] for n in notes_list)
        total_samples = int(max_end * quarter_duration * sample_rate)
        audio = np.zeros(total_samples)

        for note_info in notes_list:
            freq = librosa.midi_to_hz(note_info['pitch'])
            start_sample = int(note_info['start'] * quarter_duration * sample_rate)
            duration_samples = int(note_info['duration'] * quarter_duration * sample_rate)
            end_sample = min(start_sample + duration_samples, total_samples)
            actual = end_sample - start_sample
            if actual <= 0:
                continue

            t = np.arange(actual) / sample_rate
            sine_wave = 0.3 * np.sin(2 * np.pi * freq * t)

            # Envelope
            attack = min(100, actual // 4)
            decay = min(100, actual // 4)
            envelope = np.ones(actual)
            if attack > 0:
                envelope[:attack] = np.linspace(0, 1, attack)
            if decay > 0:
                envelope[-decay:] = np.linspace(1, 0, decay)
            sine_wave *= envelope

            audio[start_sample:end_sample] += sine_wave

        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.9

        Path(output_wav_path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_wav_path), audio, sample_rate)
        return True

    except Exception as e:
        print(f"Error in simple MusicXML to WAV conversion: {e}")
        return False


def convert_musicxml_to_wav(musicxml_path, output_wav_path, method='musescore', **kwargs):
    """Convert MusicXML to WAV. Methods: 'musescore', 'simple'."""
    musicxml_path = Path(musicxml_path)
    output_wav_path = Path(output_wav_path)

    if not musicxml_path.exists():
        print(f"MusicXML file not found: {musicxml_path}")
        return False

    output_wav_path.parent.mkdir(parents=True, exist_ok=True)

    if method == 'musescore':
        success = musicxml_to_wav_musescore(musicxml_path, output_wav_path, **kwargs)
        if not success:
            print("MuseScore failed, falling back to simple method")
            success = musicxml_to_wav_simple(musicxml_path, output_wav_path)
        return success

    return musicxml_to_wav_simple(musicxml_path, output_wav_path, **kwargs)
