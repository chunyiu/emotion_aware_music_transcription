"""
Consolidated MusicXML/MIDI to WAV conversion.
Supports MuseScore (platform-aware), FluidSynth (future), and simple sine-wave fallback.
"""

import subprocess
import tempfile
import os
import platform
from pathlib import Path


def get_musescore_executable():
    """
    Returns the path to the MuseScore executable depending on the operating system.
    Returns None if no suitable path is found.
    """
    system = platform.system()

    if system == "Windows":
        path = r"C:\Program Files\MuseScore 3\bin\musescore3.exe"
        if Path(path).exists():
            return path
        alt = r"C:\Program Files (x86)\MuseScore 3\bin\musescore3.exe"
        if Path(alt).exists():
            return alt

    elif system == "Darwin": 
        path = "/Applications/MuseScore 3.app/Contents/MacOS/mscore"
        if Path(path).exists():
            return path

        brew_path = "/opt/homebrew/bin/mscore"
        if Path(brew_path).exists():
            return brew_path

    elif system == "Linux":
        return "mscore"

    return None


def musicxml_to_wav_musescore(musicxml_path, output_wav_path):
    """Convert MusicXML to WAV using MuseScore's command-line interface."""
    musescore_exe = get_musescore_executable()

    if musescore_exe is None:
        print("[MuseScore] Executable not found for this platform.")
        return False

    print(f"[MuseScore] Using executable: {musescore_exe}")

    try:
        cmd = [musescore_exe, '-o', str(output_wav_path), str(musicxml_path)]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120  # prevent hanging forever
        )

        if result.returncode != 0:
            print(f"[ERROR] MuseScore failed (rc={result.returncode}):")
            print(result.stderr[:400])
            return False

        if Path(output_wav_path).exists():
            print(f"[MuseScore] Successfully created: {output_wav_path}")
            return True
        else:
            print("[MuseScore] Command succeeded but output file missing.")
            return False

    except subprocess.TimeoutExpired:
        print("[MuseScore] Conversion timed out after 120 seconds.")
        return False
    except FileNotFoundError:
        print(f"[MuseScore] Executable not found: {musescore_exe}")
        return False
    except Exception as e:
        print(f"[ERROR] MuseScore conversion exception: {e}")
        return False


def musicxml_to_wav_simple(musicxml_path, output_wav_path, sample_rate=22050):
    """Simple MusicXML to WAV converter using sine waves (no external deps)."""
    try:
        import numpy as np
        import librosa
        import soundfile as sf
        from music21 import converter

        print("[Simple converter] Parsing MusicXML...")
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
            print("[Simple converter] No notes found in score.")
            return False

        bpm = 120  # hardcoded for now – could be extracted from score
        quarter_duration = 60.0 / bpm
        max_end = max(n['start'] + n['duration'] for n in notes_list)
        total_samples = int(max_end * quarter_duration * sample_rate)
        audio = np.zeros(total_samples)

        print(f"[Simple converter] Rendering {len(notes_list)} notes...")
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
        print(f"[Simple converter] Wrote WAV: {output_wav_path}")
        return True

    except ImportError as e:
        print(f"[Simple converter] Missing library: {e}")
        return False
    except Exception as e:
        print(f"[Simple converter] Error: {e}")
        return False


def convert_musicxml_to_wav(musicxml_path, output_wav_path, prefer_musescore=True):
    """
    Convert MusicXML to WAV.
    Tries MuseScore first (if available), then falls back to simple sine-wave method.
    """
    musicxml_path = Path(musicxml_path)
    output_wav_path = Path(output_wav_path)

    if not musicxml_path.exists():
        print(f"MusicXML file not found: {musicxml_path}")
        return False

    output_wav_path.parent.mkdir(parents=True, exist_ok=True)

    success = False

    if prefer_musescore:
        print("[Converter] Attempting MuseScore conversion...")
        success = musicxml_to_wav_musescore(musicxml_path, output_wav_path)

    if not success:
        print("[Converter] MuseScore failed or not available -> falling back to simple sine-wave method")
        success = musicxml_to_wav_simple(musicxml_path, output_wav_path)

    return success