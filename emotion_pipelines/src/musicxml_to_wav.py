"""
MusicXML to WAV Converter
Converts MusicXML files to WAV audio for emotion re-classification
"""


import os
# Configure FluidSynth path - update FLUIDSYNTH_DIR to match your FluidSynth installation
FLUIDSYNTH_DIR = os.environ.get("FLUIDSYNTH_DIR", "")
if FLUIDSYNTH_DIR and os.path.isdir(FLUIDSYNTH_DIR):
    os.add_dll_directory(FLUIDSYNTH_DIR)
from pathlib import Path
from music21 import converter, midi as m21midi
import subprocess
import tempfile


def musicxml_to_wav_fluidsynth(musicxml_path, output_wav_path, soundfont_path=None):
    """
    Convert MusicXML to WAV using FluidSynth
    
    Args:
        musicxml_path: Path to input MusicXML file
        output_wav_path: Path to output WAV file
        soundfont_path: Path to soundfont (.sf2) file
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Load MusicXML and convert to MIDI
        score = converter.parse(str(musicxml_path))
        
        # Create temporary MIDI file
        with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as tmp_midi:
            midi_path = tmp_midi.name
            score.write('midi', fp=midi_path)
        
        # Use default soundfont if not provided
        if soundfont_path is None:
            print("[ERROR] No soundfont_path provided. Please specify a valid .sf2 file.")
            return False
        
        # Convert MIDI to WAV using FluidSynth
        if soundfont_path and os.path.exists(soundfont_path):
            fluidsynth_bin = os.environ.get("FLUIDSYNTH_BIN", "fluidsynth")
            cmd = [
                fluidsynth_bin,
                '-ni',  # No interactive mode
                soundfont_path,
                midi_path,
                '-F', str(output_wav_path),
                '-r', '22050'  # Match our emotion classifier sample rate
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print("[ERROR] FluidSynth failed with return code:", result.returncode)
                print("[ERROR] FluidSynth stderr:\n", result.stderr)
                print("[ERROR] FluidSynth stdout:\n", result.stdout)
            os.remove(midi_path)
            return result.returncode == 0
        else:
            print(f"[ERROR] SoundFont not found or invalid: {soundfont_path}")
            os.remove(midi_path)
            return False
            
    except Exception as e:
        print(f"Error converting MusicXML to WAV: {e}")
        return False


def musicxml_to_wav_simple(musicxml_path, output_wav_path, sample_rate=22050):
    """
    Simple MusicXML to WAV converter using music21's MIDI export
    Creates a basic sine wave representation of the notes
    
    This is a fallback method that doesn't require external dependencies
    """
    try:
        import numpy as np
        import librosa
        import soundfile as sf
        
        # Load MusicXML
        score = converter.parse(str(musicxml_path))
        
        # Get all notes from the score
        notes_list = []
        for element in score.flatten().notesAndRests:
            if hasattr(element, 'pitch'):  # It's a note (not a rest)
                notes_list.append({
                    'pitch': element.pitch.midi,
                    'start': element.offset,
                    'duration': element.quarterLength,
                    'velocity': 80
                })
        
        if not notes_list:
            return False
        
        # Calculate total duration
        max_end = max(n['start'] + n['duration'] for n in notes_list)
        
        # Convert quarter lengths to seconds (assume 120 BPM)
        bpm = 120
        quarter_duration = 60.0 / bpm  # seconds per quarter note
        
        total_duration = max_end * quarter_duration
        total_samples = int(total_duration * sample_rate)
        
        # Create audio signal
        audio = np.zeros(total_samples)
        
        for note_info in notes_list:
            # Convert note to frequency
            freq = librosa.midi_to_hz(note_info['pitch'])
            
            # Calculate timing
            start_sec = note_info['start'] * quarter_duration
            duration_sec = note_info['duration'] * quarter_duration
            
            start_sample = int(start_sec * sample_rate)
            duration_samples = int(duration_sec * sample_rate)
            end_sample = start_sample + duration_samples
            
            if end_sample > total_samples:
                end_sample = total_samples
            
            # Generate sine wave
            t = np.arange(duration_samples) / sample_rate
            sine_wave = 0.3 * np.sin(2 * np.pi * freq * t)
            
            # Apply envelope (attack and decay)
            envelope = np.ones_like(sine_wave)
            attack_samples = min(100, len(sine_wave) // 4)
            decay_samples = min(100, len(sine_wave) // 4)
            
            if attack_samples > 0:
                envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
            if decay_samples > 0:
                envelope[-decay_samples:] = np.linspace(1, 0, decay_samples)
            
            sine_wave *= envelope
            
            # Add to audio buffer
            actual_samples = end_sample - start_sample
            audio[start_sample:end_sample] += sine_wave[:actual_samples]
        
        # Normalize
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.9
        
        # Save to WAV
        sf.write(str(output_wav_path), audio, sample_rate)
        
        return True
        
    except Exception as e:
        print(f"Error in simple MusicXML to WAV conversion: {e}")
        return False


def convert_musicxml_to_wav(musicxml_path, output_wav_path, method='simple', **kwargs):
    """
    Convert MusicXML to WAV audio file
    
    Args:
        musicxml_path: Path to input MusicXML file
        output_wav_path: Path to output WAV file
        method: 'simple' (sine waves) or 'fluidsynth' (realistic)
        **kwargs: Additional arguments passed to specific converter
    
    Returns:
        bool: True if successful, False otherwise
    """
    musicxml_path = Path(musicxml_path)
    output_wav_path = Path(output_wav_path)
    
    if not musicxml_path.exists():
        print(f"MusicXML file not found: {musicxml_path}")
        return False
    
    # Create output directory
    output_wav_path.parent.mkdir(parents=True, exist_ok=True)
    
    if method == 'fluidsynth':
        success = musicxml_to_wav_fluidsynth(musicxml_path, output_wav_path, **kwargs)
        if not success:
            print("FluidSynth failed, falling back to simple method")
            method = 'simple'
    
    if method == 'simple':
        success = musicxml_to_wav_simple(musicxml_path, output_wav_path, **kwargs)
    
    return success


if __name__ == "__main__":
    # Test the converter
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python musicxml_to_wav.py <input.musicxml> <output.wav> [method]")
        print("Methods: simple (default), fluidsynth")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    method = sys.argv[3] if len(sys.argv) > 3 else 'simple'
    
    success = convert_musicxml_to_wav(input_file, output_file, method=method)
    
    if success:
        print(f"✓ Successfully converted to {output_file}")
    else:
        print(f"✗ Failed to convert {input_file}")
