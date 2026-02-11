import os

"""
Pipeline B: Simple PYIN with Emotion Classification (Before/After)
PYIN-based pitch detection with RAVDESS emotion classifier
Compares emotion before (original audio) and after (MusicXML→WAV reconstruction)
"""
import librosa
import numpy as np
import pretty_midi
from pathlib import Path
import json
from typing import Dict, Tuple, List
import mir_eval
import os
import pandas as pd
import sys
import pickle

# Add src to path for emotion classifier
sys.path.append(str(Path(__file__).parent.parent / 'src'))

# Import music21 for MusicXML/PDF export
from music21 import stream, note, meter, tempo, clef, metadata, converter

# Import MusicXML to WAV converter

import subprocess

def musicxml_to_wav_musescore(musicxml_path, output_wav_path, musescore_exe=r"C:\\Program Files\\MuseScore 4\\bin\\musescore4.exe"):
    """
    Convert MusicXML to WAV using MuseScore's command-line interface.
    Args:
        musicxml_path: Path to input MusicXML file
        output_wav_path: Path to output WAV file
        musescore_exe: Path to MuseScore executable
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        cmd = [musescore_exe, '-o', str(output_wav_path), str(musicxml_path)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print("[ERROR] MuseScore failed with return code:", result.returncode)
            print("[ERROR] MuseScore stderr:\n", result.stderr)
            print("[ERROR] MuseScore stdout:\n", result.stdout)
            return False
        return True
    except Exception as e:
        print(f"[ERROR] Exception during MuseScore conversion: {e}")
        return False

# ============================================================================
# EMOTION CLASSIFICATION
# ============================================================================

class EmotionClassifier:
    """RAVDESS-trained emotion classifier"""
    
    def __init__(self, model_dir):
        model_dir = Path(model_dir)
        
        # Load trained model, scaler, and label encoder
        with open(model_dir / 'emotion_classifier.pkl', 'rb') as f:
            self.model = pickle.load(f)
        with open(model_dir / 'scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
        with open(model_dir / 'label_encoder.pkl', 'rb') as f:
            self.le = pickle.load(f)
    
    def extract_features(self, audio_path, sr=22050, duration=3.0):
        """Extract 339 audio features for emotion classification"""
        try:
            y, sr_actual = librosa.load(audio_path, sr=sr, duration=duration, mono=True)
            
            # Pad if too short
            target_length = int(sr * duration)
            if len(y) < target_length:
                y = np.pad(y, (0, target_length - len(y)), mode='constant')
            
            # MFCC features (40 mean + 40 std = 80 features)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)
            
            # Mel-spectrogram (128 mean + 128 std = 256 features)
            mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            mel_mean = np.mean(mel, axis=1)
            mel_std = np.std(mel, axis=1)
            
            # Spectral features (3 features)
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
            
            # Total: 80 + 256 + 3 = 339 features
            features = np.concatenate([
                mfcc_mean, mfcc_std,
                mel_mean, mel_std,
                [spectral_centroid, spectral_bandwidth, spectral_rolloff]
            ])
            
            return features
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None
    
    def predict(self, audio_path):
        """Predict emotion with top 2 probabilities"""
        features = self.extract_features(audio_path)
        if features is None:
            return None
        
        # Scale and predict
        features_scaled = self.scaler.transform([features])
        proba = self.model.predict_proba(features_scaled)[0]
        
        # Get top 2 emotions
        top2_idx = np.argsort(proba)[-2:][::-1]
        
        return {
            'top_emotion': self.le.classes_[top2_idx[0]],
            'top_confidence': float(proba[top2_idx[0]]),
            'second_emotion': self.le.classes_[top2_idx[1]],
            'second_confidence': float(proba[top2_idx[1]]),
            'top2': [
                {
                    'emotion': self.le.classes_[top2_idx[0]],
                    'confidence': float(proba[top2_idx[0]])
                },
                {
                    'emotion': self.le.classes_[top2_idx[1]],
                    'confidence': float(proba[top2_idx[1]])
                }
            ]
        }


# ============================================================================
# PYIN-BASED VOCAL TO MIDI CONVERTER
# ============================================================================

class VocalToMIDIConverter_PYIN:
    """PYIN-based vocal to MIDI converter"""
    
    def __init__(self, sr=16000, hop_length=512, fmin=65.4, fmax=2093):
        self.sr = sr
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax
    
    def extract_pitch_pyin(self, audio_path):
        """Extract pitch using PYIN algorithm from librosa."""
        y, sr = librosa.load(audio_path, sr=self.sr)
        
        # PYIN pitch detection
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=self.fmin,
            fmax=self.fmax,
            sr=sr,
            hop_length=self.hop_length,
            fill_na=None
        )
        
        times = librosa.times_like(f0, sr=sr, hop_length=self.hop_length)
        voiced_ratio = np.sum(voiced_flag) / len(voiced_flag)
        
        return times, f0, voiced_flag, voiced_probs, y, sr
    
    def detect_notes_from_pitch(self, f0, voiced_flag, times, min_note_duration=0.1):
        """Convert pitch contour to discrete notes with onset/offset detection."""
        notes = []
        current_note = None
        min_frames = int(min_note_duration * self.sr / self.hop_length)
        
        for i, (pitch, is_voiced) in enumerate(zip(f0, voiced_flag)):
            time = times[i]
            
            if is_voiced and not np.isnan(pitch):
                midi_pitch = librosa.hz_to_midi(pitch)
                
                if current_note is None:
                    # Start new note
                    current_note = {
                        'start': time,
                        'pitch_sum': midi_pitch,
                        'pitch_count': 1,
                        'start_frame': i
                    }
                else:
                    # Continue current note
                    current_note['pitch_sum'] += midi_pitch
                    current_note['pitch_count'] += 1
            else:
                # End current note if it exists
                if current_note is not None:
                    duration_frames = i - current_note['start_frame']
                    
                    if duration_frames >= min_frames:
                        avg_pitch = current_note['pitch_sum'] / current_note['pitch_count']
                        notes.append({
                            'start': current_note['start'],
                            'end': time,
                            'pitch': int(round(avg_pitch))
                        })
                    
                    current_note = None
        
        # Handle last note
        if current_note is not None:
            time = times[-1]
            avg_pitch = current_note['pitch_sum'] / current_note['pitch_count']
            notes.append({
                'start': current_note['start'],
                'end': time,
                'pitch': int(round(avg_pitch))
            })
        
        return notes
    
    def convert_to_midi(self, notes, output_path):
        """Convert detected notes to MIDI file."""
        midi = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano
        
        for n in notes:
            midi_note = pretty_midi.Note(
                velocity=100,
                pitch=n['pitch'],
                start=n['start'],
                end=n['end']
            )
            instrument.notes.append(midi_note)
        
        midi.instruments.append(instrument)
        midi.write(output_path)


# ============================================================================
# MUSIC21 EXPORT WITH EMOTIONS
# ============================================================================

def notes_to_musicxml_with_emotion(notes, emotion_result, output_path):
    """Convert notes to MusicXML with emotion labels"""
    
    # Create score
    score = stream.Score()
    
    # Add metadata with emotions
    score.metadata = metadata.Metadata()
    score.metadata.title = "PYIN Vocal Transcription with Emotion"
    
    # Add emotion text at the top
    emotion_text = f"Emotions: {emotion_result['top_emotion'].upper()} ({emotion_result['top_confidence']:.1%}), {emotion_result['second_emotion'].capitalize()} ({emotion_result['second_confidence']:.1%})"
    score.metadata.composer = emotion_text
    
    # Create part
    part = stream.Part()
    part.append(clef.TrebleClef())
    part.append(meter.TimeSignature('4/4'))
    part.append(tempo.MetronomeMark(number=120))
    
    # Add notes
    for n in notes:
        pitch_midi = int(round(n['pitch']))
        duration_sec = n['end'] - n['start']
        duration_ql = max(0.25, round(duration_sec * 2) / 2)  # Quantize to eighth notes
        
        m21_note = note.Note()
        m21_note.pitch.midi = pitch_midi
        m21_note.quarterLength = duration_ql
        
        part.append(m21_note)
    
    score.append(part)
    
    # Export
    score.write('musicxml', fp=str(output_path))
    print(f"✅ Exported MusicXML with emotions to: {output_path}")


# ============================================================================
# MAIN PROCESSING PIPELINE
# ============================================================================

def load_ground_truth(audio_path):
    """Load ground truth from GTSinger JSON file"""
    json_path = Path(audio_path).with_suffix('.json')
    
    if not json_path.exists():
        return None
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract ground truth notes (MIDI numbers, excluding note=0 which is silence)
        gt_notes = []
        gt_emotion = None
        
        # Handle GTSinger JSON format: data is a dict with 'value' key containing list
        entries = data.get('value', []) if isinstance(data, dict) else data
        
        for entry in entries:
            if not isinstance(entry, dict):
                continue
                
            notes = entry.get('note', [])
            for note_midi in notes:
                if note_midi > 0:  # Exclude silence
                    gt_notes.append(note_midi)
            
            # Get emotion label (first non-empty emotion found)
            if not gt_emotion and entry.get('emotion'):
                gt_emotion = entry['emotion']
        
        return {
            'notes': gt_notes,
            'note_count': len(gt_notes),
            'emotion': gt_emotion
        }
    except Exception as e:
        print(f"  ⚠️ Could not load ground truth: {e}")
        return None


def process_audio_file(audio_path, converter, emotion_classifier, output_dir, input_base_dir):
    """Process single audio file with Before/After emotion classification"""
    
    audio_path_obj = Path(audio_path)
    
    print(f"\n{'='*70}")
    print(f"Processing: {audio_path_obj.name}")
    print(f"{'='*70}")
    
    # Create unique filename based on folder structure
    try:
        rel_path = audio_path_obj.relative_to(input_base_dir)
        folder_parts = rel_path.parts[:-1]
        unique_id = "_".join(folder_parts) + "_" + audio_path_obj.stem
        unique_id = unique_id.replace(" ", "_").replace("-", "_")
    except:
        unique_id = audio_path_obj.stem
    
    print(f"📝 Unique ID: {unique_id}")
    
    # Create subdirectories
    output_dir = Path(output_dir)
    musicxml_dir = output_dir / "musicxml"
    wav_after_dir = output_dir / "wav_after"
    emotion_results_dir = output_dir / "emotion_results"
    
    musicxml_dir.mkdir(parents=True, exist_ok=True)
    wav_after_dir.mkdir(parents=True, exist_ok=True)
    emotion_results_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load ground truth
    ground_truth = load_ground_truth(audio_path)
    if ground_truth:
        print(f"📊 Ground Truth: {ground_truth['note_count']} notes, Emotion: {ground_truth['emotion']}")
    
    # 2. BEFORE: Predict emotions from original audio
    print("🎭 [BEFORE] Classifying emotion from original audio...")
    emotion_before = emotion_classifier.predict(audio_path)
    
    if emotion_before:
        print(f"  Top 1: {emotion_before['top_emotion']} ({emotion_before['top_confidence']:.1%})")
        print(f"  Top 2: {emotion_before['second_emotion']} ({emotion_before['second_confidence']:.1%})")
    else:
        print("  ⚠️ Emotion classification failed")
        return None
    
    # 3. Extract pitch with PYIN
    print("🎵 Extracting pitch with PYIN...")
    times, f0, voiced_flag, voiced_probs, y, sr = converter.extract_pitch_pyin(audio_path)
    
    voiced_ratio = np.sum(voiced_flag) / len(voiced_flag)
    print(f"  Voiced frame ratio: {voiced_ratio*100:.1f}%")
    
    # 4. Detect notes
    print("� Detecting notes...")
    detected_notes = converter.detect_notes_from_pitch(f0, voiced_flag, times, min_note_duration=0.1)
    print(f"  Detected {len(detected_notes)} notes")
    
    # 5. Export MusicXML with emotions using unique ID
    musicxml_path = musicxml_dir / f"{unique_id}_emotion.musicxml"
    notes_to_musicxml_with_emotion(detected_notes, emotion_before, musicxml_path)

    
    # 6. AFTER: Convert MusicXML → WAV and classify emotion
    print("🔄 [AFTER] Converting MusicXML to WAV (MuseScore)...")
    wav_after_path = wav_after_dir / f"{unique_id}_after.wav"
    try:
        success = musicxml_to_wav_musescore(str(musicxml_path), str(wav_after_path))
        if success:
            print(f"  ✓ Saved to: {wav_after_path.name}")
        else:
            print(f"  ⚠️ MuseScore conversion failed for {musicxml_path}")
            wav_after_path = None
    except Exception as e:
        print(f"  ⚠️ WAV conversion failed: {e}")
        wav_after_path = None

    emotion_after = None
    if wav_after_path and wav_after_path.exists():
        print("� [AFTER] Classifying emotion from regenerated audio...")
        emotion_after = emotion_classifier.predict(str(wav_after_path))
        
        if emotion_after:
            print(f"  Top 1: {emotion_after['top_emotion']} ({emotion_after['top_confidence']:.1%})")
            print(f"  Top 2: {emotion_after['second_emotion']} ({emotion_after['second_confidence']:.1%})")
        else:
            print("  ⚠️ AFTER emotion classification failed")

    

    # 7. Save emotion comparison
    emotion_comparison = {
        'file': str(audio_path_obj.name),
        'unique_id': unique_id,
        'before': emotion_before,
        'after': emotion_after,
        'emotion_preserved': emotion_before['top_emotion'] == emotion_after['top_emotion'] if emotion_after else None,
        'musicxml_path': str(musicxml_path.relative_to(output_dir)),
        'wav_after_path': str(wav_after_path.relative_to(output_dir)) if wav_after_path else None
    }
    
    emotion_json_path = emotion_results_dir / f"{unique_id}_emotion_comparison.json"
    with open(emotion_json_path, 'w') as f:
        json.dump(emotion_comparison, f, indent=2)
    
    print(f"💾 Saved emotion comparison to: {emotion_json_path.name}")

    # 8. Prepare results in standard pipeline format
    result = {
        'file': str(audio_path_obj.name),
        'notes': len(detected_notes),
        'musicxml': str(musicxml_path.relative_to(output_dir)),
        'wav_after': str(wav_after_path.relative_to(output_dir)) if wav_after_path else None,
        'emotion_before': emotion_before,
        'emotion_after': emotion_after if emotion_after else {},
        'success': True
    }
    
    return result


def process_dataset(input_dir, output_dir, model_dir, max_files=None):
    """Process entire dataset with Before/After emotion comparison"""
    
    # Initialize
    converter = VocalToMIDIConverter_PYIN(sr=16000, hop_length=512)
    emotion_classifier = EmotionClassifier(model_dir)
    
    # Find audio files
    input_path = Path(input_dir)
    audio_files = list(input_path.rglob("*.wav"))
    
    print(f"\n📁 Found {len(audio_files)} audio files")
    
    if max_files and len(audio_files) > max_files:
        import random
        audio_files = random.sample(audio_files, max_files)
        print(f"  Processing {max_files} files...")
    
    # Process each file
    results = []
    failed_count = 0
    
    for i, audio_file in enumerate(audio_files, 1):
        print(f"\n[{i}/{len(audio_files)}]")
        
        try:
            result = process_audio_file(
                str(audio_file),
                converter,
                emotion_classifier,
                output_dir,
                input_path  # Pass base directory for unique IDs
            )
            
            if result:
                results.append(result)
            else:
                failed_count += 1
        
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            failed_count += 1
    
    # Save summary JSON
    if results:
        output_dir_path = Path(output_dir)
        
        # Create comprehensive summary matching standard pipeline format
        summary_data = {
            'pipeline': 'Pipeline B: Simple PYIN',
            'total_files': len(results) + failed_count,
            'successful': len(results),
            'failed': failed_count,
            'output_directories': {
                'musicxml': 'musicxml',
                'wav_after': 'wav_after',
                'emotion_results': 'emotion_results'
            },
            'results': results
        }
        
        summary_json_path = output_dir_path / "lex_pyin_summary.json"
        with open(summary_json_path, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        print(f"\n📊 Pipeline summary saved to: {summary_json_path}")
        
        # Calculate statistics for console output
        successful_before = [r for r in results if r.get('emotion_before')]
        successful_after = [r for r in results if r.get('emotion_after') and r['emotion_after']]
        
        if successful_before:
            emotion_preserved = sum(
                1 for r in results 
                if r.get('emotion_before') and r.get('emotion_after') and r['emotion_after']
                and r['emotion_before']['top_emotion'] == r['emotion_after']['top_emotion']
            )
            
            print(f"\n{'='*70}")
            print("PIPELINE STATISTICS")
            print(f"{'='*70}")
            print(f"Total files processed: {len(results)}")
            print(f"Successful BEFORE emotion: {len(successful_before)}")
            print(f"Successful AFTER emotion: {len(successful_after)}")
            
            if len(successful_after) > 0:
                preservation_rate = (emotion_preserved / len(successful_after)) * 100
                print(f"Emotion preserved: {emotion_preserved}/{len(successful_after)} ({preservation_rate:.1f}%)")
            
            # Emotion distribution BEFORE
            print(f"\n[BEFORE Processing - Emotion Distribution]")
            before_emotions = {}
            for r in successful_before:
                emotion = r['emotion_before']['top_emotion']
                before_emotions[emotion] = before_emotions.get(emotion, 0) + 1
            for emotion, count in sorted(before_emotions.items(), key=lambda x: x[1], reverse=True):
                print(f"  {emotion}: {count}")
            
            # Emotion distribution AFTER
            if successful_after:
                print(f"\n[AFTER Processing - Emotion Distribution]")
                after_emotions = {}
                for r in successful_after:
                    if r['emotion_after']:
                        emotion = r['emotion_after']['top_emotion']
                        after_emotions[emotion] = after_emotions.get(emotion, 0) + 1
                for emotion, count in sorted(after_emotions.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {emotion}: {count}")
            
            # Average note count
            avg_notes = sum(r['notes'] for r in results) / len(results)
            print(f"\n[Transcription Quality]")
            print(f"Average notes detected: {avg_notes:.1f}")
    
    return results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Configuration - using absolute paths for reliability
    BASE_DIR = Path(__file__).parent  # Project root
    INPUT_DIR = BASE_DIR / "vocal-to-score-demo" / "Input" / "GTSinger_sample_50"
    OUTPUT_DIR = BASE_DIR / "output"
    MODEL_DIR = BASE_DIR / "results" / "emotion_model"
    MAX_FILES = None  # Process all 50 files for full comparison
    
    if INPUT_DIR.exists() and MODEL_DIR.exists():
        results = process_dataset(str(INPUT_DIR), str(OUTPUT_DIR), str(MODEL_DIR), max_files=MAX_FILES)
        print(f"\n✅ Processed {len(results)} files successfully!")
    else:
        print(f"❌ Error: Check paths exist:")
        print(f"  Input: {INPUT_DIR} (exists: {INPUT_DIR.exists()})")
        print(f"  Model: {MODEL_DIR} (exists: {MODEL_DIR.exists()})")
