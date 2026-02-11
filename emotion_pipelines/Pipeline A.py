
import os
import librosa
import numpy as np
import pretty_midi
from pathlib import Path
import json
from typing import Dict, Tuple, List
import mir_eval
import pandas as pd
import sys
import pickle

# Add src to path for emotion classifier
sys.path.append(str(Path(__file__).parent.parent / 'src'))

import subprocess

class EmotionClassifier:
    def __init__(self, model_dir):
        model_dir = Path(model_dir)
        with open(model_dir / 'emotion_classifier.pkl', 'rb') as f:
            self.model = pickle.load(f)
        with open(model_dir / 'scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
        with open(model_dir / 'label_encoder.pkl', 'rb') as f:
            self.le = pickle.load(f)
    def extract_features(self, audio_path, sr=22050, duration=3.0):
        try:
            y, sr_actual = librosa.load(audio_path, sr=sr, duration=duration, mono=True)
            target_length = int(sr * duration)
            if len(y) < target_length:
                y = np.pad(y, (0, target_length - len(y)), mode='constant')
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)
            mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            mel_mean = np.mean(mel, axis=1)
            mel_std = np.std(mel, axis=1)
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
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
        features = self.extract_features(audio_path)
        if features is None:
            return None
        features_scaled = self.scaler.transform([features])
        proba = self.model.predict_proba(features_scaled)[0]
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

class VocalToMIDIConverter_PYIN:
    def __init__(self, sr=16000, hop_length=512, fmin=65.4, fmax=2093):
        self.sr = sr
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax
    def extract_pitch_pyin(self, audio_path):
        y, sr = librosa.load(audio_path, sr=self.sr)
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
        notes = []
        current_note = None
        min_frames = int(min_note_duration * self.sr / self.hop_length)
        for i, (pitch, is_voiced) in enumerate(zip(f0, voiced_flag)):
            time = times[i]
            if is_voiced and not np.isnan(pitch):
                midi_pitch = librosa.hz_to_midi(pitch)
                if current_note is None:
                    current_note = {
                        'start': time,
                        'pitch_sum': midi_pitch,
                        'pitch_count': 1,
                        'start_frame': i
                    }
                else:
                    current_note['pitch_sum'] += midi_pitch
                    current_note['pitch_count'] += 1
            else:
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
        midi = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=0)
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

def notes_to_json(notes, output_path):
    with open(output_path, 'w') as f:
        json.dump(notes, f, indent=2)
    print(f"✅ Exported notes to JSON: {output_path}")

def json_to_notes(json_path):
    with open(json_path, 'r') as f:
        notes = json.load(f)
    return notes

def notes_to_midi_from_json(notes, output_path):
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)
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

def midi_to_wav(midi_path, wav_path, soundfont_path=None):
    # Use fluidsynth or MuseScore for MIDI to WAV conversion
    # Here, we use MuseScore for consistency
    musescore_exe = r"C:\\Program Files\\MuseScore 4\\bin\\musescore4.exe"
    try:
        cmd = [musescore_exe, '-o', str(wav_path), str(midi_path)]
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

def process_audio_file_json(audio_path, converter, emotion_classifier, output_dir, input_base_dir):
    audio_path_obj = Path(audio_path)
    print(f"\n{'='*70}")
    print(f"Processing: {audio_path_obj.name}")
    print(f"{'='*70}")
    try:
        rel_path = audio_path_obj.relative_to(input_base_dir)
        folder_parts = rel_path.parts[:-1]
        unique_id = "_".join(folder_parts) + "_" + audio_path_obj.stem
        unique_id = unique_id.replace(" ", "_").replace("-", "_")
    except:
        unique_id = audio_path_obj.stem
    print(f"📝 Unique ID: {unique_id}")
    output_dir = Path(output_dir)
    json_dir = output_dir / "json_notes"
    wav_after_dir = output_dir / "wav_after_json"
    emotion_results_dir = output_dir / "emotion_results_json"
    json_dir.mkdir(parents=True, exist_ok=True)
    wav_after_dir.mkdir(parents=True, exist_ok=True)
    emotion_results_dir.mkdir(parents=True, exist_ok=True)
    # 1. BEFORE: Predict emotions from original audio
    print("🎭 [BEFORE] Classifying emotion from original audio...")
    emotion_before = emotion_classifier.predict(audio_path)
    if emotion_before:
        print(f"  Top 1: {emotion_before['top_emotion']} ({emotion_before['top_confidence']:.1%})")
        print(f"  Top 2: {emotion_before['second_emotion']} ({emotion_before['second_confidence']:.1%})")
    else:
        print("  ⚠️ Emotion classification failed")
        return None
    # 2. Extract pitch with PYIN
    print("🎵 Extracting pitch with PYIN...")
    times, f0, voiced_flag, voiced_probs, y, sr = converter.extract_pitch_pyin(audio_path)
    voiced_ratio = np.sum(voiced_flag) / len(voiced_flag)
    print(f"  Voiced frame ratio: {voiced_ratio*100:.1f}%")
    # 3. Detect notes
    print("� Detecting notes...")
    detected_notes = converter.detect_notes_from_pitch(f0, voiced_flag, times, min_note_duration=0.1)
    print(f"  Detected {len(detected_notes)} notes")
    # 4. Export notes to JSON
    json_notes_path = json_dir / f"{unique_id}_notes.json"
    notes_to_json(detected_notes, json_notes_path)
    # 5. Convert JSON back to notes, synthesize WAV, and compare emotions
    print("🔄 [AFTER-JSON] Converting JSON notes back to MIDI and WAV...")
    notes_from_json = json_to_notes(json_notes_path)
    midi_json_path = json_dir / f"{unique_id}_fromjson.mid"
    notes_to_midi_from_json(notes_from_json, midi_json_path)
    wav_after_json_path = wav_after_dir / f"{unique_id}_after_json.wav"
    try:
        success_json = midi_to_wav(str(midi_json_path), str(wav_after_json_path))
        if success_json:
            print(f"  ✓ Saved to: {wav_after_json_path.name}")
        else:
            print(f"  ⚠️ MuseScore conversion failed for {midi_json_path}")
            wav_after_json_path = None
    except Exception as e:
        print(f"  ⚠️ WAV conversion failed: {e}")
        wav_after_json_path = None
    emotion_after_json = None
    if wav_after_json_path and wav_after_json_path.exists():
        print("� [AFTER-JSON] Classifying emotion from JSON-reconstructed audio...")
        emotion_after_json = emotion_classifier.predict(str(wav_after_json_path))
        if emotion_after_json:
            print(f"  Top 1: {emotion_after_json['top_emotion']} ({emotion_after_json['top_confidence']:.1%})")
            print(f"  Top 2: {emotion_after_json['second_emotion']} ({emotion_after_json['second_confidence']:.1%})")
        else:
            print("  ⚠️ AFTER-JSON emotion classification failed")
    # 6. Save emotion comparison (JSON roundtrip only)
    emotion_comparison = {
        'file': str(audio_path_obj.name),
        'unique_id': unique_id,
        'before': emotion_before,
        'after_json': emotion_after_json,
        'emotion_preserved_json': emotion_before['top_emotion'] == emotion_after_json['top_emotion'] if emotion_after_json else None,
        'json_notes_path': str(json_notes_path.relative_to(output_dir)),
        'midi_json_path': str(midi_json_path.relative_to(output_dir)),
        'wav_after_json_path': str(wav_after_json_path.relative_to(output_dir)) if wav_after_json_path else None
    }
    emotion_json_path = emotion_results_dir / f"{unique_id}_emotion_comparison.json"
    with open(emotion_json_path, 'w') as f:
        json.dump(emotion_comparison, f, indent=2)
    print(f"💾 Saved emotion comparison to: {emotion_json_path.name}")
    # 7. Prepare results in standard pipeline format
    result = {
        'file': str(audio_path_obj.name),
        'notes': len(detected_notes),
        'json_notes': str(json_notes_path.relative_to(output_dir)),
        'midi_json': str(midi_json_path.relative_to(output_dir)),
        'wav_after_json': str(wav_after_json_path.relative_to(output_dir)) if wav_after_json_path else None,
        'emotion_before': emotion_before,
        'emotion_after_json': emotion_after_json if emotion_after_json else {},
        'success': True
    }
    return result

def process_dataset_json(input_dir, output_dir, model_dir, max_files=None):
    converter = VocalToMIDIConverter_PYIN(sr=16000, hop_length=512)
    emotion_classifier = EmotionClassifier(model_dir)
    input_path = Path(input_dir)
    audio_files = list(input_path.rglob("*.wav"))
    print(f"\n📁 Found {len(audio_files)} audio files")
    if max_files and len(audio_files) > max_files:
        import random
        audio_files = random.sample(audio_files, max_files)
        print(f"  Processing {max_files} files...")
    results = []
    failed_count = 0
    for i, audio_file in enumerate(audio_files, 1):
        print(f"\n[{i}/{len(audio_files)}]")
        try:
            result = process_audio_file_json(
                str(audio_file),
                converter,
                emotion_classifier,
                output_dir,
                input_path
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
    if results:
        output_dir_path = Path(output_dir)
        summary_data = {
            'pipeline': 'Pipeline A: PYIN JSON Roundtrip',
            'total_files': len(results) + failed_count,
            'successful': len(results),
            'failed': failed_count,
            'output_directories': {
                'json_notes': 'json_notes',
                'wav_after_json': 'wav_after_json',
                'emotion_results_json': 'emotion_results_json'
            },
            'results': results
        }
        summary_json_path = output_dir_path / "lex_pyin_json_summary.json"
        with open(summary_json_path, 'w') as f:
            json.dump(summary_data, f, indent=2)
        print(f"\n📊 Pipeline summary saved to: {summary_json_path}")
        successful_before = [r for r in results if r.get('emotion_before')]
        successful_after_json = [r for r in results if r.get('emotion_after_json') and r['emotion_after_json']]
        if successful_before:
            emotion_preserved = sum(
                1 for r in results 
                if r.get('emotion_before') and r.get('emotion_after_json') and r['emotion_after_json']
                and r['emotion_before']['top_emotion'] == r['emotion_after_json']['top_emotion']
            )
            print(f"\n{'='*70}")
            print("PIPELINE STATISTICS (JSON Roundtrip)")
            print(f"{'='*70}")
            print(f"Total files processed: {len(results)}")
            print(f"Successful BEFORE emotion: {len(successful_before)}")
            print(f"Successful AFTER-JSON emotion: {len(successful_after_json)}")
            if len(successful_after_json) > 0:
                preservation_rate = (emotion_preserved / len(successful_after_json)) * 100
                print(f"Emotion preserved: {emotion_preserved}/{len(successful_after_json)} ({preservation_rate:.1f}%)")
            print(f"\n[BEFORE Processing - Emotion Distribution]")
            before_emotions = {}
            for r in successful_before:
                emotion = r['emotion_before']['top_emotion']
                before_emotions[emotion] = before_emotions.get(emotion, 0) + 1
            for emotion, count in sorted(before_emotions.items(), key=lambda x: x[1], reverse=True):
                print(f"  {emotion}: {count}")
            if successful_after_json:
                print(f"\n[AFTER-JSON Processing - Emotion Distribution]")
                after_json_emotions = {}
                for r in successful_after_json:
                    if r['emotion_after_json']:
                        emotion = r['emotion_after_json']['top_emotion']
                        after_json_emotions[emotion] = after_json_emotions.get(emotion, 0) + 1
                for emotion, count in sorted(after_json_emotions.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {emotion}: {count}")
            avg_notes = sum(r['notes'] for r in results) / len(results)
            print(f"\n[Transcription Quality]")
            print(f"Average notes detected: {avg_notes:.1f}")
    return results

if __name__ == "__main__":
    BASE_DIR = Path(__file__).parent
    INPUT_DIR = BASE_DIR / "vocal-to-score-demo" / "Input" / "GTSinger_sample_50"
    OUTPUT_DIR = BASE_DIR / "output"
    MODEL_DIR = BASE_DIR / "results" / "emotion_model"
    MAX_FILES = None
    if INPUT_DIR.exists() and MODEL_DIR.exists():
        results = process_dataset_json(str(INPUT_DIR), str(OUTPUT_DIR), str(MODEL_DIR), max_files=MAX_FILES)
        print(f"\n✅ Processed {len(results)} files successfully!")
    else:
        print(f"❌ Error: Check paths exist:")
        print(f"  Input: {INPUT_DIR} (exists: {INPUT_DIR.exists()})")
        print(f"  Model: {MODEL_DIR} (exists: {MODEL_DIR.exists()})")
