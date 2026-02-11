"""
Pipeline C: PYIN + HMM + Viterbi with Emotion Classification (JSON Roundtrip)
Performs roundtrip via JSON (notes export/import and MIDI synthesis), not MusicXML.
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
import scipy.signal
import random
import sys
import pickle
import subprocess 

# Add src to path for emotion classifier
sys.path.append(str(Path(__file__).parent.parent / 'src'))


# ============================================================================
# EMOTION CLASSIFICATION
# ============================================================================

class EmotionClassifier:
    """RAVDESS-trained emotion classifier"""
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
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
            zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
            features = np.concatenate([
                mfcc_mean, mfcc_std,
                mel_mean, mel_std,
                [spectral_centroid, spectral_rolloff, zero_crossing_rate]
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
            'top1_emotion': self.le.classes_[top2_idx[0]],
            'top1_confidence': float(proba[top2_idx[0]]),
            'top2_emotion': self.le.classes_[top2_idx[1]],
            'top2_confidence': float(proba[top2_idx[1]])
        }

# ============================================================================
# EXPERIMENT PARAMETERS
# ============================================================================
class ExperimentConfig:
    min_note_duration: float = 0.3
    pitch_threshold: float = 0.5
    vibrato_tolerance: float = 0.3
    merge_threshold: float = 0.5
    min_gap: float = 0.08
    median_kernel_size: int = 7
    pyin_fmin: float = 65.4
    pyin_fmax: float = 1047.0
    pyin_frame_length: int = 2048
    pyin_hop_length: int = 256
    self_trans_prob: float = 0.85
    neighbor_prob: float = 0.06
    unvoiced_stay: float = 0.5

# ============================================================================
# PITCH DETECTION & NOTE SEGMENTATION
# ============================================================================
def improved_midi_to_note_events(times, midi_seq, config: ExperimentConfig):
    def round_time(t, step=0.01):
        return np.round(t / step) * step
    notes = []
    if len(midi_seq) == 0:
        return notes
    window_size = 5
    pitch_variance = np.zeros_like(midi_seq, dtype=float)
    for i in range(len(midi_seq)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(midi_seq), i + window_size // 2 + 1)
        window = midi_seq[start_idx:end_idx]
        voiced_window = window[window > 0]
        if len(voiced_window) > 1:
            pitch_variance[i] = np.std(voiced_window)
    current_pitch = midi_seq[0]
    start_time = times[0]
    pitch_accumulator = [current_pitch] if current_pitch > 0 else []
    for i in range(1, len(midi_seq)):
        current_val = midi_seq[i]
        prev_val = midi_seq[i-1]
        local_variance = pitch_variance[i]
        adaptive_threshold = config.pitch_threshold + (
            config.vibrato_tolerance if local_variance > 0.5 else 0
        )
        pitch_diff = abs(current_val - current_pitch)
        is_silence_boundary = (
            (current_val == 0 and prev_val > 0) or 
            (current_val > 0 and prev_val == 0)
        )
        if pitch_diff > adaptive_threshold or is_silence_boundary:
            end_time = times[i]
            duration = end_time - start_time
            if len(pitch_accumulator) > 0:
                stable_pitch = np.median(pitch_accumulator)
                if stable_pitch > 0 and duration >= config.min_note_duration:
                    notes.append({
                        "start": float(round_time(start_time)),
                        "end": float(round_time(end_time)),
                        "pitch": float(round(stable_pitch))
                    })
            current_pitch = current_val
            start_time = times[i]
            pitch_accumulator = [current_val] if current_val > 0 else []
        else:
            if current_val > 0:
                pitch_accumulator.append(current_val)
    if len(times) > 1:
        frame_duration = times[-1] - times[-2]
        end_time = times[-1] + frame_duration
    else:
        end_time = times[-1]
    duration = end_time - start_time
    if len(pitch_accumulator) > 0:
        stable_pitch = np.median(pitch_accumulator)
        if stable_pitch > 0 and duration >= config.min_note_duration:
            notes.append({
                "start": float(round_time(start_time)),
                "end": float(round_time(end_time)),
                "pitch": float(round(stable_pitch))
            })
    return notes

def post_process_notes(notes, config: ExperimentConfig):
    if len(notes) <= 1:
        return notes
    merged = []
    current = notes[0].copy()
    for next_note in notes[1:]:
        pitch_diff = abs(next_note['pitch'] - current['pitch'])
        time_gap = next_note['start'] - current['end']
        if pitch_diff < config.merge_threshold and time_gap < config.min_gap:
            current['end'] = next_note['end']
            current['pitch'] = (current['pitch'] + next_note['pitch']) / 2
        else:
            merged.append(current)
            current = next_note.copy()
    merged.append(current)
    return merged

# ============================================================================
# VITERBI HMM DECODER
# ============================================================================
class VocalToMIDIConverter_Viterbi:
    def __init__(self, config: ExperimentConfig):
        self.config = config
    def viterbi_decode(self, f0, voiced_probs):
        n_frames = len(f0)
        midi_pitches = []
        for freq in f0:
            if freq is None or np.isnan(freq) or freq <= 0:
                midi_pitches.append(0)
            else:
                midi_pitches.append(librosa.hz_to_midi(freq))
        all_midi = [m for m in midi_pitches if m > 0]
        if len(all_midi) == 0:
            return np.zeros(n_frames)
        min_midi = int(np.min(all_midi))
        max_midi = int(np.max(all_midi))
        states = list(range(min_midi, max_midi + 1)) + [0]
        n_states = len(states)
        state_to_idx = {s: i for i, s in enumerate(states)}
        trans_prob = np.full((n_states, n_states), 1e-10)
        for i, state in enumerate(states):
            if state == 0:
                trans_prob[i, i] = self.config.unvoiced_stay
                trans_prob[i, :] = (1 - self.config.unvoiced_stay) / (n_states - 1)
            else:
                trans_prob[i, i] = self.config.self_trans_prob
                for j, other_state in enumerate(states):
                    if other_state == 0:
                        continue
                    diff = abs(other_state - state)
                    if diff == 1:
                        trans_prob[i, j] = self.config.neighbor_prob
                row_sum = trans_prob[i, :].sum()
                if row_sum > 0:
                    trans_prob[i, :] /= row_sum
        viterbi_path = np.zeros(n_frames, dtype=int)
        delta = np.zeros((n_frames, n_states))
        psi = np.zeros((n_frames, n_states), dtype=int)
        for s_idx, state in enumerate(states):
            if state == 0:
                delta[0, s_idx] = 1 - voiced_probs[0] if not np.isnan(voiced_probs[0]) else 0.5
            else:
                obs_midi = midi_pitches[0]
                if obs_midi > 0:
                    pitch_diff = abs(state - obs_midi)
                    emission_prob = np.exp(-pitch_diff / 2.0)
                else:
                    emission_prob = 0.01
                delta[0, s_idx] = (voiced_probs[0] if not np.isnan(voiced_probs[0]) else 0.5) * emission_prob
        for t in range(1, n_frames):
            for s_idx, state in enumerate(states):
                if state == 0:
                    emission_prob = 1 - voiced_probs[t] if not np.isnan(voiced_probs[t]) else 0.5
                else:
                    obs_midi = midi_pitches[t]
                    if obs_midi > 0:
                        pitch_diff = abs(state - obs_midi)
                        emission_prob = np.exp(-pitch_diff / 2.0)
                    else:
                        emission_prob = 0.01
                trans_times_delta = delta[t-1, :] * trans_prob[:, s_idx]
                psi[t, s_idx] = np.argmax(trans_times_delta)
                delta[t, s_idx] = np.max(trans_times_delta) * emission_prob
        viterbi_path[n_frames - 1] = np.argmax(delta[n_frames - 1, :])
        for t in range(n_frames - 2, -1, -1):
            viterbi_path[t] = psi[t + 1, viterbi_path[t + 1]]
        smoothed_midi = np.array([states[idx] for idx in viterbi_path])
        return smoothed_midi

# --- JSON Export/Import Functions ---
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
            pitch=int(round(n['pitch'])),
            start=n['start'],
            end=n['end']
        )
        instrument.notes.append(midi_note)
    midi.instruments.append(instrument)
    midi.write(output_path)

def midi_to_wav(midi_path, wav_path):
    musescore_exe = r"C:\Program Files\MuseScore 4\bin\musescore4.exe"
    try:
        cmd = [musescore_exe, "-o", str(wav_path), str(midi_path)]
        print("[DEBUG] Running:", cmd)

        # shell=False is important here
        result = subprocess.run(cmd, shell=False, capture_output=True, text=True)

        if result.returncode != 0:
            print("[ERROR] MuseScore failed with return code:", result.returncode)
            print("[ERROR] stderr:\n", result.stderr)
            print("[ERROR] stdout:\n", result.stdout)
            return False

        return True
    except Exception as e:
        print(f"[ERROR] Exception during MuseScore conversion: {e}")
        return False

# --- Main Processing Function (JSON roundtrip only) ---
def process_audio_file_json(audio_path, config, emotion_classifier, output_dir, input_base_dir):
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
        print(f"  Top 1: {emotion_before['top1_emotion']} ({emotion_before['top1_confidence']:.1%})")
        print(f"  Top 2: {emotion_before['top2_emotion']} ({emotion_before['top2_confidence']:.1%})")
    else:
        print("  ⚠️ Emotion classification failed")
        return None
    # 2. Load audio and extract pitch
    print("🎵 Extracting pitch with PYIN...")
    y, sr = librosa.load(audio_path, sr=22050)
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, fmin=config.pyin_fmin, fmax=config.pyin_fmax, sr=sr,
        frame_length=config.pyin_frame_length, hop_length=config.pyin_hop_length
    )
    times = librosa.times_like(f0, sr=sr, hop_length=config.pyin_hop_length)
    # 3. Viterbi decoding
    print("🔧 Applying Viterbi HMM smoothing...")
    converter = VocalToMIDIConverter_Viterbi(config)
    smoothed_midi = converter.viterbi_decode(f0, voiced_probs)
    smoothed_midi = scipy.signal.medfilt(smoothed_midi, kernel_size=config.median_kernel_size)
    # 4. Detect notes
    print("🎼 Detecting notes...")
    detected_notes = improved_midi_to_note_events(times, smoothed_midi, config)
    detected_notes = post_process_notes(detected_notes, config)
    print(f"  Detected {len(detected_notes)} notes")
    # 5. Export notes to JSON
    json_notes_path = json_dir / f"{unique_id}_notes.json"
    notes_to_json(detected_notes, json_notes_path)
    # 6. Convert JSON back to notes, synthesize WAV, and compare emotions
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
            print(f"  Top 1: {emotion_after_json['top1_emotion']} ({emotion_after_json['top1_confidence']:.1%})")
            print(f"  Top 2: {emotion_after_json['top2_emotion']} ({emotion_after_json['top2_confidence']:.1%})")
        else:
            print("  ⚠️ AFTER-JSON emotion classification failed")
    # 7. Save emotion comparison (JSON roundtrip only)
    emotion_comparison = {
        'file': str(audio_path_obj.name),
        'unique_id': unique_id,
        'before': emotion_before,
        'after_json': emotion_after_json,
        'emotion_preserved_json': emotion_before['top1_emotion'] == emotion_after_json['top1_emotion'] if emotion_after_json else None,
        'json_notes_path': str(json_notes_path.relative_to(output_dir)),
        'midi_json_path': str(midi_json_path.relative_to(output_dir)),
        'wav_after_json_path': str(wav_after_json_path.relative_to(output_dir)) if wav_after_json_path else None
    }
    emotion_json_path = emotion_results_dir / f"{unique_id}_emotion_comparison.json"
    with open(emotion_json_path, 'w') as f:
        json.dump(emotion_comparison, f, indent=2)
    print(f"💾 Saved emotion comparison to: {emotion_json_path.name}")
    # 8. Prepare results in standard pipeline format
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
    config = ExperimentConfig()
    emotion_classifier = EmotionClassifier(model_dir)
    input_path = Path(input_dir)
    audio_files = list(input_path.rglob("*.wav"))
    print(f"\n📁 Found {len(audio_files)} audio files")
    if max_files and len(audio_files) > max_files:
        audio_files = random.sample(audio_files, max_files)
        print(f"  Processing {max_files} files...")
    results = []
    failed_count = 0
    for i, audio_file in enumerate(audio_files, 1):
        print(f"\n[{i}/{len(audio_files)}]")
        try:
            result = process_audio_file_json(
                str(audio_file),
                config,
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
            'pipeline': 'Pipeline C: Viterbi PYIN JSON Roundtrip',
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
        summary_json_path = output_dir_path / "wx_pyin_json_summary.json"
        with open(summary_json_path, 'w') as f:
            json.dump(summary_data, f, indent=2)
        print(f"\n📊 Pipeline summary saved to: {summary_json_path}")
        successful_before = [r for r in results if r.get('emotion_before')]
        successful_after_json = [r for r in results if r.get('emotion_after_json') and r['emotion_after_json']]
        if successful_before:
            emotion_preserved = sum(
                1 for r in results 
                if r.get('emotion_before') and r.get('emotion_after_json') and r['emotion_after_json']
                and r['emotion_before']['top1_emotion'] == r['emotion_after_json']['top1_emotion']
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
                emotion = r['emotion_before']['top1_emotion']
                before_emotions[emotion] = before_emotions.get(emotion, 0) + 1
            for emotion, count in sorted(before_emotions.items(), key=lambda x: x[1], reverse=True):
                print(f"  {emotion}: {count}")
            if successful_after_json:
                print(f"\n[AFTER-JSON Processing - Emotion Distribution]")
                after_json_emotions = {}
                for r in successful_after_json:
                    if r['emotion_after_json']:
                        emotion = r['emotion_after_json']['top1_emotion']
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
