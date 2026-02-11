"""
Pipeline H: TorchCrepe with Emotion Classification (JSON Roundtrip)
Performs roundtrip via JSON (notes export/import and MIDI synthesis), not MusicXML.
"""
import os
import numpy as np
import librosa
import torch
import torchcrepe
from scipy.signal import medfilt
import json
from pathlib import Path
from typing import Dict, List
import pandas as pd
import pickle
import sys
import subprocess

# Add src to path for emotion classifier
sys.path.append(str(Path(__file__).parent.parent / 'src'))

# --- EmotionClassifier class is identical to torchcrepe_emotion.py ---
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
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
            mfccs_mean = np.mean(mfccs, axis=1)
            mfccs_std = np.std(mfccs, axis=1)
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            mel_spec_mean = np.mean(mel_spec, axis=1)
            mel_spec_std = np.std(mel_spec, axis=1)
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
            zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
            features = np.concatenate([
                mfccs_mean, mfccs_std, mel_spec_mean, mel_spec_std,
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
        features = features.reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        proba = self.model.predict_proba(features_scaled)[0]
        top_2_idx = np.argsort(proba)[-2:][::-1]
        emotions = self.le.inverse_transform(top_2_idx)
        confidences = proba[top_2_idx]
        return {
            'top1_emotion': emotions[0],
            'top1_confidence': confidences[0],
            'top2_emotion': emotions[1],
            'top2_confidence': confidences[1]
        }

# --- TorchCrepe pitch extraction and note segmentation functions are identical ---
FRAME_RATE = 100
MEDIAN_FILTER_SIZE = 5
FMIN_HZ, FMAX_HZ = 50.0, 1100.0
CONF_THRESHOLD = 0.45
MIN_NOTE_SEC = 0.15
GAP_JOIN_SEC = 0.20
CHUNK_SECONDS = 10
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def trim_and_normalize(y):
    y, _ = librosa.effects.trim(y, top_db=25)
    y = y / (np.max(np.abs(y)) + 1e-9)
    return y

def smooth_pitch(freqs, window=MEDIAN_FILTER_SIZE):
    if window > 1:
        return medfilt(freqs, kernel_size=window)
    return freqs

def hz_to_midi(hz):
    if hz <= 0:
        return None
    return 69 + 12 * np.log2(hz / 440.0)

def extract_f0_conf(y, sr, model='full'):
    hop_length = int(sr / FRAME_RATE)
    audio = torch.tensor(y, dtype=torch.float32).unsqueeze(0).to(device)
    max_chunk = int(sr * CHUNK_SECONDS)
    T = audio.shape[1]
    num_chunks = max(1, int(np.ceil(T / max_chunk)))
    all_f0, all_conf = [], []
    for i in range(num_chunks):
        start = i * max_chunk
        end = min((i + 1) * max_chunk, T)
        chunk_audio = audio[:, start:end]
        f0_chunk, conf_chunk = torchcrepe.predict(
            chunk_audio,
            sr,
            hop_length,
            FMIN_HZ,
            FMAX_HZ,
            model=model,
            batch_size=1024,
            device=device,
            return_periodicity=True
        )
        all_f0.append(f0_chunk.squeeze(0).cpu().numpy())
        all_conf.append(conf_chunk.squeeze(0).cpu().numpy())
    f0 = np.concatenate(all_f0)
    conf = np.concatenate(all_conf)
    f0[conf < CONF_THRESHOLD] = 0.0
    f0 = smooth_pitch(f0)
    times = np.arange(len(f0)) / FRAME_RATE
    return times, f0, conf

def segment_notes(times, freqs, min_note_sec=MIN_NOTE_SEC, gap_join_sec=GAP_JOIN_SEC):
    events = []
    start_idx, last_pitch = None, None
    for i in range(1, len(freqs) + 1):
        cur = freqs[i-1] if i-1 < len(freqs) else 0.0
        nxt = freqs[i] if i < len(freqs) else 0.0
        cur_note = hz_to_midi(cur) if cur > 0 else None
        nxt_note = hz_to_midi(nxt) if nxt > 0 else None
        if cur_note is not None and start_idx is None:
            start_idx, last_pitch = i - 1, cur_note
        change = (
            (cur_note is None and last_pitch is not None) or
            (cur_note is not None and nxt_note is None) or
            (cur_note is not None and nxt_note is not None and abs(cur_note - nxt_note) >= 0.75)
        )
        if start_idx is not None and change:
            end_idx = i
            onset, offset = times[start_idx], times[end_idx - 1]
            dur = max(0.0, offset - onset)
            if dur >= min_note_sec:
                seg_midis = [hz_to_midi(f) for f in freqs[start_idx:end_idx] if f > 0]
                if seg_midis:
                    midi_val = float(np.median(seg_midis))
                    events.append([onset, offset, midi_val])
            start_idx, last_pitch = None, None
    merged = []
    if events:
        merged = [events[0]]
        for (on2, off2, m2) in events[1:]:
            on1, off1, m1 = merged[-1]
            if abs(m1 - m2) <= 0.5 and (on2 - off1) <= gap_join_sec:
                merged[-1][1] = off2
                merged[-1][2] = (m1 + m2) / 2.0
            else:
                merged.append([on2, off2, m2])
    return merged

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
    import pretty_midi
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)
    for n in notes:
        midi_note = pretty_midi.Note(
            velocity=100,
            pitch=int(round(n[2]) if isinstance(n, list) else n['pitch']),
            start=n[0] if isinstance(n, list) else n['start'],
            end=n[1] if isinstance(n, list) else n['end']
        )
        instrument.notes.append(midi_note)
    midi.instruments.append(instrument)
    midi.write(output_path)

def midi_to_wav(midi_path, wav_path):
    musescore_exe = r"C:\Program Files\MuseScore 4\bin\musescore4.exe"
    try:
        cmd = [musescore_exe, "-o", str(wav_path), str(midi_path)]
        print("[DEBUG] Running:", cmd)

        result = subprocess.run(cmd, capture_output=True, text=True)

        print("[DEBUG] Return code:", result.returncode)
        if result.stdout:
            print("[STDOUT]", result.stdout.strip())
        if result.stderr:
            print("[STDERR]", result.stderr.strip())

        return result.returncode == 0
    except Exception as e:
        print(f"[ERROR] Exception during MuseScore conversion: {e}")
        return False

def process_audio_file_json(audio_path, emotion_classifier, output_dir, input_base_dir):
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
    # 2. Load and preprocess audio
    print("🎵 Loading audio...")
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    y = trim_and_normalize(y)
    # 3. Extract pitch with TorchCrepe
    print("🎹 Extracting pitch with TorchCrepe...")
    times, f0, conf = extract_f0_conf(y, sr, model='full')
    # 4. Segment into notes
    print("🎼 Detecting notes...")
    detected_notes = segment_notes(times, f0)
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
            'pipeline': 'Pipeline H: TorchCrepe JSON Roundtrip',
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
        summary_json_path = output_dir_path / "torchcrepe_json_summary.json"
        with open(summary_json_path, 'w') as f:
            json.dump(summary_data, f, indent=2)
        print(f"\n📊 Pipeline summary saved to: {summary_json_path}")
    return results

if __name__ == "__main__":
    BASE_DIR = Path(__file__).parent
    INPUT_DIR = BASE_DIR / "vocal-to-score-demo" / "Input" / "GTSinger_sample_50"
    OUTPUT_DIR = Path(__file__).parent / "output"
    MODEL_DIR = BASE_DIR / "results" / "emotion_model"
    MAX_FILES = None
    if INPUT_DIR.exists() and MODEL_DIR.exists():
        results = process_dataset_json(str(INPUT_DIR), str(OUTPUT_DIR), str(MODEL_DIR), max_files=MAX_FILES)
        print(f"\n✅ Processed {len(results)} files successfully!")
    else:
        print(f"❌ Error: Check paths exist:")
        print(f"  Input: {INPUT_DIR} (exists: {INPUT_DIR.exists()})")
        print(f"  Model: {MODEL_DIR} (exists: {MODEL_DIR.exists()})")
