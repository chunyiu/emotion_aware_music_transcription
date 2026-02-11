"""
Pipeline G: TorchCrepe with Emotion Classification
TorchCrepe pitch detection with RAVDESS emotion classifier
Adds top 2 emotions to PDF music sheet output
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

# Import music21 for MusicXML/PDF export
from music21 import stream, note, tempo, meter, clef, converter, metadata

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
        """Extract audio features for emotion classification"""
        try:
            y, sr_actual = librosa.load(audio_path, sr=sr, duration=duration, mono=True)
            
            # Pad if too short
            target_length = int(sr * duration)
            if len(y) < target_length:
                y = np.pad(y, (0, target_length - len(y)), mode='constant')
            
            # Extract MFCCs (mean and std)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
            mfccs_mean = np.mean(mfccs, axis=1)
            mfccs_std = np.std(mfccs, axis=1)
            
            # Extract mel-spectrogram (mean and std)
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            mel_spec_mean = np.mean(mel_spec, axis=1)
            mel_spec_std = np.std(mel_spec, axis=1)
            
            # Extract spectral features
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
            zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
            
            # Combine all features (must match RAVDESS model training: 339 features)
            features = np.concatenate([
                mfccs_mean,      # 40 features
                mfccs_std,       # 40 features
                mel_spec_mean,   # 128 features
                mel_spec_std,    # 128 features
                [spectral_centroid, spectral_rolloff, zero_crossing_rate]  # 3 features
            ])  # Total: 339 features
            
            return features
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None
    
    def predict(self, audio_path):
        """Predict emotion and return top 2 with confidence scores"""
        features = self.extract_features(audio_path)
        
        if features is None:
            return None
        
        # Reshape and scale
        features = features.reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        
        # Get prediction probabilities
        proba = self.model.predict_proba(features_scaled)[0]
        
        # Get top 2 predictions
        top_2_idx = np.argsort(proba)[-2:][::-1]
        
        emotions = self.le.inverse_transform(top_2_idx)
        confidences = proba[top_2_idx]
        
        return {
            'top1_emotion': emotions[0],
            'top1_confidence': confidences[0],
            'top2_emotion': emotions[1],
            'top2_confidence': confidences[1]
        }


# ============================================================================
# TORCHCREPE PITCH DETECTION
# ============================================================================

# Configuration
FRAME_RATE = 100
MEDIAN_FILTER_SIZE = 5
FMIN_HZ, FMAX_HZ = 50.0, 1100.0
CONF_THRESHOLD = 0.45
MIN_NOTE_SEC = 0.15
GAP_JOIN_SEC = 0.20
MIN_Q_LEN_BEATS = 0.25
DEFAULT_BPM = 120
CHUNK_SECONDS = 10

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def trim_and_normalize(y):
    """Trim silence and normalize audio"""
    y, _ = librosa.effects.trim(y, top_db=25)
    y = y / (np.max(np.abs(y)) + 1e-9)
    return y

def smooth_pitch(freqs, window=MEDIAN_FILTER_SIZE):
    """Apply median filter smoothing"""
    if window > 1:
        return medfilt(freqs, kernel_size=window)
    return freqs

def hz_to_midi(hz):
    """Convert Hz to MIDI note number"""
    if hz <= 0:
        return None
    return 69 + 12 * np.log2(hz / 440.0)

def extract_f0_conf(y, sr, model='full'):
    """Extract F0 using TorchCrepe"""
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
    
    # Apply confidence threshold
    f0[conf < CONF_THRESHOLD] = 0.0
    
    # Smooth
    f0 = smooth_pitch(f0)
    
    times = np.arange(len(f0)) / FRAME_RATE
    
    return times, f0, conf

def segment_notes(times, freqs, min_note_sec=MIN_NOTE_SEC, gap_join_sec=GAP_JOIN_SEC):
    """Segment continuous pitch into discrete notes"""
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
    
    # Merge similar adjacent notes
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

def notes_to_musicxml_with_emotion(notes_list, emotion_result, output_path, bpm=DEFAULT_BPM):
    """Convert notes to MusicXML with emotion labels"""
    sc = stream.Score()
    
    # Add metadata with emotions
    sc.metadata = metadata.Metadata()
    sc.metadata.title = "Vocal Transcription with Emotion Analysis"
    
    if emotion_result:
        emotion_text = f"Emotions: {emotion_result['top1_emotion']} ({emotion_result['top1_confidence']:.1%}), {emotion_result['top2_emotion']} ({emotion_result['top2_confidence']:.1%})"
        sc.metadata.composer = emotion_text
    
    # Create part
    p = stream.Part()
    p.insert(0, tempo.MetronomeMark(number=bpm))
    p.insert(0, meter.TimeSignature("4/4"))
    p.insert(0, clef.TrebleClef())
    
    sec_per_beat = 60.0 / bpm
    
    for onset, offset, midi_val in notes_list:
        dur_sec = max(0.0, offset - onset)
        if dur_sec < MIN_NOTE_SEC:
            continue
        
        ql = max(MIN_Q_LEN_BEATS, round((dur_sec / sec_per_beat) / (1/32)) * (1/32))
        n = note.Note(int(round(midi_val)))
        n.duration.quarterLength = ql
        p.append(n)
    
    sc.insert(0, p)
    sc.write("musicxml", fp=str(output_path))
    print(f"✅ Exported MusicXML with emotions to: {output_path}")


# ============================================================================
# GROUND TRUTH LOADING
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


# ============================================================================
# MAIN PROCESSING PIPELINE
# ============================================================================

def process_audio_file(audio_path, emotion_classifier, output_dir, input_base_dir):
    """Process single audio file with emotion classification"""
    
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
    
    # 1. Load ground truth
    ground_truth = load_ground_truth(audio_path)
    if ground_truth:
        print(f"📊 Ground Truth: {ground_truth['note_count']} notes, Emotion: {ground_truth['emotion']}")
    
    # 2. BEFORE: Predict emotions from original audio
    print("🎭 [BEFORE] Classifying emotion from original audio...")
    emotion_before = emotion_classifier.predict(audio_path)
    if emotion_before:
        print(f"  Top 1: {emotion_before['top1_emotion']} ({emotion_before['top1_confidence']:.1%})")
        print(f"  Top 2: {emotion_before['top2_emotion']} ({emotion_before['top2_confidence']:.1%})")
    else:
        print("  ⚠️ BEFORE emotion classification failed")
        return None

    # 3. Load and preprocess audio
    print("🎵 Loading audio...")
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    y = trim_and_normalize(y)

    # 4. Extract pitch with TorchCrepe
    print("🎹 Extracting pitch with TorchCrepe...")
    times, f0, conf = extract_f0_conf(y, sr, model='full')

    # 5. Segment into notes
    print("🎼 Detecting notes...")
    detected_notes = segment_notes(times, f0)
    print(f"  Detected {len(detected_notes)} notes")

    # 6. Export MusicXML with emotions using unique ID
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    musicxml_path = output_dir / f"{unique_id}_emotion.musicxml"
    notes_to_musicxml_with_emotion(detected_notes, emotion_before, musicxml_path)

    # 7. AFTER: Convert MusicXML to WAV and classify emotion
    print("🔄 [AFTER] Converting MusicXML to WAV and classifying emotion...")
    musescore_exe = r"C:\Program Files\MuseScore 4\bin\musescore4.exe"
    wav_after_path = output_dir / f"{unique_id}_after.wav"

    try:
        cmd = [musescore_exe, "-o", str(wav_after_path), str(musicxml_path)]
        print("  [DEBUG] Running:", cmd)

        result = subprocess.run(cmd, capture_output=True, text=True)

        print("  [DEBUG] Return code:", result.returncode)
        if result.stdout:
            print("  [STDOUT]", result.stdout.strip())
        if result.stderr:
            print("  [STDERR]", result.stderr.strip())

        if result.returncode == 0 and wav_after_path.exists():
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
            print(f"  Top 1: {emotion_after['top1_emotion']} ({emotion_after['top1_confidence']:.1%})")
            print(f"  Top 2: {emotion_after['top2_emotion']} ({emotion_after['top2_confidence']:.1%})")
        else:
            print("  ⚠️ AFTER emotion classification failed")

    # Per-file emotion comparison JSON (optional, pipe1-style)
    emotion_json = {
        "file": str(audio_path_obj.name),
        "unique_id": unique_id,
        "musicxml": str(musicxml_path.relative_to(output_dir)),
        "wav_after": str(wav_after_path.relative_to(output_dir)) if wav_after_path else None,
        "emotion_before": emotion_before,
        "emotion_after": emotion_after if emotion_after else None,
        "top1_changed": (
            emotion_before is not None and emotion_after is not None
            and emotion_before.get('top1_emotion') != emotion_after.get('top1_emotion')
        )
    }

    emotion_json_path = output_dir / f"{unique_id}_emotion_comparison.json"
    with open(emotion_json_path, 'w') as f:
        json.dump(emotion_json, f, indent=2)
    print(f"💾 Emotion comparison saved to: {emotion_json_path.name}")


    # 8. Prepare results with ground truth comparison and before/after emotions
    result = {
    'file': str(audio_path_obj.name),
    'unique_id': unique_id,
    'notes': len(detected_notes),
    'musicxml_output': str(musicxml_path.relative_to(output_dir)),
    'wav_after': str(wav_after_path.relative_to(output_dir)) if wav_after_path else None,
    'emotion_before': emotion_before,
    'emotion_after': emotion_after if emotion_after else {},
    'success': wav_after_path is not None and emotion_after is not None
}

    return result


def process_dataset(input_dir, output_dir, model_dir, max_files=None):
    """Process entire dataset"""
    
    # Initialize
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
    
    for i, audio_file in enumerate(audio_files, 1):
        print(f"\n[{i}/{len(audio_files)}]")
        
        try:
            result = process_audio_file(
                str(audio_file),
                emotion_classifier,
                output_dir,
                input_path
            )
            
            if result:
                results.append(result)
        
        except Exception as e:
            print(f"❌ Error processing {audio_file.name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save summary
    if results:
        output_path = Path(output_dir)
        summary_path = output_path / "processing_summary.csv"
        df = pd.DataFrame(results)
        df.to_csv(summary_path, index=False)
        print(f"\n📊 Saved summary to: {summary_path}")

        # Save JSON summary in standard pipeline format
        summary_json = {
    'pipeline': 'Pipeline G: TorchCrepe',
    'total_files': len(results),
    'successful': sum(1 for r in results if r['success']),
    'failed': sum(1 for r in results if not r['success']),
    'output_directories': {
        'musicxml': str(output_path),
        'wav_after': str(output_path),
        'emotion_results': str(output_path),
    },
    'results': results
}
        summary_json_path = output_path / "torchcrepe_summary.json"
        with open(summary_json_path, 'w') as f:
            json.dump(summary_json, f, indent=2)
        print(f"📄 JSON summary saved to: {summary_json_path}")

        # Print statistics
        print(f"\n{'='*70}")
        print("EMOTION DISTRIBUTION")
        print(f"{'='*70}")

        # Collect emotion + confidence + note counts from results
        before_emotions = {}
        before_confidences = []
        notes_counts = []

        for r in results:
            eb = r.get('emotion_before')
            if eb:
                emo = eb.get('top1_emotion')
                conf = eb.get('top1_confidence')
                if emo is not None:
                    before_emotions[emo] = before_emotions.get(emo, 0) + 1
                if conf is not None:
                    before_confidences.append(conf)
            notes_counts.append(r.get('notes', 0))

        # Print emotion distribution
        for emo, count in sorted(before_emotions.items(), key=lambda x: x[1], reverse=True):
            print(f"  {emo}: {count}")

        if before_confidences:
            print(f"\nAverage confidence: {np.mean(before_confidences):.1%}")
        else:
            print("\nAverage confidence: N/A")

        if notes_counts:
            print(f"Average notes detected: {np.mean(notes_counts):.1f}")
        else:
            print("Average notes detected: N/A")

    
    return results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Configuration - using absolute paths for reliability
    BASE_DIR = Path(__file__).parent  # Project root
    INPUT_DIR = BASE_DIR / "vocal-to-score-demo" / "Input" / "GTSinger_sample_50"
    OUTPUT_DIR = Path(__file__).parent / "output"
    MODEL_DIR = BASE_DIR / "results" / "emotion_model"
    MAX_FILES = None  # Process all 50 files for full comparison
    
    if INPUT_DIR.exists() and MODEL_DIR.exists():
        results = process_dataset(str(INPUT_DIR), str(OUTPUT_DIR), str(MODEL_DIR), max_files=MAX_FILES)
        print(f"\n✅ Processed {len(results)} files successfully!")
    else:
        print(f"❌ Error: Check paths exist:")
        print(f"  Input: {INPUT_DIR} (exists: {INPUT_DIR.exists()})")
        print(f"  Model: {MODEL_DIR} (exists: {MODEL_DIR.exists()})")
