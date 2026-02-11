"""
Pipeline F: Librosa + CREPE → JSON with Emotion Classification
Uses regular CREPE for pitch detection, outputs JSON format
Includes before/after emotion analysis (uses temp MusicXML for WAV conversion)
"""

import numpy as np
import librosa
import crepe
from scipy.signal import medfilt
from pathlib import Path
import json
import warnings
import pickle
import sys
import tempfile
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from musicxml_to_wav import convert_musicxml_to_wav
from music21 import stream, note, tempo, meter, metadata


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
        """Extract 339 audio features"""
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
            spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
            
            features = np.concatenate([
                mfccs_mean, mfccs_std,
                mel_spec_mean, mel_spec_std,
                [spectral_centroid, spectral_bandwidth, spectral_rolloff]
            ])
            
            return features
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None
    
    def predict(self, audio_path):
        """Predict emotion"""
        features = self.extract_features(audio_path)
        if features is None:
            return None
        
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        probas = self.model.predict_proba(features_scaled)[0]
        
        top2_indices = np.argsort(probas)[-2:][::-1]
        top2_emotions = [
            {
                'emotion': self.le.inverse_transform([idx])[0],
                'confidence': float(probas[idx])
            }
            for idx in top2_indices
        ]
        
        return {
            'top_emotion': top2_emotions[0]['emotion'],
            'top_confidence': top2_emotions[0]['confidence'],
            'second_emotion': top2_emotions[1]['emotion'],
            'second_confidence': top2_emotions[1]['confidence'],
            'top2': top2_emotions
        }


# ============================================================================
# CONFIGURATION
# ============================================================================
CREPE_MODEL = 'full'
CREPE_STEP_SIZE = 10
MEDIAN_FILTER_SIZE = 5
CONF_THRESHOLD = 0.45
MIN_NOTE_SEC = 0.15
GAP_JOIN_SEC = 0.20


# ============================================================================
# CREPE PITCH DETECTION
# ============================================================================
def extract_f0_conf_crepe(audio_path, model=CREPE_MODEL, step_size=CREPE_STEP_SIZE):
    """Extract F0 and confidence using CREPE"""
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    
    time, frequency, confidence, activation = crepe.predict(
        y, sr,
        model_capacity=model,
        step_size=step_size,
        viterbi=True
    )
    
    return time, frequency, confidence


# ============================================================================
# NOTE SEGMENTATION
# ============================================================================
def segment_notes(times, f0, conf, conf_threshold=CONF_THRESHOLD, min_note_sec=MIN_NOTE_SEC, gap_join_sec=GAP_JOIN_SEC):
    """Segment pitch into notes"""
    f0_filt = f0.copy()
    f0_filt[conf < conf_threshold] = 0.0
    
    f0_smooth = medfilt(f0_filt, kernel_size=MEDIAN_FILTER_SIZE)
    
    midi = np.zeros_like(f0_smooth)
    mask_voiced = (f0_smooth > 0)
    midi[mask_voiced] = librosa.hz_to_midi(f0_smooth[mask_voiced])
    
    notes = []
    in_note = False
    note_start_idx = None
    note_pitches = []
    
    for i in range(len(midi)):
        if midi[i] > 0:
            if not in_note:
                in_note = True
                note_start_idx = i
                note_pitches = [midi[i]]
            else:
                if len(note_pitches) > 0:
                    pitch_diff = abs(midi[i] - np.median(note_pitches))
                    if pitch_diff > 0.5:
                        note_end_idx = i - 1
                        duration = times[note_end_idx] - times[note_start_idx]
                        if duration >= min_note_sec:
                            notes.append({
                                'start': float(times[note_start_idx]),
                                'end': float(times[note_end_idx]),
                                'pitch': float(np.median(note_pitches)),
                                'duration': float(duration)
                            })
                        note_start_idx = i
                        note_pitches = [midi[i]]
                    else:
                        note_pitches.append(midi[i])
        else:
            if in_note:
                note_end_idx = i - 1
                duration = times[note_end_idx] - times[note_start_idx]
                if duration >= min_note_sec:
                    notes.append({
                        'start': float(times[note_start_idx]),
                        'end': float(times[note_end_idx]),
                        'pitch': float(np.median(note_pitches)),
                        'duration': float(duration)
                    })
                in_note = False
                note_pitches = []
    
    if in_note and len(note_pitches) > 0:
        duration = times[-1] - times[note_start_idx]
        if duration >= min_note_sec:
            notes.append({
                'start': float(times[note_start_idx]),
                'end': float(times[-1]),
                'pitch': float(np.median(note_pitches)),
                'duration': float(duration)
            })
    
    # Merge close notes
    if len(notes) > 1:
        merged = [notes[0]]
        for n in notes[1:]:
            prev = merged[-1]
            gap = n['start'] - prev['end']
            pitch_diff = abs(n['pitch'] - prev['pitch'])
            
            if gap <= gap_join_sec and pitch_diff <= 0.5:
                prev['end'] = n['end']
                prev['duration'] = prev['end'] - prev['start']
                prev['pitch'] = (prev['pitch'] + n['pitch']) / 2
            else:
                merged.append(n)
        notes = merged
    
    return notes


# ============================================================================
# JSON EXPORT
# ============================================================================
def notes_to_json(notes, output_path, audio_filename):
    """Export notes to JSON format"""
    output_data = {
        'audio_file': audio_filename,
        'pipeline': 'Pipeline F: CREPE → JSON',
        'num_notes': len(notes),
        'notes': notes
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    return len(notes)


def create_temp_musicxml_from_notes(notes, title="CREPE Transcription"):
    """Create temporary MusicXML from notes for WAV conversion"""
    melody = stream.Part(id="Melody")
    melody.append(meter.TimeSignature('4/4'))
    melody.append(tempo.MetronomeMark(number=120))
    
    score = stream.Score()
    score.insert(0, metadata.Metadata())
    score.metadata.title = title
    score.metadata.composer = "Pipeline F: CREPE → JSON"
    
    for n in notes:
        pitch_midi = int(round(n['pitch']))
        duration_sec = n['duration']
        ql = max(0.25, round(duration_sec / 0.25) * 0.25)
        
        music_note = note.Note()
        music_note.pitch.midi = pitch_midi
        music_note.quarterLength = ql
        melody.append(music_note)
    
    score.append(melody)
    
    # Save to temp file
    temp_file = tempfile.NamedTemporaryFile(suffix='.musicxml', delete=False)
    temp_path = Path(temp_file.name)
    temp_file.close()
    
    score.write('musicxml', fp=str(temp_path))
    
    return temp_path


# ============================================================================
# BATCH PROCESSING WITH EMOTION
# ============================================================================
def process_dataset(input_dir, output_dir, model_dir, max_files=None):
    """Process all audio files with before/after emotion classification"""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    json_dir = output_dir / "json"
    wav_after_dir = output_dir / "wav_after"
    emotion_dir = output_dir / "emotion_results"
    
    json_dir.mkdir(exist_ok=True)
    wav_after_dir.mkdir(exist_ok=True)
    emotion_dir.mkdir(exist_ok=True)
    
    # Initialize emotion classifier
    emotion_classifier = EmotionClassifier(model_dir)
    
    audio_files = list(input_dir.rglob('*.wav'))
    audio_files = sorted(audio_files)
    
    if max_files:
        audio_files = audio_files[:max_files]
    
    print(f"Found {len(audio_files)} audio files\n")
    
    results = []
    for i, audio_path in enumerate(audio_files, 1):
        print(f"[{i}/{len(audio_files)}] Processing: {audio_path.name}")
        
        temp_musicxml = None
        try:
            # BEFORE: Emotion on original audio
            print(f"  [BEFORE] Classifying original audio...")
            emotion_before = emotion_classifier.predict(str(audio_path))
            
            if emotion_before:
                print(f"    Top: {emotion_before['top_emotion']} ({emotion_before['top_confidence']:.2%})")
                print(f"    2nd: {emotion_before['second_emotion']} ({emotion_before['second_confidence']:.2%})")
            
            # Extract pitch with CREPE
            print(f"  [PROCESS] Extracting pitch with CREPE...")
            times, f0, conf = extract_f0_conf_crepe(audio_path)
            
            # Segment notes
            notes = segment_notes(times, f0, conf)
            
            # Export to JSON
            stem = audio_path.stem
            json_path = json_dir / f"{stem}.json"
            
            num_notes = notes_to_json(notes, json_path, audio_path.name)
            print(f"    Detected {num_notes} notes")
            
            # AFTER: Create temp MusicXML for WAV conversion
            print(f"  [AFTER] Creating temp MusicXML for emotion analysis...")
            temp_musicxml = create_temp_musicxml_from_notes(notes, title=stem)
            
            # Convert MusicXML to WAV
            wav_after_path = wav_after_dir / f"{stem}_after.wav"
            success = convert_musicxml_to_wav(temp_musicxml, wav_after_path, method='simple')
            
            if success:
                print(f"    ✓ Generated WAV from notes")
                
                # Classify emotion on reconstructed audio
                print(f"  [AFTER] Classifying reconstructed audio...")
                emotion_after = emotion_classifier.predict(str(wav_after_path))
                
                if emotion_after:
                    print(f"    Top: {emotion_after['top_emotion']} ({emotion_after['top_confidence']:.2%})")
                    print(f"    2nd: {emotion_after['second_emotion']} ({emotion_after['second_confidence']:.2%})")
            else:
                print(f"    ✗ Failed to convert to WAV")
                emotion_after = None
            
            # Save comparison results
            result = {
                'file': audio_path.name,
                'notes': num_notes,
                'json': str(json_path.relative_to(output_dir)),
                'wav_after': str(wav_after_path.relative_to(output_dir)) if success else None,
                'emotion_before': emotion_before,
                'emotion_after': emotion_after,
                'success': True
            }
            
            result_json_path = emotion_dir / f"{stem}_emotion_comparison.json"
            with open(result_json_path, 'w') as f:
                json.dump(result, f, indent=2)
            
            results.append(result)
            print(f"  ✓ Saved to {result_json_path.name}\n")
            
        except Exception as e:
            print(f"  ✗ Error: {e}\n")
            import traceback
            traceback.print_exc()
            results.append({
                'file': audio_path.name,
                'success': False,
                'error': str(e)
            })
        
        finally:
            # Clean up temp MusicXML
            if temp_musicxml and temp_musicxml.exists():
                temp_musicxml.unlink()
    
    # Save summary
    print(f"{'='*60}")
    successful = sum(1 for r in results if r['success'])
    print(f"Processing complete! Successful: {successful}/{len(results)}")
    print(f"{'='*60}\n")
    
    summary_path = output_dir / "pipeline_f_summary.json"
    summary = {
        'pipeline': 'Pipeline F: CREPE → JSON',
        'total_files': len(results),
        'successful': successful,
        'failed': len(results) - successful,
        'results': results
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Saved summary to {summary_path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    BASE_DIR = Path(__file__).parent
    INPUT_DIR = BASE_DIR / "vocal-to-score-demo" / "Input" / "GTSinger_sample_50"
    OUTPUT_DIR = BASE_DIR / "output_pipeline_f_emotion"
    MODEL_DIR = BASE_DIR / "results" / "emotion_model"
    MAX_FILES = None
    
    print("="*60)
    print("PIPELINE F: CREPE → JSON + EMOTION")
    print("="*60)
    print(f"Input:  {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Model:  {CREPE_MODEL}")
    print(f"Emotion Model: {MODEL_DIR}")
    print("="*60)
    print()
    
    process_dataset(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        model_dir=MODEL_DIR,
        max_files=MAX_FILES
    )
