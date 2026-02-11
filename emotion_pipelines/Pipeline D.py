"""
Pipeline D: Librosa + PYIN + HMM + Viterbi → MusicXML with Emotion Classification
Based on PYIN + HMM + Viterbi transcription, outputs MusicXML files with before/after emotion analysis
"""

import librosa
import numpy as np
from pathlib import Path
import json
import scipy.signal
from music21 import stream, note, meter, tempo, metadata
import warnings
import pickle
import sys
warnings.filterwarnings('ignore')

# Add src to path for utilities
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from musicxml_to_wav import convert_musicxml_to_wav


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
            
            # Extract features
            # 1. MFCCs (40 mean + 40 std = 80 features)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
            mfccs_mean = np.mean(mfccs, axis=1)
            mfccs_std = np.std(mfccs, axis=1)
            
            # 2. Mel spectrogram (128 mean + 128 std = 256 features)
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            mel_spec_mean = np.mean(mel_spec, axis=1)
            mel_spec_std = np.std(mel_spec, axis=1)
            
            # 3. Spectral features (3 features)
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
            
            # Concatenate all features (80 + 256 + 3 = 339 features)
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
        """Predict emotion from audio file"""
        features = self.extract_features(audio_path)
        if features is None:
            return None
        
        # Scale and predict
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        probas = self.model.predict_proba(features_scaled)[0]
        
        # Get top 2 emotions
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
# EXPERIMENT PARAMETERS
# ============================================================================
class ExperimentConfig:
    """Central configuration for tunable parameters"""
    
    # Note Detection Parameters
    min_note_duration: float = 0.3
    pitch_threshold: float = 0.5
    vibrato_tolerance: float = 0.3
    
    # Post-processing Parameters
    merge_threshold: float = 0.5
    min_gap: float = 0.08
    
    # Signal Processing Parameters
    median_kernel_size: int = 7
    
    # pYIN Parameters
    pyin_fmin: float = 65.4
    pyin_fmax: float = 1047.0
    pyin_frame_length: int = 2048
    pyin_hop_length: int = 256
    
    # Viterbi HMM Parameters
    self_trans_prob: float = 0.85
    neighbor_prob: float = 0.06
    unvoiced_stay: float = 0.5


# ============================================================================
# VITERBI CONVERTER
# ============================================================================
class VocalToMIDIConverter_Viterbi:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.fmin = 65.4
        self.fmax = 1047.0
        self.midi_min = int(librosa.hz_to_midi(self.fmin))
        self.midi_max = int(librosa.hz_to_midi(self.fmax))
        self.midi_notes = np.arange(self.midi_min, self.midi_max + 1)
        self.n_states = len(self.midi_notes)
        
        self.transition_matrix = self.build_transition_matrix()
        self.start_prob = np.ones(self.n_states + 1) / (self.n_states + 1)

    def build_transition_matrix(self):
        """Build HMM transition matrix"""
        n = self.n_states
        A = np.zeros((n + 1, n + 1))
        
        cfg = self.config
        for i in range(n):
            A[i, i] = cfg.self_trans_prob
            if i > 0:
                A[i, i - 1] = cfg.neighbor_prob
            if i < n - 1:
                A[i, i + 1] = cfg.neighbor_prob
            
            remaining = 1 - A[i].sum()
            A[i, -1] = max(0, remaining)
        
        A[-1, -1] = cfg.unvoiced_stay
        remaining = 1 - cfg.unvoiced_stay
        A[-1, :-1] = remaining / n
        
        return A

    def compute_emission_probabilities(self, observed_midi, voiced_prob, sigma=2.0):
        """Compute emission probabilities for HMM"""
        emissions = np.zeros(self.n_states + 1)
        
        if observed_midi > 0 and voiced_prob > 0.1:
            for i, midi_note in enumerate(self.midi_notes):
                diff = abs(observed_midi - midi_note)
                emissions[i] = np.exp(-0.5 * (diff / sigma) ** 2) * voiced_prob
            emissions[-1] = (1 - voiced_prob) * 0.2
        else:
            emissions[:-1] = 0.1
            emissions[-1] = 0.8
        
        emissions /= (emissions.sum() + 1e-12)
        return emissions

    def extract_pitch_features(self, audio_path):
        """Extract pitch features using pYIN"""
        y, sr = librosa.load(audio_path, sr=None)
        
        cfg = self.config
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=cfg.pyin_fmin,
            fmax=cfg.pyin_fmax,
            sr=sr,
            frame_length=cfg.pyin_frame_length,
            hop_length=cfg.pyin_hop_length
            # win_length removed - will use default (frame_length // 2)
        )
        
        times = librosa.times_like(f0, sr=sr, hop_length=cfg.pyin_hop_length)
        return times, f0, voiced_probs

    def viterbi_decode(self, f0, voiced_probs):
        """Viterbi decoding with log probabilities"""
        T = len(f0)
        N = self.n_states + 1
        
        log_delta = np.full((T, N), -np.inf)
        psi = np.zeros((T, N), dtype=int)
        
        # Convert to MIDI
        midi_obs = np.zeros(T)
        for t in range(T):
            if f0[t] is not None and not np.isnan(f0[t]) and f0[t] > 0:
                midi_obs[t] = librosa.hz_to_midi(f0[t])
        
        # Initialize
        em0 = self.compute_emission_probabilities(midi_obs[0], voiced_probs[0])
        log_delta[0] = np.log(self.start_prob + 1e-12) + np.log(em0 + 1e-12)
        
        # Forward pass
        log_A = np.log(self.transition_matrix + 1e-12)
        
        for t in range(1, T):
            em_t = self.compute_emission_probabilities(midi_obs[t], voiced_probs[t])
            log_em_t = np.log(em_t + 1e-12)
            
            for j in range(N):
                trans_probs = log_delta[t-1] + log_A[:, j]
                psi[t, j] = np.argmax(trans_probs)
                log_delta[t, j] = trans_probs[psi[t, j]] + log_em_t[j]
        
        # Backtrack
        path = np.zeros(T, dtype=int)
        path[-1] = np.argmax(log_delta[-1])
        
        for t in range(T-2, -1, -1):
            path[t] = psi[t+1, path[t+1]]
        
        # Convert to MIDI sequence
        smoothed = np.zeros(T)
        for t in range(T):
            if path[t] < self.n_states:
                smoothed[t] = self.midi_notes[path[t]]
            else:
                smoothed[t] = 0
        
        return smoothed

    def convert(self, audio_path):
        """Main conversion pipeline"""
        times, f0, voiced_probs = self.extract_pitch_features(audio_path)
        
        # Viterbi decoding
        smoothed_midi = self.viterbi_decode(f0, voiced_probs)
        
        # Median filtering
        smoothed_midi = scipy.signal.medfilt(
            smoothed_midi, 
            kernel_size=self.config.median_kernel_size
        )
        
        # Convert to note events
        notes = self.midi_to_note_events(times, smoothed_midi)
        notes = self.post_process_notes(notes)
        
        return notes

    def midi_to_note_events(self, times, midi_seq):
        """Convert MIDI sequence to note events"""
        notes = []
        if len(midi_seq) == 0:
            return notes
        
        current_pitch = midi_seq[0]
        start_time = times[0]
        pitch_accumulator = [current_pitch] if current_pitch > 0 else []
        
        for i in range(1, len(midi_seq)):
            current_val = midi_seq[i]
            prev_val = midi_seq[i-1]
            
            pitch_diff = abs(current_val - current_pitch)
            is_silence_boundary = (
                (current_val == 0 and prev_val > 0) or 
                (current_val > 0 and prev_val == 0)
            )
            is_pitch_change = pitch_diff > self.config.pitch_threshold
            
            if is_silence_boundary or (is_pitch_change and current_val > 0):
                if len(pitch_accumulator) > 0:
                    median_pitch = np.median(pitch_accumulator)
                    duration = times[i-1] - start_time
                    
                    if duration >= self.config.min_note_duration:
                        notes.append({
                            'start': start_time,
                            'end': times[i-1],
                            'pitch': median_pitch
                        })
                
                start_time = times[i]
                pitch_accumulator = [current_val] if current_val > 0 else []
                current_pitch = current_val
            elif current_val > 0:
                pitch_accumulator.append(current_val)
        
        # Final note
        if len(pitch_accumulator) > 0:
            median_pitch = np.median(pitch_accumulator)
            duration = times[-1] - start_time
            if duration >= self.config.min_note_duration:
                notes.append({
                    'start': start_time,
                    'end': times[-1],
                    'pitch': median_pitch
                })
        
        return notes

    def post_process_notes(self, notes):
        """Merge similar adjacent notes"""
        if len(notes) <= 1:
            return notes
        
        processed = []
        current_note = notes[0].copy()
        
        for next_note in notes[1:]:
            pitch_diff = abs(next_note['pitch'] - current_note['pitch'])
            time_gap = next_note['start'] - current_note['end']
            
            should_merge = (
                (pitch_diff <= self.config.merge_threshold and time_gap <= self.config.min_gap) or
                (pitch_diff <= self.config.merge_threshold and time_gap <= 0)
            )
            
            if should_merge:
                current_note['end'] = next_note['end']
                dur1 = current_note['end'] - current_note['start']
                dur2 = next_note['end'] - next_note['start']
                current_note['pitch'] = (
                    current_note['pitch'] * dur1 + next_note['pitch'] * dur2
                ) / (dur1 + dur2)
            else:
                processed.append(current_note)
                current_note = next_note.copy()
        
        processed.append(current_note)
        return processed


# ============================================================================
# MUSICXML EXPORT
# ============================================================================
def notes_to_musicxml(notes, output_path, title="PYIN+HMM+Viterbi Transcription"):
    """Convert notes to MusicXML format"""
    # Create melody part
    melody = stream.Part()
    melody.id = "Melody"
    
    # Add metadata
    melody.append(meter.TimeSignature('4/4'))
    melody.append(tempo.MetronomeMark(number=90))
    
    # Create score with metadata
    score = stream.Score()
    score.insert(0, metadata.Metadata())
    score.metadata.title = title
    score.metadata.composer = "Pipeline D: PYIN + HMM + Viterbi"
    
    # Convert notes
    for n in notes:
        pitch_midi = int(round(n['pitch']))
        duration_sec = n['end'] - n['start']
        
        # Quantize duration to nearest 0.25 quarter notes
        ql = max(0.25, round(duration_sec / 0.25) * 0.25)
        
        # Create note
        music_note = note.Note()
        music_note.pitch.midi = pitch_midi
        music_note.quarterLength = ql
        melody.append(music_note)
    
    score.append(melody)
    
    # Save to MusicXML
    output_path.parent.mkdir(parents=True, exist_ok=True)
    score.write('musicxml', fp=str(output_path))
    
    return len(notes)


# ============================================================================
# BATCH PROCESSING WITH EMOTION
# ============================================================================
def process_dataset(input_dir, output_dir, model_dir, max_files=None):
    """Process all audio files with before/after emotion classification"""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for different outputs
    musicxml_dir = output_dir / "musicxml"
    wav_after_dir = output_dir / "wav_after"
    emotion_dir = output_dir / "emotion_results"
    
    musicxml_dir.mkdir(exist_ok=True)
    wav_after_dir.mkdir(exist_ok=True)
    emotion_dir.mkdir(exist_ok=True)
    
    # Initialize converter and emotion classifier
    config = ExperimentConfig()
    converter = VocalToMIDIConverter_Viterbi(config)
    emotion_classifier = EmotionClassifier(model_dir)
    
    # Find audio files
    audio_files = list(input_dir.rglob('*.wav'))
    audio_files = sorted(audio_files)
    
    if max_files:
        audio_files = audio_files[:max_files]
    
    print(f"Found {len(audio_files)} audio files")
    print(f"Output directory: {output_dir}\n")
    
    results = []
    for i, audio_path in enumerate(audio_files, 1):
        print(f"[{i}/{len(audio_files)}] Processing: {audio_path.name}")
        
        try:
            # ========== BEFORE: Emotion on original audio ==========
            print(f"  [BEFORE] Classifying original audio...")
            emotion_before = emotion_classifier.predict(str(audio_path))
            
            if emotion_before:
                print(f"    Top: {emotion_before['top_emotion']} ({emotion_before['top_confidence']:.2%})")
                print(f"    2nd: {emotion_before['second_emotion']} ({emotion_before['second_confidence']:.2%})")
            
            # ========== Convert audio to notes ==========
            print(f"  [PROCESS] Converting audio to MusicXML...")
            notes = converter.convert(audio_path)
            
            # Generate output path
            stem = audio_path.stem
            musicxml_path = musicxml_dir / f"{stem}.musicxml"
            
            # Export to MusicXML
            num_notes = notes_to_musicxml(notes, musicxml_path, title=stem)
            print(f"    Detected {num_notes} notes")
            
            # ========== AFTER: Convert MusicXML back to WAV ==========
            print(f"  [AFTER] Converting MusicXML to WAV...")
            wav_after_path = wav_after_dir / f"{stem}_after.wav"
            
            success = convert_musicxml_to_wav(musicxml_path, wav_after_path, method='simple')
            
            if success:
                print(f"    ✓ Generated WAV from MusicXML")
                
                # Classify emotion on reconstructed audio
                print(f"  [AFTER] Classifying reconstructed audio...")
                emotion_after = emotion_classifier.predict(str(wav_after_path))
                
                if emotion_after:
                    print(f"    Top: {emotion_after['top_emotion']} ({emotion_after['top_confidence']:.2%})")
                    print(f"    2nd: {emotion_after['second_emotion']} ({emotion_after['second_confidence']:.2%})")
            else:
                print(f"    ✗ Failed to convert MusicXML to WAV")
                emotion_after = None
            
            # ========== Save comparison results ==========
            result = {
                'file': audio_path.name,
                'notes': num_notes,
                'musicxml': str(musicxml_path.relative_to(output_dir)),
                'wav_after': str(wav_after_path.relative_to(output_dir)) if success else None,
                'emotion_before': emotion_before,
                'emotion_after': emotion_after,
                'success': True
            }
            
            # Save individual result JSON
            result_json_path = emotion_dir / f"{stem}_emotion_comparison.json"
            with open(result_json_path, 'w') as f:
                json.dump(result, f, indent=2)
            
            results.append(result)
            print(f"  ✓ Saved emotion comparison to {result_json_path.name}\n")
            
        except Exception as e:
            print(f"  ✗ Error: {e}\n")
            results.append({
                'file': audio_path.name,
                'success': False,
                'error': str(e)
            })
    
    # ========== Save summary ==========
    print(f"{'='*60}")
    print(f"Processing complete!")
    successful = sum(1 for r in results if r['success'])
    print(f"Successful: {successful}/{len(results)}")
    print(f"{'='*60}\n")
    
    # Save comprehensive summary
    summary_path = output_dir / "pipeline_d_summary.json"
    summary = {
        'pipeline': 'Pipeline D: PYIN + HMM + Viterbi',
        'total_files': len(results),
        'successful': successful,
        'failed': len(results) - successful,
        'output_directories': {
            'musicxml': str(musicxml_dir.relative_to(output_dir)),
            'wav_after': str(wav_after_dir.relative_to(output_dir)),
            'emotion_results': str(emotion_dir.relative_to(output_dir))
        },
        'results': results
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Saved summary to {summary_path}")
    
    # Print emotion comparison statistics
    print(f"\n{'='*60}")
    print("EMOTION COMPARISON STATISTICS")
    print(f"{'='*60}")
    
    successful_results = [r for r in results if r['success'] and r['emotion_before'] and r['emotion_after']]
    
    if successful_results:
        # Count emotion matches
        matches = sum(1 for r in successful_results 
                     if r['emotion_before']['top_emotion'] == r['emotion_after']['top_emotion'])
        
        print(f"Files with emotion analysis: {len(successful_results)}")
        print(f"Emotion preserved (before = after): {matches}/{len(successful_results)} ({matches/len(successful_results)*100:.1f}%)")
        
        # Show emotion distribution
        emotions_before = {}
        emotions_after = {}
        
        for r in successful_results:
            em_before = r['emotion_before']['top_emotion']
            em_after = r['emotion_after']['top_emotion']
            
            emotions_before[em_before] = emotions_before.get(em_before, 0) + 1
            emotions_after[em_after] = emotions_after.get(em_after, 0) + 1
        
        print(f"\nEmotion Distribution (BEFORE):")
        for em, count in sorted(emotions_before.items(), key=lambda x: -x[1]):
            print(f"  {em}: {count}")
        
        print(f"\nEmotion Distribution (AFTER):")
        for em, count in sorted(emotions_after.items(), key=lambda x: -x[1]):
            print(f"  {em}: {count}")
    
    print(f"{'='*60}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    # Configuration
    BASE_DIR = Path(__file__).parent
    INPUT_DIR = BASE_DIR / "vocal-to-score-demo" / "Input" / "GTSinger_sample_50"
    OUTPUT_DIR = BASE_DIR / "output_pipeline_d_emotion"
    MODEL_DIR = BASE_DIR / "results" / "emotion_model"
    MAX_FILES = None  # None = process all
    
    print("="*60)
    print("PIPELINE D: PYIN + HMM + VITERBI → MUSICXML + EMOTION")
    print("="*60)
    print(f"Input:  {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Model:  {MODEL_DIR}")
    print(f"Max files: {MAX_FILES or 'ALL'}")
    print("="*60)
    print()
    
    process_dataset(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        model_dir=MODEL_DIR,
        max_files=MAX_FILES
    )
