"""
Pipeline 3: Librosa + TorchCrepe + Music21 Harmony with Emotion Classification
Combines TorchCrepe pitch detection with Music21 harmony generation
Includes before/after emotion analysis
"""

import os
import numpy as np
import librosa
import torch
import torchcrepe
from scipy.signal import medfilt
from pathlib import Path
from music21 import stream, note, chord, tempo, meter, clef, key as m21key, roman, analysis, metadata
import warnings
import pickle
import json
import sys
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from musicxml_to_wav import convert_musicxml_to_wav


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
FRAME_RATE = 100
MEDIAN_FILTER_SIZE = 5
FMIN_HZ, FMAX_HZ = 50.0, 1100.0
CONF_THRESHOLD = 0.45
MIN_NOTE_SEC = 0.15
GAP_JOIN_SEC = 0.20
MIN_Q_LEN_BEATS = 0.25
DEFAULT_BPM = 120
CHUNK_SECONDS = 10

# Harmony settings
HARMONY_MIN_MIDI = 55   # G3
HARMONY_MAX_MIDI = 79   # G5


# ============================================================================
# TORCHCREPE PITCH DETECTION
# ============================================================================
def extract_f0_conf(audio_path, frame_rate=FRAME_RATE, fmin=FMIN_HZ, fmax=FMAX_HZ, chunk_seconds=CHUNK_SECONDS):
    """Extract F0 and confidence using TorchCrepe (robust to short/odd files)."""
    y, sr = librosa.load(audio_path, sr=16000, mono=True)

    if y.size == 0:
        # Empty / unreadable audio
        raise ValueError(f"Empty audio signal for {audio_path}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    hop_length = int(sr / frame_rate)
    max_chunk = int(sr * chunk_seconds)

    # Short files: single call
    if len(y) <= max_chunk:
        try:
            out = torchcrepe.predict(
                torch.tensor(y[None, :], device=device),
                sr,
                hop_length=hop_length,
                fmin=fmin,
                fmax=fmax,
                model='full',
                batch_size=128,
                device=device,
                return_periodicity=True
            )
        except Exception as e:
            raise RuntimeError(f"TorchCrepe failed on {audio_path}: {e}")

        # Handle different return shapes robustly
        if isinstance(out, (list, tuple)) and len(out) >= 2:
            pitch, confidence = out[0], out[1]
        else:
            # If periodicity not returned for some reason, fake confidence = 1.0
            pitch = out[0] if isinstance(out, (list, tuple)) and len(out) == 1 else out
            confidence = torch.ones_like(pitch)

        f0 = pitch.squeeze(0).cpu().numpy()
        conf = confidence.squeeze(0).cpu().numpy()

        f0 = np.atleast_1d(f0)
        conf = np.atleast_1d(conf)

    else:
        all_f0, all_conf = [], []

        for start in range(0, len(y), max_chunk):
            end = min(start + max_chunk, len(y))
            chunk = y[start:end]
            if chunk.size == 0:
                continue

            try:
                out = torchcrepe.predict(
                    torch.tensor(chunk[None, :], device=device),
                    sr,
                    hop_length=hop_length,
                    fmin=fmin,
                    fmax=fmax,
                    model='full',
                    batch_size=128,
                    device=device,
                    return_periodicity=True
                )
            except Exception as e:
                print(f"    [TorchCrepe] Skipping chunk {start}:{end} for {audio_path}: {e}")
                continue

            if isinstance(out, (list, tuple)) and len(out) >= 2:
                pitch, confidence = out[0], out[1]
            else:
                pitch = out[0] if isinstance(out, (list, tuple)) and len(out) == 1 else out
                confidence = torch.ones_like(pitch)

            f0_chunk = pitch.squeeze(0).cpu().numpy()
            conf_chunk = confidence.squeeze(0).cpu().numpy()

            f0_chunk = np.atleast_1d(f0_chunk)
            conf_chunk = np.atleast_1d(conf_chunk)

            if f0_chunk.size == 0:
                continue

            all_f0.append(f0_chunk)
            all_conf.append(conf_chunk)

        if not all_f0:
            raise RuntimeError(f"TorchCrepe produced no frames for {audio_path}")

        f0 = np.concatenate(all_f0)
        conf = np.concatenate(all_conf)

    times = np.arange(len(f0)) / frame_rate
    return times, f0, conf



# ============================================================================
# NOTE SEGMENTATION
# ============================================================================
def segment_notes(times, f0, conf, conf_threshold=CONF_THRESHOLD, min_note_sec=MIN_NOTE_SEC, gap_join_sec=GAP_JOIN_SEC):
    """Segment pitch into notes"""
    # Apply confidence threshold
    f0_filt = f0.copy()
    f0_filt[conf < conf_threshold] = 0.0
    
    # Median filtering
    f0_smooth = medfilt(f0_filt, kernel_size=MEDIAN_FILTER_SIZE)
    
    # Convert to MIDI
    midi = np.zeros_like(f0_smooth)
    mask_voiced = (f0_smooth > 0)
    midi[mask_voiced] = librosa.hz_to_midi(f0_smooth[mask_voiced])
    
    # Segment into notes
    notes = []
    in_note = False
    note_start_idx = None
    note_pitches = []
    
    for i in range(len(midi)):
        if midi[i] > 0:
            if not in_note:
                # Start new note
                in_note = True
                note_start_idx = i
                note_pitches = [midi[i]]
            else:
                # Continue note or start new if pitch change
                if len(note_pitches) > 0:
                    pitch_diff = abs(midi[i] - np.median(note_pitches))
                    if pitch_diff > 0.5:  # New note
                        # End current note
                        note_end_idx = i - 1
                        duration = times[note_end_idx] - times[note_start_idx]
                        if duration >= min_note_sec:
                            notes.append({
                                'start': times[note_start_idx],
                                'end': times[note_end_idx],
                                'pitch': np.median(note_pitches)
                            })
                        # Start new note
                        note_start_idx = i
                        note_pitches = [midi[i]]
                    else:
                        note_pitches.append(midi[i])
        else:
            if in_note:
                # End note
                note_end_idx = i - 1
                duration = times[note_end_idx] - times[note_start_idx]
                if duration >= min_note_sec:
                    notes.append({
                        'start': times[note_start_idx],
                        'end': times[note_end_idx],
                        'pitch': np.median(note_pitches)
                    })
                in_note = False
                note_pitches = []
    
    # Handle final note
    if in_note and len(note_pitches) > 0:
        duration = times[-1] - times[note_start_idx]
        if duration >= min_note_sec:
            notes.append({
                'start': times[note_start_idx],
                'end': times[-1],
                'pitch': np.median(note_pitches)
            })
    
    # Merge close notes with similar pitch
    if len(notes) > 1:
        merged = [notes[0]]
        for n in notes[1:]:
            prev = merged[-1]
            gap = n['start'] - prev['end']
            pitch_diff = abs(n['pitch'] - prev['pitch'])
            
            if gap <= gap_join_sec and pitch_diff <= 0.5:
                # Merge
                prev['end'] = n['end']
                prev['pitch'] = (prev['pitch'] + n['pitch']) / 2
            else:
                merged.append(n)
        notes = merged
    
    return notes


# ============================================================================
# HARMONY GENERATION (Music21)
# ============================================================================
def get_melody_notes_at_measure(melody_part, measure_num, beats_per_measure=4.0):
    """Extract pitch classes of notes in a specific measure"""
    notes_in_measure = []
    measure_start = (measure_num - 1) * beats_per_measure
    measure_end = measure_start + beats_per_measure
    
    for n in melody_part.flatten().notes:
        if hasattr(n, 'pitch'):
            note_offset = float(n.offset)
            if measure_start <= note_offset < measure_end:
                notes_in_measure.append(n.pitch.pitchClass)
    
    return set(notes_in_measure)


def score_chord_fit(chord_rn, melody_pcs, key_obj):
    """Score how well a chord fits the melody notes"""
    try:
        rn = roman.RomanNumeral(chord_rn, key_obj)
        chord_pcs = {p.pitchClass for p in rn.pitches}
    except:
        return -100
    
    if not melody_pcs:
        return 0
    
    score = 0
    
    # Melody notes IN chord (highest weight)
    notes_in_chord = len(melody_pcs & chord_pcs)
    score += notes_in_chord * 10
    
    # Penalize melody notes NOT in chord
    notes_not_in_chord = len(melody_pcs - chord_pcs)
    score -= notes_not_in_chord * 5
    
    # Prefer stable chords
    if chord_rn in ["I", "i"]:
        score += 2
    elif chord_rn in ["IV", "iv", "V", "v"]:
        score += 1
    
    return score


def choose_best_chord(melody_pcs, key_obj, available_chords, prev_chord=None):
    """Choose best chord for melody notes"""
    best_chord = available_chords[0]
    best_score = -1000
    
    for chord_rn in available_chords:
        score = score_chord_fit(chord_rn, melody_pcs, key_obj)
        
        # Smooth voice leading bonus
        if prev_chord:
            if chord_rn == prev_chord:
                score += 3
            try:
                prev_rn = roman.RomanNumeral(prev_chord, key_obj)
                curr_rn = roman.RomanNumeral(chord_rn, key_obj)
                interval_dist = abs(prev_rn.scaleDegree - curr_rn.scaleDegree)
                if interval_dist in [4, 5, 3]:
                    score += 2
            except:
                pass
        
        if score > best_score:
            best_score = score
            best_chord = chord_rn
    
    return best_chord


def generate_harmony(melody_part, est_key=None):
    """Generate harmony using Music21"""
    # Analyze key if not provided
    if est_key is None:
        melody_score = stream.Score()
        melody_score.insert(0, melody_part)
        analyzer = analysis.discrete.KrumhanslSchmuckler()
        est_key = analyzer.getSolution(melody_score)
    
    if not isinstance(est_key, m21key.Key):
        est_key = m21key.Key('C')
    
    # Diatonic chords
    if est_key.mode == 'minor':
        diatonic_rn = ["i", "iio", "III", "iv", "v", "VI", "VII"]
    else:
        diatonic_rn = ["I", "ii", "iii", "IV", "V", "vi", "viio"]
    
    # Build harmony
    harmony_part = stream.Part(id="Harmony")
    harmony_part.append(meter.TimeSignature('4/4'))
    
    q_len = melody_part.duration.quarterLength
    num_measures = int(np.ceil(q_len / 4.0))
    
    prev_chord_rn = None
    for measure_num in range(1, num_measures + 1):
        melody_pcs = get_melody_notes_at_measure(melody_part, measure_num)
        best_chord_rn = choose_best_chord(melody_pcs, est_key, diatonic_rn, prev_chord_rn)
        
        try:
            rn = roman.RomanNumeral(best_chord_rn, est_key)
            ch = chord.Chord(rn.pitches)
            ch.quarterLength = 4.0
            harmony_part.append(ch)
            prev_chord_rn = best_chord_rn
        except:
            # Fallback to I
            rn = roman.RomanNumeral("I" if est_key.mode != 'minor' else "i", est_key)
            ch = chord.Chord(rn.pitches)
            ch.quarterLength = 4.0
            harmony_part.append(ch)
    
    return harmony_part, est_key


# ============================================================================
# MUSICXML EXPORT
# ============================================================================
def create_score_with_harmony(notes, output_path, title="TorchCrepe + Harmony"):
    """Create score with melody and harmony"""
    # Create melody part
    melody = stream.Part(id="Melody")
    melody.append(meter.TimeSignature('4/4'))
    melody.append(tempo.MetronomeMark(number=DEFAULT_BPM))
    
    for n in notes:
        pitch_midi = int(round(n['pitch']))
        duration_sec = n['end'] - n['start']
        ql = max(MIN_Q_LEN_BEATS, round(duration_sec * (DEFAULT_BPM / 60) / MIN_Q_LEN_BEATS) * MIN_Q_LEN_BEATS)
        
        music_note = note.Note()
        music_note.pitch.midi = pitch_midi
        music_note.quarterLength = ql
        melody.append(music_note)
    
    # Generate harmony
    harmony_part, est_key = generate_harmony(melody)
    
    # Create score
    score = stream.Score()
    score.insert(0, metadata.Metadata())
    score.metadata.title = title
    score.metadata.composer = f"Pipeline 3: TorchCrepe + Music21 | Key: {est_key}"
    
    score.append(melody)
    score.append(harmony_part)
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    score.write('musicxml', fp=str(output_path))
    
    return len(notes), est_key


# ============================================================================
# BATCH PROCESSING WITH EMOTION
# ============================================================================
def process_dataset(input_dir, output_dir, model_dir, max_files=None):
    """Process all audio files with before/after emotion classification"""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    musicxml_dir = output_dir / "musicxml"
    wav_after_dir = output_dir / "wav_after"
    emotion_dir = output_dir / "emotion_results"
    
    musicxml_dir.mkdir(exist_ok=True)
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
        
        try:
            # BEFORE: Emotion on original audio
            print(f"  [BEFORE] Classifying original audio...")
            emotion_before = emotion_classifier.predict(str(audio_path))
            
            if emotion_before:
                print(f"    Top: {emotion_before['top_emotion']} ({emotion_before['top_confidence']:.2%})")
                print(f"    2nd: {emotion_before['second_emotion']} ({emotion_before['second_confidence']:.2%})")
            
            # Extract pitch
            print(f"  [PROCESS] Extracting pitch with TorchCrepe...")
            times, f0, conf = extract_f0_conf(audio_path)
            
            if len(f0) == 0:
                raise RuntimeError("No F0 frames extracted; skipping file.")

            # Segment notes
            notes = segment_notes(times, f0, conf)
            
            
            # Generate harmony score
            stem = audio_path.stem
            musicxml_path = musicxml_dir / f"{stem}_harmony.musicxml"
            
            num_notes, est_key = create_score_with_harmony(notes, musicxml_path, title=stem)
            print(f"    Detected {num_notes} notes, Key: {est_key}")
            
            # AFTER: Convert MusicXML to WAV (includes harmony!)
            print(f"  [AFTER] Converting MusicXML+Harmony to WAV...")
            wav_after_path = wav_after_dir / f"{stem}_after.wav"
            
            success = convert_musicxml_to_wav(musicxml_path, wav_after_path, method='simple')
            
            if success:
                print(f"    ✓ Generated WAV from MusicXML+Harmony")
                
                # Classify emotion on reconstructed audio
                print(f"  [AFTER] Classifying reconstructed audio...")
                emotion_after = emotion_classifier.predict(str(wav_after_path))
                
                if emotion_after:
                    print(f"    Top: {emotion_after['top_emotion']} ({emotion_after['top_confidence']:.2%})")
                    print(f"    2nd: {emotion_after['second_emotion']} ({emotion_after['second_confidence']:.2%})")
                    print(f"    NOTE: AFTER emotion includes harmony influence!")
            else:
                print(f"    ✗ Failed to convert MusicXML to WAV")
                emotion_after = None
            
            # Save comparison results
            top1_before = (
                emotion_before.get('top_emotion')
                if isinstance(emotion_before, dict) else None
            )
            top1_after = (
                emotion_after.get('top_emotion')
                if isinstance(emotion_after, dict) else None
            )

            # define the path first so we can safely reference it in `result`
            harmony_json_path = emotion_dir / f"{stem}_harmony_emotion_delta.json"

            result = {
                'file': audio_path.name,
                'notes': num_notes,
                'key': str(est_key),
                'musicxml': str(musicxml_path.relative_to(output_dir)),
                'wav_after': str(wav_after_path.relative_to(output_dir)) if success else None,

                # full emotions
                'emotion_before': emotion_before,
                'emotion_after': emotion_after,

                # existing flag
                'emotion_preserved': (
                    top1_before is not None
                    and top1_after is not None
                    and top1_before == top1_after
                ),

                # --- harmony-related summary fields ---
                'harmony_top1_source': top1_before,
                'harmony_top1_after': top1_after,
                'harmony_source_confidence': (
                    float(emotion_before.get("top_confidence"))
                    if isinstance(emotion_before, dict)
                    and "top_confidence" in emotion_before
                    else None
                ),
                'harmony_after_confidence': (
                    float(emotion_after.get("top_confidence"))
                    if isinstance(emotion_after, dict)
                    and "top_confidence" in emotion_after
                    else None
                ),
                'harmony_top1_changed': (
                    top1_before is not None
                    and top1_after is not None
                    and top1_before != top1_after
                ),
                'harmony_json_path': str(harmony_json_path.relative_to(output_dir)),

                'note': 'AFTER emotion includes harmony (melody + chords); harmony delta recorded.',
                'success': True
            }

            # separate JSON focusing on harmony delta
            harmony_compare = {
                "file": audio_path.name,
                "musicxml": str(musicxml_path.relative_to(output_dir)),
                "wav_after": str(wav_after_path.relative_to(output_dir)) if success else None,
                "emotion_source": emotion_before,
                "emotion_harmony": emotion_after,
                "top1_source": top1_before,
                "top1_harmony": top1_after,
                "source_confidence": (
                    float(emotion_before.get("top_confidence"))
                    if isinstance(emotion_before, dict)
                    and "top_confidence" in emotion_before
                    else None
                ),
                "harmony_confidence": (
                    float(emotion_after.get("top_confidence"))
                    if isinstance(emotion_after, dict)
                    and "top_confidence" in emotion_after
                    else None
                ),
                "top1_changed": (
                    top1_before is not None
                    and top1_after is not None
                    and top1_before != top1_after
                )
            }

            with open(harmony_json_path, 'w') as f:
                json.dump(harmony_compare, f, indent=2)
            print(f"  💾 Harmony emotion comparison saved to: {harmony_json_path.name}\n")


            # ---------------------------------------------------------
            # NEW: separate JSON focusing ONLY on emotion comparison
            # ---------------------------------------------------------
            harmony_compare = {
                "file": audio_path.name,
                "musicxml": str(musicxml_path.relative_to(output_dir)),
                "wav_after": str(wav_after_path.relative_to(output_dir)) if success else None,

                # Emotion that harmony was generated from
                "emotion_source": emotion_before,        # original vocal

                # Emotion of the audio that includes the generated harmony
                "emotion_harmony": emotion_after,        # melody + chords

                # Convenience fields
                "top1_source": top1_before,
                "top1_harmony": top1_after,
                "source_confidence": (
                    float(emotion_before.get("top_confidence"))
                    if isinstance(emotion_before, dict)
                    and "top_confidence" in emotion_before
                    else None
                ),
                "harmony_confidence": (
                    float(emotion_after.get("top_confidence"))
                    if isinstance(emotion_after, dict)
                    and "top_confidence" in emotion_after
                    else None
                ),
                "top1_changed": (
                    top1_before is not None
                    and top1_after is not None
                    and top1_before != top1_after
                )
            }

            harmony_json_path = emotion_dir / f"{stem}_harmony_emotion_delta.json"
            with open(harmony_json_path, 'w') as f:
                json.dump(harmony_compare, f, indent=2)
            print(f"  💾 Harmony emotion comparison saved to: {harmony_json_path.name}\n")

            # Track in list for summary
            results.append(result)
            
        except Exception as e:
            print(f"  ✗ Error: {e}\n")
            results.append({
                'file': audio_path.name,
                'success': False,
                'error': str(e)
            })
    
    # Save summary
    print(f"{'='*60}")
    successful = sum(1 for r in results if r['success'])
    print(f"Processing complete! Successful: {successful}/{len(results)}")
    print(f"{'='*60}\n")
    
    summary_path = output_dir / "pipeline_3_summary.json"
    summary = {
        'pipeline': 'Pipeline 3: TorchCrepe + Music21 Harmony',
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
    OUTPUT_DIR = BASE_DIR / "output_pipeline_3_emotion"
    MODEL_DIR = BASE_DIR / "results" / "emotion_model"
    MAX_FILES = None
    
    print("="*60)
    print("PIPELINE 3: TORCHCREPE + HARMONY + EMOTION")
    print("="*60)
    print(f"Input:  {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"Emotion Model: {MODEL_DIR}")
    print("="*60)
    print()
    
    process_dataset(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        model_dir=MODEL_DIR,
        max_files=MAX_FILES
    )
