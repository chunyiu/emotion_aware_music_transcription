import pickle
import sys
from pathlib import Path
from music21 import converter, stream
sys.path.append(str(Path(__file__).parent.parent / "src"))
from musicxml_to_wav import musicxml_to_wav_fluidsynth  # (legacy, unused)

# Use MuseScore-based MusicXML-to-WAV conversion
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
import subprocess

# ============================================================================
# EMOTION CLASSIFIER
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
            'top1_confidence': float(confidences[0]),
            'top2_emotion': emotions[1],
            'top2_confidence': float(confidences[1])
        }
"""
Pipeline 1: Librosa + HMM + Music21 + Music21
Consolidated script to process GTSinger dataset samples
Uses music21 for harmony generation (no mingus dependency)
"""

import librosa
import librosa.display
import music21 as m21
import numpy as np
import json
import matplotlib.pyplot as plt
import mir_eval
import scipy.signal
import os
from fractions import Fraction
from music21 import stream, duration, meter, note, chord, tie
import pandas as pd
from pathlib import Path
import random

# ============================================================================
# CONFIGURATION
# ============================================================================

class ExperimentConfig:
    """Central configuration for tunable parameters"""
    # Note Detection Parameters
    min_note_duration: float = 0.1    
    pitch_threshold: float = 0.5        # Semitones - pitch change detection
    vibrato_tolerance: float = 0.3      # Semitones - vibrato handling
    
    # Post-processing Parameters
    merge_threshold: float = 0.5        # Semitones - merge similar pitches
    min_gap: float = 0.08              # Seconds - minimum silence between notes
    
    # Signal Processing Parameters
    median_kernel_size: int = 3        
    
    # pYIN Parameters
    pyin_fmin: float = librosa.note_to_hz('G3') # 196.0 Hz
    pyin_fmax: float = librosa.note_to_hz('G5') # 784.0 Hz
    pyin_frame_length: int = 2048      # Samples
    pyin_hop_length: int = 256         # Samples
    
    # Viterbi HMM Parameters
    self_trans_prob: float = 0.7       
    neighbor_prob: float = 0.15       
    unvoiced_stay: float = 0.5         # Stay unvoiced

    def to_dict(self):
        return {
            'min_note_duration': self.min_note_duration,
            'pitch_threshold': self.pitch_threshold,
            'vibrato_tolerance': self.vibrato_tolerance,
            'merge_threshold': self.merge_threshold,
            'min_gap': self.min_gap,
            'median_kernel_size': self.median_kernel_size,
            'pyin_fmin': self.pyin_fmin,
            'pyin_fmax': self.pyin_fmax,
            'pyin_frame_length': self.pyin_frame_length,
            'pyin_hop_length': self.pyin_hop_length,
            'self_trans_prob': self.self_trans_prob,
            'neighbor_prob': self.neighbor_prob,
            'unvoiced_stay': self.unvoiced_stay
        }

# ============================================================================
# EVALUATION CLASS
# ============================================================================

class VocalMIDIEvaluator:
    """Evaluate vocal to MIDI conversion"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
    
    def load_ground_truth_from_json(self, json_path: str):
        """Load ground truth from GTSinger JSON format"""
        try:
            with open(json_path, 'r') as f: data = json.load(f)
            ref_intervals, ref_pitches = [], []
            for word_entry in data:
                note_starts = word_entry.get('note_start', [])
                note_ends = word_entry.get('note_end', [])
                notes = word_entry.get('note', [])
                for start, end, midi_note in zip(note_starts, note_ends, notes):
                    if midi_note > 0:
                        ref_intervals.append([float(start), float(end)])
                        ref_pitches.append(float(midi_note))
            return np.array(ref_intervals), np.array(ref_pitches)
        except Exception as e:
            print(f"Error loading JSON: {e}")
            return np.array([]), np.array([])
    
    def predictions_to_intervals(self, notes_dict_list):
        """Converts the list of note dicts to mir_eval format"""
        pred_intervals = []
        pred_pitches = []
        for n in notes_dict_list:
            pred_intervals.append([n['start'], n['end']])
            pred_pitches.append(n['pitch'])
        return np.array(pred_intervals), np.array(pred_pitches)
    
    def evaluate(self, predicted_notes_dict, json_path):
        """Evaluate predictions against JSON ground truth"""
        ref_intervals, ref_pitches = self.load_ground_truth_from_json(json_path)
        
        if len(ref_pitches) == 0:
            print("Error: No ground truth notes found.")
            return self._empty_metrics()
        
        pred_intervals, pred_pitches = self.predictions_to_intervals(predicted_notes_dict)
        print(f"Ground truth notes: {len(ref_pitches)}")
        print(f"Predicted notes: {len(pred_pitches)}")
        
        if len(pred_pitches) == 0:
            print("Warning: No notes predicted.")
            return self._empty_metrics(ref_count=len(ref_pitches))
        
        # Note-level metrics
        try:
            precision, recall, f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
                ref_intervals=ref_intervals,
                ref_pitches=ref_pitches,
                est_intervals=pred_intervals,
                est_pitches=pred_pitches,
                onset_tolerance=0.1,
                pitch_tolerance=0.5
            )
        except Exception as e:
            print(f"Note-level evaluation error: {e}")
            precision = recall = f1 = 0.0
        
        metrics = {
            'precision': float(precision),
            'recall': float(recall),
            'f_measure': float(f1),
            'ref_count': int(len(ref_pitches)),
            'pred_count': int(len(pred_pitches))
        }
        print(f"    ✓ F1: {f1:.3f} | P: {precision:.3f} | R: {recall:.3f}")
        return metrics

    def _empty_metrics(self, ref_count=0, pred_count=0):
        """Return empty metrics dict"""
        return {
            'precision': 0.0, 'recall': 0.0, 'f_measure': 0.0,
            'ref_count': ref_count, 'pred_count': pred_count
        }

# ============================================================================
# VITERBI HMM CONVERTER
# ============================================================================

class VocalToMIDIConverter_Viterbi:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.fmin = config.pyin_fmin
        self.fmax = config.pyin_fmax
        self.midi_min = int(librosa.hz_to_midi(self.fmin))
        self.midi_max = int(librosa.hz_to_midi(self.fmax))
        self.midi_notes = np.arange(self.midi_min, self.midi_max + 1)
        self.n_states = len(self.midi_notes)
        
        self.transition_matrix = self.build_transition_matrix()
        self.start_prob = np.ones(self.n_states + 1) / (self.n_states + 1)

    def build_transition_matrix(self):
        n = self.n_states
        A = np.zeros((n + 1, n + 1))
        cfg = self.config
        for i in range(n):
            A[i, i] = cfg.self_trans_prob
            if i > 0: A[i, i - 1] = cfg.neighbor_prob
            if i < n - 1: A[i, i + 1] = cfg.neighbor_prob
            remaining = 1 - A[i].sum()
            A[i, -1] = max(0, remaining)
        A[-1, -1] = cfg.unvoiced_stay
        remaining = 1 - cfg.unvoiced_stay
        A[-1, :-1] = remaining / n
        return A

    def compute_emission_probabilities(self, observed_midi, voiced_prob, sigma=2.0):
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

    def viterbi_decode(self, f0, voiced_probs):
        n_frames = len(f0)
        n_states = self.n_states + 1
        log_A = np.log(self.transition_matrix + 1e-12)
        log_pi = np.log(self.start_prob + 1e-12)
        log_delta = np.zeros((n_frames, n_states))
        psi = np.zeros((n_frames, n_states), dtype=int)
        
        obs_midi = np.zeros_like(f0)
        valid_mask = (f0 > 0) & np.isfinite(f0)
        obs_midi[valid_mask] = librosa.hz_to_midi(f0[valid_mask])
        
        log_delta[0] = log_pi + np.log(self.compute_emission_probabilities(obs_midi[0], voiced_probs[0]) + 1e-12)
        
        for t in range(1, n_frames):
            emission = np.log(self.compute_emission_probabilities(obs_midi[t], voiced_probs[t]) + 1e-12)
            for j in range(n_states):
                seq_probs = log_delta[t - 1] + log_A[:, j]
                psi[t, j] = np.argmax(seq_probs)
                log_delta[t, j] = np.max(seq_probs) + emission[j]
        
        states = np.zeros(n_frames, dtype=int)
        states[-1] = np.argmax(log_delta[-1])
        for t in range(n_frames - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]
        
        smoothed_midi = np.array([self.midi_notes[s] if s < self.n_states else 0 for s in states])
        return smoothed_midi

# ============================================================================
# NOTE DETECTION FUNCTIONS
# ============================================================================

def improved_midi_to_note_events(times, midi_seq, config: ExperimentConfig):
    """
    Two-pass note detection with local pitch stability analysis
    """
    def round_time(t, step=0.01):
        return np.round(t / step) * step
    
    notes = []
    if len(midi_seq) == 0:
        return notes
    
    # Pass 1: Compute local pitch stability
    window_size = 5
    pitch_variance = np.zeros_like(midi_seq, dtype=float)
    
    for i in range(len(midi_seq)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(midi_seq), i + window_size // 2 + 1)
        window = midi_seq[start_idx:end_idx]
        
        voiced_window = window[window > 0]
        if len(voiced_window) > 1:
            pitch_variance[i] = np.std(voiced_window)
    
    # Pass 2: Segment with stability awareness
    current_pitch = midi_seq[0]
    start_time = times[0]
    pitch_accumulator = [current_pitch] if current_pitch > 0 else []
    
    for i in range(1, len(midi_seq)):
        current_val = midi_seq[i]
        prev_val = midi_seq[i-1]
        
        # Dynamic threshold based on local stability
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
    
    # Handle final note
    if len(times) > 1:
        frame_duration = times[-1] - times[-2]
        end_time = times[-1] + frame_duration
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
    """Post-process note list to merge similar adjacent notes"""
    if len(notes) <= 1:
        return notes
    
    processed = []
    current_note = notes[0].copy()
    
    for i in range(1, len(notes)):
        next_note = notes[i]
        
        pitch_diff = abs(next_note['pitch'] - current_note['pitch'])
        time_gap = next_note['start'] - current_note['end']
        
        should_merge = (
            pitch_diff <= config.merge_threshold and 
            time_gap <= config.min_gap
        )
        
        if should_merge:
            current_note['end'] = next_note['end']
            current_note['pitch'] = (current_note['pitch'] + next_note['pitch']) / 2
        else:
            processed.append(current_note)
            current_note = next_note.copy()
    
    processed.append(current_note)
    return processed

# ============================================================================
# MUSICXML EXPORT FUNCTIONS
# ============================================================================

LEGAL_QL = (2.0, 1.0, 0.5, 0.25, 0.125, 0.75, 0.375, 0.1875)

def _nearest_legal_ql(ql: float, legal=LEGAL_QL):
    return min(legal, key=lambda x: abs(x - ql))

def _split_into_legal_segments(ql: float, legal=LEGAL_QL):
    chunks = []
    remaining = ql
    eps = 1e-9
    while remaining > eps:
        candidates = [l for l in legal if l <= remaining + eps]
        if not candidates: l = _nearest_legal_ql(remaining, legal)
        else: l = max(candidates)
        chunks.append(l)
        remaining -= l
    return chunks

def _retime_part_flat(part, denom_limit=8):
    from music21 import note, chord, duration
    part_flat = part.flatten().sorted()
    cleaned = stream.Part()
    cleaned.id = part.id
    cleaned.partName = getattr(part, 'partName', None)
    cur_offset = 0.0
    for el in part_flat.notesAndRests:
        raw_ql = float(el.duration.quarterLength)
        if raw_ql <= 0: raw_ql = 1.0 / denom_limit
        ql = float(Fraction(raw_ql).limit_denominator(denom_limit))
        segs = _split_into_legal_segments(ql, LEGAL_QL)
        def _clone_like(e):
            if isinstance(e, note.Note):
                c = note.Note(e.pitch)
                for ly in getattr(e, 'lyrics', []) or []:
                    t = getattr(ly, 'text', None)
                    if t: c.addLyric(t)
                return c
            elif isinstance(e, chord.Chord):
                c = chord.Chord(e.pitches)
                for ly in getattr(e, 'lyrics', []) or []:
                    t = getattr(ly, 'text', None)
                    if t: c.addLyric(t)
                return c
            else: return note.Rest()
        for i, seg in enumerate(segs):
            new_el = _clone_like(el)
            new_el.duration = duration.Duration(seg)
            cleaned.insert(cur_offset, new_el)
            if len(segs) > 1:
                if i == 0: new_el.tie = tie.Tie('start')
                elif i == len(segs) - 1: new_el.tie = tie.Tie('stop')
                else: new_el.tie = tie.Tie('continue')
            cur_offset += seg
    return cleaned

def export_musicxml_safely(score: stream.Score, fp: str, ts='4/4', denom_limit=8):
    cleaned_score = stream.Score()
    cleaned_score.metadata = getattr(score, 'metadata', None)
    for p in score.parts:
        cp = _retime_part_flat(p, denom_limit=denom_limit)
        if not cp.recurse().getElementsByClass(meter.TimeSignature).first():
            cp.insert(0.0, meter.TimeSignature(ts))
        cleaned_score.insert(0.0, cp)
    cleaned_score.makeMeasures(inPlace=True)
    cleaned_score.makeNotation(inPlace=True)
    cleaned_score.stripTies(inPlace=True)
    cleaned_score.sort()
    bad = []
    for part in cleaned_score.parts:
        for n in part.recurse().notesAndRests:
            t = getattr(n.duration, 'type', None)
            if t in ('inexpressible', 'complex') or n.duration.quarterLength <= 0:
                bad.append(n)
    if bad:
        for n in bad:
            n.duration = duration.Duration(1.0)
            if hasattr(n, 'tie'): n.tie = None
        cleaned_score.makeMeasures(inPlace=True)
        cleaned_score.makeNotation(inPlace=True)
        cleaned_score.stripTies(inPlace=True)
        cleaned_score.sort()
    cleaned_score.write('musicxml', fp=fp)
    return cleaned_score

# ============================================================================
# HARMONY GENERATION (MUSIC21 ONLY)
# ============================================================================

def generate_harmony(score, interval_str='-M3'):
    """
    Generate harmony by transposing the melody part
    Uses music21 transpose (no mingus dependency)
    """
    melody_part = score.getElementById('melody')
    if melody_part is None:
        print("Error: 'melody' part not found for harmony.")
        return score
    
    harmony_part = melody_part.transpose(interval_str)
    harmony_part.id = 'harmony'
    harmony_part.partName = "Harmony"   # <-- add this
    harmony_part.insert(0, m21.instrument.fromString('Voice'))
    harmony_part.insert(0, m21.clef.TenorClef())
    score.insert(0, harmony_part)
    
    print("Harmony generation complete (music21 transpose).")
    return score

# ============================================================================
# HARMONY-ONLY EXTRACTION FOR EMOTION ANALYSIS
# ============================================================================

from music21 import converter as m21_converter

def extract_harmony_only_xml(full_xml_path, harmony_only_xml_path):
    """
    Extract only the 'harmony' part from a MusicXML file and save as a new XML.
    Returns True if a harmony part is found and written, else False.
    Heuristics:
      1) Prefer parts whose id/partName contains 'harm'
      2) If none, and there are >= 2 parts, assume the LAST part is harmony
    """
    try:
        score = m21_converter.parse(full_xml_path)
        harmony_score = stream.Score()
        harmony_parts = []

        # 1) try to find part by id/partName
        for p in score.parts:
            pid = (getattr(p, "id", "") or "").lower()
            pname = (getattr(p, "partName", "") or "").lower()
            if "harm" in pid or "harm" in pname:
                harmony_parts.append(p)

        # 2) fallback: if none found but there are multiple parts,
        #    assume the LAST part is harmony (melody is usually first)
        if not harmony_parts and len(score.parts) >= 2:
            fallback_part = score.parts[-1]
            harmony_parts.append(fallback_part)
            print("    [HARMONY] No explicit 'harmony' label; "
                  "using last part as harmony fallback.")

        if not harmony_parts:
            print("    [HARMONY] No harmony part detected in MusicXML (even after fallback).")
            return False

        for hp in harmony_parts:
            harmony_score.append(hp)

        harmony_only_xml_path.parent.mkdir(parents=True, exist_ok=True)
        harmony_score.write("musicxml", fp=str(harmony_only_xml_path))
        print(f"    [HARMONY] Harmony-only MusicXML written to: {harmony_only_xml_path}")
        return True

    except Exception as e:
        print(f"    [HARMONY] Failed to extract harmony-only XML: {e}")
        return False



# ============================================================================
# ANNOTATION FUNCTIONS
# ============================================================================

def load_annotations_from_json(json_path):
    with open(json_path, 'r') as f: data = json.load(f)
    annotations = []
    for entry in data:
        word, start, emotion = entry['word'], entry['start_time'], entry.get('emotion', '')
        if word not in ["<AP>", "<SP>"]:
            annotations.append((start, word, emotion))
    return annotations

def add_annotations_to_score(score, annotations, tempo_bpm=100):
    melody_part = score.getElementById('melody')
    if melody_part is None:
        print("Error: 'melody' part not found in score.")
        return score
        
    tempo_mark = melody_part.getElementsByClass(m21.tempo.MetronomeMark).first()
    if tempo_mark:
        tempo_bpm = tempo_mark.number
        
    all_notes = list(melody_part.recurse().notes)
    
    for start_time, word, emotion in annotations:
        offset = start_time * (tempo_bpm / 60)
        note_to_add_lyric = None
        min_offset_diff = float('inf')
        
        for n in all_notes:
            if n.offset >= offset:
                offset_diff = n.offset - offset
                if offset_diff < min_offset_diff:
                    min_offset_diff = offset_diff
                    note_to_add_lyric = n
        if note_to_add_lyric:
            note_to_add_lyric.addLyric(word)
            if emotion:
                text_label = m21.expressions.TextExpression(f"({emotion})")
                text_label.placement = 'above'
                melody_part.insert(note_to_add_lyric.offset, text_label)
            
    return score

def estimate_tempo_safely(y, sr):
    """
    Safely estimate tempo with fallback for different librosa versions
    """
    try:
        # Try new API (librosa >= 0.10.0)
        tempo_array = librosa.feature.rhythm.tempo(y=y, sr=sr)
        estimated_tempo = tempo_array[0] if len(tempo_array) > 0 else 100.0
    except AttributeError:
        try:
            # Try old API (librosa < 0.10.0)
            tempo_array = librosa.beat.tempo(y=y, sr=sr)
            estimated_tempo = tempo_array[0] if len(tempo_array) > 0 else 100.0
        except:
            # Fallback to default
            print("    Warning: Could not estimate tempo, using default 100 BPM")
            estimated_tempo = 100.0
    
    if estimated_tempo == 0 or not np.isfinite(estimated_tempo):
        estimated_tempo = 100.0
    
    return float(estimated_tempo)

def notes_to_score_v2(notes_dict, y, sr, instrument_name='Voice'):
    """
    Converts the list of note dictionaries into a music21 score
    """
    score = m21.stream.Score()
    melody_part = m21.stream.Part(id='melody')
    melody_part.insert(0, m21.instrument.fromString(instrument_name))
    
    # Use safe tempo estimation
    estimated_tempo = estimate_tempo_safely(y, sr)
    print(f"  - Estimated tempo: {estimated_tempo:.2f} BPM")
    tempo_mark = m21.tempo.MetronomeMark(number=estimated_tempo)
    melody_part.insert(0, tempo_mark)
    
    for n_dict in notes_dict:
        n = m21.note.Note()
        n.pitch.midi = n_dict['pitch']
        
        start_time = n_dict['start']
        end_time = n_dict['end']
        duration_in_seconds = end_time - start_time
        
        quarter_length = duration_in_seconds * (estimated_tempo / 60)
        offset = start_time * (estimated_tempo / 60)
        
        n.duration.quarterLength = quarter_length
        melody_part.insert(offset, n)
        
    score.insert(0, melody_part)
    return score

# ============================================================================
# MAIN PROCESSING PIPELINE
# ============================================================================

def process_single_file(audio_path, json_path, config, output_musicxml=None):
    """Process a single audio file through the pipeline"""
    
    # Load audio
    y, sr = librosa.load(audio_path, sr=44100)
    
    # Extract pitch using pYIN
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, fmin=config.pyin_fmin, fmax=config.pyin_fmax, sr=sr,
        frame_length=config.pyin_frame_length, hop_length=config.pyin_hop_length
    )
    times = librosa.times_like(f0, sr=sr, hop_length=config.pyin_hop_length)

    # Viterbi decoding
    converter = VocalToMIDIConverter_Viterbi(config)
    smoothed_midi = converter.viterbi_decode(f0, voiced_probs)
    smoothed_midi = scipy.signal.medfilt(smoothed_midi, kernel_size=config.median_kernel_size)
    
    # Detect notes
    detected_notes = improved_midi_to_note_events(times, smoothed_midi, config)
    detected_notes = post_process_notes(detected_notes, config)
    
    # Evaluate
    evaluator = VocalMIDIEvaluator(config)
    metrics = evaluator.evaluate(detected_notes, json_path)
    
    # Generate score
    generated_score = notes_to_score_v2(detected_notes, y, sr)
    
    # Add annotations
    if os.path.exists(json_path):
        annotations = load_annotations_from_json(json_path)
        generated_score = add_annotations_to_score(generated_score, annotations)
    
    # Generate harmony (using music21 transpose)
    score_with_harmony = generate_harmony(generated_score)


    
    # Export
    if output_musicxml:
        os.makedirs(os.path.dirname(output_musicxml), exist_ok=True)
        export_musicxml_safely(score_with_harmony, fp=output_musicxml, ts='4/4', denom_limit=8)
        print(f"✅ Exported MusicXML to {output_musicxml}")
    
    return metrics, detected_notes

def process_gtsinger_dataset(dataset_path, output_dir, config, max_files=50):
    """Process entire GTSinger dataset"""
    
    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all audio files
    audio_files = [f for f in dataset_path.rglob("*.wav")
                  if "Paired_Speech_Group" not in f.parts]
    
    print(f"\n✓ Found {len(audio_files)} audio files")
    
    if max_files and len(audio_files) > max_files:
        audio_files = random.sample(audio_files, max_files)
        print(f"  Processing {max_files} files...")
    
    all_results = []
    # Load emotion classifier
    emotion_model_dir = Path('results/emotion_model')
    emotion_classifier = EmotionClassifier(emotion_model_dir)
    
    for i, audio_file in enumerate(audio_files, 1):
        print(f"\n[{i}/{len(audio_files)}] 🎵 {audio_file.name}")
        print(f"🎵 Processing: {audio_file.relative_to(dataset_path)}")
        rel_path = audio_file.relative_to(dataset_path)
        output_subdir = output_dir / rel_path.parent
        output_subdir.mkdir(parents=True, exist_ok=True)
        musicxml_output = output_subdir / f"{audio_file.stem}_emotion.musicxml"
        wav_after_output = output_subdir / f"{audio_file.stem}_after.wav"
        json_path = audio_file.with_suffix('.json')
        before_emotion = emotion_classifier.predict(str(audio_file))
        try:
            if json_path.exists():
                metrics, notes = process_single_file(
                    str(audio_file),
                    str(json_path),
                    config,
                    output_musicxml=str(musicxml_output)
                )
                # Convert MusicXML to WAV for AFTER emotion
                after_emotion = None
                try:
                    print(f"    [INFO] Converting MusicXML to WAV (MuseScore): {musicxml_output} -> {wav_after_output}")
                    success = musicxml_to_wav_musescore(str(musicxml_output), str(wav_after_output))
                    if not success:
                        print(f"    [ERROR] musicxml_to_wav_musescore returned False for {musicxml_output}")
                    if not wav_after_output.exists():
                        print(f"    [ERROR] Expected WAV file not found: {wav_after_output}")
                    if success and wav_after_output.exists():
                        print(f"    [INFO] WAV file created: {wav_after_output}")
                        after_emotion = emotion_classifier.predict(str(wav_after_output))
                        if after_emotion is None:
                            print(f"    [ERROR] Emotion classifier returned None for {wav_after_output}")
                    else:
                        print(f"    [ERROR] MusicXML-to-WAV conversion failed for {musicxml_output}")
                except Exception as e:
                    print(f"    ✗ Exception in MusicXML→WAV or after emotion: {e}")
                    import traceback
                    traceback.print_exc()

                # Write per-file before/after emotion JSON
                emotion_json = {
                    "file": audio_file.name,
                    "musicxml": str(musicxml_output.relative_to(output_dir)),
                    "wav_after": str(wav_after_output.relative_to(output_dir)),
                    "emotion_before": before_emotion,
                    "emotion_after": after_emotion,
                    "top1_changed": (
                        before_emotion is not None and after_emotion is not None and
                        before_emotion.get('top1_emotion') != after_emotion.get('top1_emotion')
                    )
                }
                emotion_json_path = output_subdir / f"{audio_file.stem}_emotion_comparison.json"
                with open(emotion_json_path, 'w') as f:
                    json.dump(emotion_json, f, indent=2)
                print(f"    [INFO] Wrote before/after emotion JSON: {emotion_json_path}")

                                # ------------------------------------------------------------------
                # HARMONY-ONLY EMOTION COMPARISON (separate JSON)
                # ------------------------------------------------------------------
                harmony_only_xml = output_subdir / f"{audio_file.stem}_harmony_only.musicxml"
                harmony_only_wav = output_subdir / f"{audio_file.stem}_harmony_only.wav"
                harmony_emotion = None

                try:
                    ok_harm_xml = extract_harmony_only_xml(musicxml_output, harmony_only_xml)
                    if ok_harm_xml:
                        print(f"    [HARMONY] Converting harmony-only XML to WAV: {harmony_only_xml} -> {harmony_only_wav}")
                        success_harmony = musicxml_to_wav_musescore(str(harmony_only_xml), str(harmony_only_wav))

                        if success_harmony and harmony_only_wav.exists():
                            harmony_emotion = emotion_classifier.predict(str(harmony_only_wav))
                            if harmony_emotion:
                                print(f"    [HARMONY] Top1: {harmony_emotion['top1_emotion']} "
                                      f"({harmony_emotion['top1_confidence']:.1%})")
                        else:
                            print("    [HARMONY] Harmony-only WAV generation failed or file missing.")
                    else:
                        print("    [HARMONY] Skipping harmony-only emotion (no harmony part).")
                except Exception as e:
                    print(f"    [HARMONY] Exception during harmony-only processing: {e}")
                    import traceback
                    traceback.print_exc()

                # Write separate harmony emotion JSON
                harmony_json = {
                    "file": audio_file.name,
                    "melody_top1": before_emotion.get("top1_emotion") if before_emotion else None,
                    "melody_confidence": before_emotion.get("top1_confidence") if before_emotion else None,
                    "harmony_top1": harmony_emotion.get("top1_emotion") if harmony_emotion else None,
                    "harmony_confidence": harmony_emotion.get("top1_confidence") if harmony_emotion else None,
                    "match": (
                        before_emotion is not None
                        and harmony_emotion is not None
                        and before_emotion.get("top1_emotion") == harmony_emotion.get("top1_emotion")
                    ),
                    "musicxml_full": str(musicxml_output.relative_to(output_dir)),
                    "musicxml_harmony_only": (
                        str(harmony_only_xml.relative_to(output_dir))
                        if harmony_only_xml.exists() else None
                    ),
                    "wav_harmony_only": (
                        str(harmony_only_wav.relative_to(output_dir))
                        if harmony_only_wav.exists() else None
                    ),
                }

                harmony_json_path = output_subdir / f"{audio_file.stem}_harmony_emotion.json"
                with open(harmony_json_path, "w") as f:
                    json.dump(harmony_json, f, indent=2)
                print(f"    [HARMONY] Wrote harmony-only emotion JSON: {harmony_json_path}")


                                # Summary entry for pipe1_summary.json (includes harmony info)
                harmony_match_with_before = (
                    before_emotion is not None
                    and harmony_emotion is not None
                    and before_emotion.get("top1_emotion") == harmony_emotion.get("top1_emotion")
                )

                result = {
                    'file': audio_file.name,
                    'notes': len(notes) if notes is not None else 0,

                    # main outputs
                    'musicxml': str(musicxml_output.relative_to(output_dir)),
                    'wav_after': str(wav_after_output.relative_to(output_dir)),

                    # melody/vocal emotions
                    'emotion_before': before_emotion,
                    'emotion_after': after_emotion,

                    # harmony-only emotions
                    'harmony_emotion': harmony_emotion,
                    'harmony_musicxml': (
                        str(harmony_only_xml.relative_to(output_dir))
                        if harmony_only_xml.exists() else None
                    ),
                    'harmony_wav': (
                        str(harmony_only_wav.relative_to(output_dir))
                        if harmony_only_wav.exists() else None
                    ),

                    # flags
                    'success': bool(after_emotion is not None),
                    'harmony_success': bool(harmony_emotion is not None),
                    'harmony_match_with_before': harmony_match_with_before,
                }
                all_results.append(result)

            else:
                result = {
                    'file': audio_file.name,
                    'notes': 0,
                    'musicxml': None,
                    'wav_after': None,
                    'emotion_before': before_emotion,
                    'emotion_after': None,
                    'success': False
                }
                all_results.append(result)
        except Exception as e:
            print(f"    ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            result = {'file': audio_file.name, 'notes': 0, 'musicxml': None, 'wav_after': None, 'emotion_before': before_emotion, 'emotion_after': None, 'success': False}
            all_results.append(result)
    
    # Compose summary in the required format
    total_files = len(all_results)
    successful = sum(1 for r in all_results if r['success'])
    failed = total_files - successful
    summary = {
        "pipeline": "Pipeline A: Librosa+HMM+Music21+Music21",
        "total_files": total_files,
        "successful": successful,
        "failed": failed,
        "output_directories": {
            "musicxml": "musicxml",
            "wav_after": "wav_after",
            "emotion_results": "emotion_results"
        },
        "results": all_results
    }
    summary_json_path = output_dir / "pipe1_summary.json"
    with open(summary_json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n📊 Per-file summary with emotion saved to: {summary_json_path}")
    return summary

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Configuration
    DATASET_PATH = "vocal-to-score-demo/Input/GTSinger_sample_50"
    OUTPUT_DIR = "./pipeline1_output"
    MAX_FILES = 50
    
    # Create experiment config
    config = ExperimentConfig()
    
    # You can modify config parameters here
    # config.min_note_duration = 0.15
    # config.self_trans_prob = 0.6
    # etc.
    
    if os.path.exists(DATASET_PATH):
        summary = process_gtsinger_dataset(
            DATASET_PATH,
            OUTPUT_DIR,
            config,
            max_files=MAX_FILES
        )
        print(f"\n📊 Results saved to: {OUTPUT_DIR}")
    else:
        print(f"Error: {DATASET_PATH} not found")
        print("Please set DATASET_PATH to your GTSinger dataset location")


# Example usage:
# success = musicxml_to_wav_musescore('input.musicxml', 'output.wav')
# if success:
#     print('Conversion successful!')
# else:
#     print('Conversion failed.')