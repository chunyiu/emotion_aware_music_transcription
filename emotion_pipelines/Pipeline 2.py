"""
Pipeline 2: Librosa + HMM + Music21 + Transformer
Consolidated script to process GTSinger dataset samples
"""

import mido
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from music21 import stream, chord, note, converter, midi, tie, meter, duration, instrument
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
import pandas as pd
from pathlib import Path
import random

from music21 import converter as m21_converter  

def to_top2_schema(raw):
    """
    Convert emotion_classifier.predict() output:
      {top1_emotion, top1_confidence, top2_emotion, top2_confidence, ...}
    into the schema used by pipeline_3_summary / pipeline_4_summary:
      {top_emotion, top_confidence, second_emotion, second_confidence, top2: [...]}
    """
    if not raw:
        return None

    top1_emotion = raw.get("top1_emotion")
    top1_conf = raw.get("top1_confidence")
    top2_emotion = raw.get("top2_emotion")
    top2_conf = raw.get("top2_confidence")

    top2_list = []
    if top1_emotion is not None:
        top2_list.append({
            "emotion": top1_emotion,
            "confidence": top1_conf
        })
    if top2_emotion is not None:
        top2_list.append({
            "emotion": top2_emotion,
            "confidence": top2_conf
        })

    return {
        "top_emotion": top1_emotion,
        "top_confidence": top1_conf,
        "second_emotion": top2_emotion,
        "second_confidence": top2_conf,
        "top2": top2_list,
    }

def clamp_tiny_durations(score: stream.Score, min_ql=0.125):
    """
    Replace absurdly short durations (e.g. 256th/512th/1024th/2048th notes)
    with a minimum quarterLength and drop tuplets that create impossible
    durations for MusicXML export.

    IMPORTANT: operate on the entire score (score.recurse()), not just score.parts.
    """
    from music21 import duration as m21duration

    tiny_types = {"256th", "512th", "1024th", "2048th"}

    for el in score.recurse().notesAndRests:
        d = el.duration
        dur_type = getattr(d, "type", None)
        ql = float(getattr(d, "quarterLength", 0.0) or 0.0)

        if ql < min_ql or dur_type in tiny_types:
            # Kill tuplets that may be causing weird values
            if getattr(d, "tuplets", None):
                d.tuplets = ()

            # Completely replace with a fresh duration so type is reset
            el.duration = m21duration.Duration(min_ql)


# =============== EMOTION CLASSIFIER (copied from pipe1) ===============
import pickle
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

LEGAL_QL = (
    2.0,    # half
    1.0,    # quarter
    0.5,    # eighth
    0.25,   # 16th
    0.125,  # 32nd
    1.5,    # dotted half
    0.75,   # dotted quarter
    0.375,  # dotted 8th
)

def _nearest_legal_ql(ql: float, legal=LEGAL_QL):
    return min(legal, key=lambda x: abs(x - ql))

def _split_into_legal_segments(ql: float, legal=LEGAL_QL):
    """
    Split a quarterLength 'ql' into a sum of LEGAL_QL values, avoiding tiny
    leftover slivers that would become crazy tuplets.
    """
    chunks = []
    remaining = float(ql)
    eps = 1e-9

    while remaining > eps:
        # All legal values we can still fit
        candidates = [l for l in legal if l <= remaining + eps]

        if not candidates:
            # If remaining is very small, just merge it into the last chunk
            if chunks:
                chunks[-1] += remaining
            else:
                chunks.append(_nearest_legal_ql(remaining, legal))
            break

        l = max(candidates)

        # If remainder after taking 'l' would be tiny, just absorb it
        if remaining - l < eps * 10:
            chunks.append(l)
            break

        chunks.append(l)
        remaining -= l

    return chunks

def _retime_part_flat(part, denom_limit=8):
    from music21 import note, chord, duration, tie, stream
    from fractions import Fraction

    part_flat = part.flatten().sorted()
    cleaned = stream.Part()
    cleaned.id = part.id
    cleaned.partName = getattr(part, 'partName', None)

    cur_offset = 0.0
    min_ql = 0.125  # don't allow anything shorter than a 32nd here

    for el in part_flat.notesAndRests:
        raw_ql = float(el.duration.quarterLength)

        # Guard against zero / negative / absurd durations
        if raw_ql <= 0:
            raw_ql = 1.0 / denom_limit
        if raw_ql < min_ql:
            raw_ql = min_ql

        # Rationalise duration
        ql = float(Fraction(raw_ql).limit_denominator(denom_limit))

        segs = _split_into_legal_segments(ql, LEGAL_QL)

        def _clone_like(e):
            if isinstance(e, note.Note):
                c = note.Note(e.pitch)
                for ly in getattr(e, "lyrics", []) or []:
                    t = getattr(ly, "text", None)
                    if t:
                        c.addLyric(t)
                return c
            elif isinstance(e, chord.Chord):
                c = chord.Chord(e.pitches)
                for ly in getattr(e, "lyrics", []) or []:
                    t = getattr(ly, "text", None)
                    if t:
                        c.addLyric(t)
                return c
            else:
                return note.Rest()

        for i, seg in enumerate(segs):
            new_el = _clone_like(el)
            new_el.duration = duration.Duration(seg)

            cleaned.insert(cur_offset, new_el)

            if len(segs) > 1:
                if i == 0:
                    new_el.tie = tie.Tie("start")
                elif i == len(segs) - 1:
                    new_el.tie = tie.Tie("stop")
                else:
                    new_el.tie = tie.Tie("continue")

            cur_offset += seg

    return cleaned




def fix_measure_lengths(score: stream.Score, target_beats=4.0):
    """
    Make each measure sum to exactly target_beats (4.0 for 4/4).
    If too short -> pad with a rest.
    If too long  -> scale durations proportionally.
    """
    for part in score.parts:
        for m in part.getElementsByClass(stream.Measure):
            elems = list(m.notesAndRests)
            if not elems:
                continue

            total_ql = sum(e.duration.quarterLength for e in elems)

            if abs(total_ql - target_beats) < 1e-3:
                continue

            if total_ql < target_beats:
                r = note.Rest()
                r.duration.quarterLength = target_beats - total_ql
                m.append(r)
                continue

            factor = target_beats / total_ql
            for e in elems:
                e.duration.quarterLength *= factor


# --- EXPORT FUNCTION ------------------------------------------------------
def export_musicxml_safely(score: stream.Score, fp: str, ts="4/4", denom_limit=8):
    """
    Very conservative MusicXML export:
    - Use our own retiming (_retime_part_flat) which restricts durations
      to LEGAL_QL (min 0.125 quarterLength).
    - Avoid music21.makeNotation(), which can introduce tuplets such as 2048th notes.
    - Clamp any remaining tiny durations twice for safety.
    """
    cleaned_score = stream.Score()
    cleaned_score.metadata = getattr(score, "metadata", None)

    # 1) Retiming / quantising each part to LEGAL_QL
    for p in score.parts:
        cp = _retime_part_flat(p, denom_limit=denom_limit)
        # Ensure a time signature exists
        if not cp.recurse().getElementsByClass(meter.TimeSignature).first():
            cp.insert(0.0, meter.TimeSignature(ts))
        cleaned_score.insert(0.0, cp)

    # 2) Clamp tiny durations once, then make measures
    clamp_tiny_durations(cleaned_score, min_ql=0.125)   # 32nd note minimum
    cleaned_score.makeMeasures(inPlace=True)
    cleaned_score.sort()

    # 3) Clamp again in case measure splitting created small slivers
    clamp_tiny_durations(cleaned_score, min_ql=0.125)

        # 4) Final safety check: if anything is still absurd, crush it to 0.25
    tiny_types = {"256th", "512th", "1024th", "2048th"}
    for n in cleaned_score.recurse().notesAndRests:
        t = getattr(n.duration, "type", None)
        ql = float(getattr(n.duration, "quarterLength", 0.0) or 0.0)
        if t in tiny_types or ql <= 0 or ql < 0.03125:
            n.duration = duration.Duration(0.25)  # simple 16th

    # 5) Try to write the "nice" score first
    try:
        cleaned_score.write("musicxml", fp=fp)
        return cleaned_score
    except Exception as e:
        print(f"    [WARN] Primary MusicXML export failed: {e}")
        print("    [WARN] Falling back to ultra-simple durations for this file.")

        # --- FALLBACK SCORE: all notes become simple quarter notes ---
        fallback = stream.Score()
        fallback.metadata = getattr(cleaned_score, "metadata", None)

        for p in cleaned_score.parts:
            new_p = stream.Part()
            new_p.id = p.id
            new_p.partName = getattr(p, "partName", None)

            # keep instrument if present
            instr = p.getElementsByClass(instrument.Instrument).first()
            if instr is not None:
                new_p.insert(0.0, instr)

            # flatten notes/rests and give them quarter-note durations
            cur_offset = 0.0
            for el in p.flatten().notesAndRests:
                if el.isNote:
                    new_el = note.Note(el.pitch)
                elif el.isChord:
                    new_el = chord.Chord(el.pitches)
                else:
                    new_el = note.Rest()

                new_el.duration = duration.Duration(1.0)  # quarter note
                new_p.insert(cur_offset, new_el)
                cur_offset += 1.0  # next quarter

            fallback.insert(0.0, new_p)

        try:
            fallback.write("musicxml", fp=fp)
            print("    [INFO] Wrote fallback MusicXML with simplified durations.")
            return fallback
        except Exception as e2:
            print(f"    [ERROR] Fallback MusicXML export also failed: {e2}")
            # re-raise so the caller can see something went really wrong
            raise e2



# ============================================================================
# HARMONY GENERATION
# ============================================================================

try:
    import mingus.core.progressions as progressions
    import mingus.core.chords as m_chords
    import mingus.core.notes as m_notes
    from mingus.containers import NoteContainer
    MINGUS_AVAILABLE = True
except ImportError:
    MINGUS_AVAILABLE = False
    print("Warning: mingus not available, harmony generation will be skipped")

def generate_algorithmic_harmony(score: m21.stream.Score):
    """
    Generates a simple rule-based bassline harmony using mingus.
    """
    if not MINGUS_AVAILABLE:
        print("Skipping harmony generation (mingus not installed)")
        return score
        
    melody_part = score.getElementById('melody')
    if melody_part is None:
        print("Error: 'melody' part not found.")
        return score

    melody_part.makeMeasures(inPlace=True)

    print("Generating harmony using 'mingus' (algorithmic)...")
    harmony_part = stream.Part(id='harmony_algorithmic')
    harmony_part.insert(0, instrument.fromString('Acoustic Bass'))
    
    key = 'C' 

    for m in melody_part.getElementsByClass('Measure'):
        measure_notes = []
        
        for n in m.notes:
            if n.isNote:
                measure_notes.append(n.pitch.name)
            elif n.isChord:
                measure_notes.append(n.root().name)
        
        if not measure_notes:
            harmony_part.insert(m.offset, note.Rest(quarterLength=m.duration.quarterLength))
            continue
            
        try:
            determined_chord = m_chords.determine(measure_notes, shorthand=True)
            if not determined_chord:
                root_note_name = measure_notes[0]
            else:
                raw_chord_name = determined_chord[0]

            import re
            match = re.match(r'^([A-Ga-g][#b-]?)', raw_chord_name)
            if match:
                root_note_name = match.group(1).upper()
            else:
                root_note_name = measure_notes[0]
        except Exception:
            root_note_name = measure_notes[0]
        
        harmony_note = note.Note(f"{root_note_name}3")
        harmony_note.duration = m.duration
        
        harmony_part.insert(m.offset, harmony_note)

    score.insert(0, harmony_part)
    print("Algorithmic harmony generation complete.")
    return score

def extract_harmony_only_xml(full_xml_path, harmony_only_xml_path):
    """
    Extract only the harmony part(s) from a MusicXML file and save as a new XML.
    Returns True if a harmony part is found and written, else False.
    """
    try:
        score = m21_converter.parse(full_xml_path)
        harmony_score = stream.Score()
        harmony_parts = []

        parts = list(score.parts)

        # --- 1) Try by id / partName containing 'harm' ---
        for p in parts:
            pid = str(getattr(p, "id", "") or "").lower()
            pname = str(getattr(p, "partName", "") or "").lower()

            if "harm" in pid or "harm" in pname:
                harmony_parts.append(p)

        # --- 2) Fallback: assume any non-first part is harmony ---
        if not harmony_parts and len(parts) >= 2:
            # In your pipeline: P0 = melody, P1 = harmony (Acoustic Bass)
            harmony_parts = parts[1:]
            print("    [HARMONY] No explicit 'harm' id/name; using all non-first parts as harmony.")

        if not harmony_parts:
            print("    [HARMONY] No harmony part found in score.")
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
    """musicxml_output
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

def process_single_file(audio_path, json_path, config, output_midi=None, output_musicxml=None):
    """Process a single audio file through the pipeline"""
    print(f"🎵 Processing: {audio_path}")

    # -----------------------------
    # 1. Load audio
    # -----------------------------
    y, sr = librosa.load(audio_path, sr=44100)

    # Guard: audio too short for pYIN / frame_length
    if len(y) < config.pyin_frame_length:
        print(f"    ✗ Skipping {audio_path} (audio too short for pYIN: {len(y)} samples)")
        metrics = {"f1": 0.0, "precision": 0.0, "recall": 0.0}
        detected_notes = []
        return metrics, detected_notes

    # -----------------------------
    # 2. F0 extraction + Viterbi (with safety)
    # -----------------------------
    try:
        # pYIN
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=config.pyin_fmin,
            fmax=config.pyin_fmax,
            sr=sr,
            frame_length=config.pyin_frame_length,
            hop_length=config.pyin_hop_length,
        )
        times = librosa.times_like(f0, sr=sr, hop_length=config.pyin_hop_length)

        # Viterbi decoding
        converter = VocalToMIDIConverter_Viterbi(config)
        smoothed_midi = converter.viterbi_decode(f0, voiced_probs)

        # Median filter smoothing
        smoothed_midi = scipy.signal.medfilt(
            smoothed_midi, kernel_size=config.median_kernel_size
        )

    except Exception as e:
        # This is where your numba / _viterbi crash was coming from
        print(f"    ✗ Error: pYIN/Viterbi failed for {audio_path}: {e}")
        metrics = {"f1": 0.0, "precision": 0.0, "recall": 0.0}
        detected_notes = []
        return metrics, detected_notes

    # -----------------------------
    # 3. Note detection + evaluation
    # -----------------------------
    detected_notes = improved_midi_to_note_events(times, smoothed_midi, config)
    detected_notes = post_process_notes(detected_notes, config)

    evaluator = VocalMIDIEvaluator(config)
    metrics = evaluator.evaluate(detected_notes, json_path)

    # -----------------------------
    # 4. Score generation + annotations
    # -----------------------------
    generated_score = notes_to_score_v2(detected_notes, y, sr)

    if os.path.exists(json_path):
        annotations = load_annotations_from_json(json_path)
        generated_score = add_annotations_to_score(generated_score, annotations)

    # -----------------------------
    # 5. Harmony + export
    # -----------------------------
    score_with_harmony = generate_algorithmic_harmony(generated_score)

    if output_musicxml:
        os.makedirs(os.path.dirname(output_musicxml), exist_ok=True)
        try:
            export_musicxml_safely(
                score_with_harmony, fp=output_musicxml, ts="4/4", denom_limit=8
            )
            print(f"    ✅ Exported MusicXML to {output_musicxml}")
        except Exception as e:
            # If export explodes (e.g., some weird duration sneaks through),
            # don't kill the whole dataset — just log it.
            print(f"    ✗ Error: MusicXML export failed for {output_musicxml}: {e}")

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
    
    # Load emotion classifier (same as pipe1)
    emotion_model_dir = Path('results/emotion_model')
    emotion_classifier = EmotionClassifier(emotion_model_dir)

    emotion_summary = []

    for i, audio_file in enumerate(audio_files, 1):
        print(f"\n[{i}/{len(audio_files)}] 🎵 {audio_file.name}")
        print(f"🎵 Processing: {audio_file.relative_to(dataset_path)}")
        rel_path = audio_file.relative_to(dataset_path)
        output_subdir = output_dir / rel_path.parent
        output_subdir.mkdir(parents=True, exist_ok=True)
        musicxml_output = output_subdir / f"{audio_file.stem}.musicxml"
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
                    # Use MuseScore CLI for MusicXML-to-WAV (as in pipe1)
                    def musicxml_to_wav_musescore(
                        musicxml_path,
                        output_wav_path,
                        musescore_exe=r"C:\Program Files\MuseScore 3\bin\MuseScore3.exe"
                    ):
                        import subprocess, tempfile, shutil, os

                        try:
                            with tempfile.TemporaryDirectory() as tmpdir:
                                tmp_musicxml = os.path.join(tmpdir, "score.musicxml")
                                tmp_wav = os.path.join(tmpdir, "score.wav")

                                shutil.copy2(musicxml_path, tmp_musicxml)

                                cmd = [musescore_exe, "-o", tmp_wav, tmp_musicxml]
                                print("    [DEBUG] Running MuseScore CMD:", " ".join(cmd))
                                print("    [DEBUG] Temp MusicXML:", tmp_musicxml)
                                print("    [DEBUG] Temp WAV target:", tmp_wav)

                                result = subprocess.run(cmd, capture_output=True, text=True)
                                print("    [DEBUG] MuseScore return code:", result.returncode)

                                if result.returncode != 0:
                                    print("[ERROR] MuseScore failed with return code:", result.returncode)
                                    print("[ERROR] MuseScore stderr:\n", result.stderr)
                                    print("[ERROR] MuseScore stdout:\n", result.stdout)
                                    return False

                                if not os.path.exists(tmp_wav):
                                    print("[ERROR] MuseScore reported success but temp WAV not found:", tmp_wav)
                                    return False

                                os.makedirs(os.path.dirname(output_wav_path), exist_ok=True)
                                shutil.copy2(tmp_wav, output_wav_path)
                                return True

                        except Exception as e:
                            print(f"[ERROR] Exception during MuseScore conversion: {e}")
                            return False

                    print(f"    [INFO] Converting MusicXML to WAV (MuseScore): {musicxml_output} -> {wav_after_output}")
                    success = musicxml_to_wav_musescore(str(musicxml_output), str(wav_after_output))

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

                # ------------------------------------------------------
                # 1) Per-file BEFORE/AFTER emotion JSON (unchanged)
                # ------------------------------------------------------
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

                # ------------------------------------------------------
                # 2) HARMONY-ONLY EMOTION COMPARISON
                # ------------------------------------------------------
                harmony_only_xml = output_subdir / f"{audio_file.stem}_harmony_only.musicxml"
                harmony_only_wav = output_subdir / f"{audio_file.stem}_harmony_only.wav"
                harmony_emotion = None
                success_harmony = False

                try:
                    ok_harm_xml = extract_harmony_only_xml(musicxml_output, harmony_only_xml)
                    if ok_harm_xml:
                        print(f"    [HARMONY] Converting harmony-only XML to WAV: {harmony_only_xml} -> {harmony_only_wav}")
                        success_harmony = musicxml_to_wav_musescore(
                            str(harmony_only_xml),
                            str(harmony_only_wav)
                        )

                        if success_harmony and harmony_only_wav.exists():
                            harmony_emotion = emotion_classifier.predict(str(harmony_only_wav))
                            if harmony_emotion:
                                print(
                                    f"    [HARMONY] Top1: {harmony_emotion['top1_emotion']} "
                                    f"({harmony_emotion['top1_confidence']:.1%})"
                                )
                            else:
                                success_harmony = False
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

                # ------------------------------------------------------
                # 3) Summary entry for pipe2_summary.json
                # ------------------------------------------------------
                harmony_match_with_before = (
                    before_emotion is not None
                    and harmony_emotion is not None
                    and before_emotion.get("top1_emotion") == harmony_emotion.get("top1_emotion")
                )

                summary_entry = {
                        "file": audio_file.name,
                        "notes": len(notes),

                        # we don't currently compute key; keep schema consistent but set None
                        "key": None,

                        "musicxml": str(musicxml_output.relative_to(output_dir)),
                        "wav_after": str(wav_after_output.relative_to(output_dir)),

                        # convert to the same schema as pipeline_3 / pipeline_4
                        "emotion_before": to_top2_schema(before_emotion),
                        "emotion_after": to_top2_schema(after_emotion),
                        "harmony_emotion": to_top2_schema(harmony_emotion),

                        # flag name aligned with pipeline_3 / pipeline_4
                        "harmony_matches_original": harmony_match_with_before,

                        # optional: paths to harmony-only files (same meaning as pipeline_4)
                        "harmony_xml": (
                            str(harmony_only_xml.relative_to(output_dir))
                            if harmony_only_xml.exists() else None
                        ),
                        "harmony_wav": (
                            str(harmony_only_wav.relative_to(output_dir))
                            if harmony_only_wav.exists() else None
                        ),

                        # note + success flag
                        "note": "Includes AFTER (melody+harmony) and harmony-only emotion analysis.",
                        "success": bool(success and after_emotion is not None),
                    }


                emotion_summary.append(summary_entry)

                # ------------------------------------------------------
                # 4) Collect results for note F1/metrics
                # ------------------------------------------------------
                result = {
                    'file': str(rel_path),
                    'musicxml_output': str(musicxml_output.relative_to(output_dir)),
                    **metrics
                }
                all_results.append(result)




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

                # --- Collect results ---
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
                        success_harmony = musicxml_to_wav_musescore(str(harmony_only_xml),
                                                                     str(harmony_only_wav))

                        if success_harmony and harmony_only_wav.exists():
                            harmony_emotion = emotion_classifier.predict(str(harmony_only_wav))
                            if harmony_emotion:
                                print(
                                    f"    [HARMONY] Top1: {harmony_emotion['top1_emotion']} "
                                    f"({harmony_emotion['top1_confidence']:.1%})"
                                )
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




                result = {
                    'file': str(rel_path),
                    'musicxml_output': str(musicxml_output.relative_to(output_dir)),
                    **metrics
                }
                all_results.append(result)
            else:
                result = {
                    'file': str(rel_path),
                    'status': 'no_ground_truth'
                }
                all_results.append(result)
        except Exception as e:
            print(f"    ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            result = {'file': str(rel_path), 'status': 'error', 'error': str(e)}
            all_results.append(result)
    
    # Compute averages
    evaluated_results = [r for r in all_results if 'f_measure' in r]
    avg_metrics = None
    df = None

    if evaluated_results:
        avg_metrics = {
            'avg_f_measure': np.mean([r['f_measure'] for r in evaluated_results]),
            'avg_precision': np.mean([r['precision'] for r in evaluated_results]),
            'avg_recall': np.mean([r['recall'] for r in evaluated_results]),
            'total_evaluated': len(evaluated_results)
        }

        print("\n" + "="*70)
        print("AVERAGE METRICS")
        print("="*70)
        for key, value in avg_metrics.items():
            if 'total' not in key:
                print(f"{key:30s}: {value:.4f}")
            else:
                print(f"{key:30s}: {value}")
        print("="*70)

        with open(output_dir / "average_metrics.json", 'w') as f:
            json.dump(avg_metrics, f, indent=2)

        df = pd.DataFrame(evaluated_results)
        df.to_csv(output_dir / "evaluation_results.csv", index=False)
    else:
        print("\nNo files were successfully evaluated")

    # ---------- NEW: write pipe2_summary.json ----------
    pipe2_summary_path = output_dir / "pipe2_summary.json"

    total_files = len(audio_files)
    successful = sum(1 for s in emotion_summary if s.get("success"))
    failed = total_files - successful

    # If you want to record a key, you can later plug in a real estimate here
    # For now we keep it simple and only add a descriptive note.
    for entry in emotion_summary:
        # Optional: add a note like in pipeline_3 / pipeline_4
        entry.setdefault(
            "note",
            "Includes AFTER (melody+harmony) and harmony-only emotion analysis generated using mingus."
        )
        # Optional: if you have a key string, you can attach it:
        # entry.setdefault("key", detected_key_string)

    total_files = len(audio_files)
    successful = sum(1 for s in emotion_summary if s.get("success"))
    failed = total_files - successful

    summary_payload = {
        "pipeline": "Pipeline 2: Librosa + HMM + Music21 + mingus harmony",
        "total_files": total_files,
        "successful": successful,
        "failed": failed,
        "dataset_path": str(dataset_path),
        "results": emotion_summary,
    }

    with open(pipe2_summary_path, "w") as f:
        json.dump(summary_payload, f, indent=2)
    print(f"\n[SUMMARY] Wrote harmony summary to: {pipe2_summary_path}")


    # final return (keep the same external API)
    return df, avg_metrics
    
    

# ============================================================================
# MAIN EXECUTION
# ============================================================================



if __name__ == "__main__":
    # Configuration
    DATASET_PATH = "vocal-to-score-demo/Input/GTSinger_sample_50"
    OUTPUT_DIR = "./pipeline2_output"
    MAX_FILES = 50
    
    # Create experiment config
    config = ExperimentConfig()
    
    # You can modify config parameters here
    # config.min_note_duration = 0.15
    # config.self_trans_prob = 0.6
    # etc.
    
    if os.path.exists(DATASET_PATH):
        df_results, avg_metrics = process_gtsinger_dataset(
            DATASET_PATH,
            OUTPUT_DIR,
            config,
            max_files=MAX_FILES
        )
        
        if df_results is not None:
            print(f"\n📊 Results saved to: {OUTPUT_DIR}")
            print("\nTop 10 results:")
            print(df_results.head(10))
    else:
        print(f"Error: {DATASET_PATH} not found")
        print("Please set DATASET_PATH to your GTSinger dataset location")