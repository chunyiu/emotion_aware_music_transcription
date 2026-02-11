import librosa
import numpy as np
import pretty_midi
from pathlib import Path
import json
from typing import Dict, Tuple, List
import mir_eval
import os
from IPython.display import display, Audio, HTML
import pandas as pd
import scipy.signal
import random


# ============================================================================
# EXPERIMENT PARAMETERS - Modify these for experiments
# ============================================================================
class ExperimentConfig:
    """Central configuration for tunable parameters"""
    
    # Note Detection Parameters
    min_note_duration: float = 0.5      # Minimum note length (seconds)
    pitch_threshold: float = 0.6        # Semitones - pitch change detection
    vibrato_tolerance: float = 0.5      # Semitones - vibrato handling
    
    # Post-processing Parameters
    merge_threshold: float = 0.7       # Semitones - merge similar pitches
    min_gap: float = 0.12              # Seconds - minimum silence between notes
    
    # Signal Processing Parameters
    median_kernel_size: int = 7         # Median filter kernel size (odd number)
    
    # Evaluation Parameters
    onset_tolerance: float = 0.05          # Seconds - onset matching tolerance
    pitch_tolerance: float = 50        # Cents - pitch matching tolerance
    
    # pYIN Parameters
    pyin_fmin: float = 65.4            # Hz - minimum frequency
    pyin_fmax: float = 1047.0          # Hz - maximum frequency
    pyin_frame_length: int = 2048      # Samples
    pyin_hop_length: int = 256         # Samples
    
    # Viterbi HMM Parameters
    self_trans_prob: float = 0.94      # Stay in same pitch state
    neighbor_prob: float = 0.02        # Move to adjacent pitch
    unvoiced_stay: float = 0.5         # Stay unvoiced
    
    @classmethod
    def to_dict(cls):
        """Export config as dictionary"""
        return {k: v for k, v in cls.__dict__.items() 
                if not k.startswith('_') and not callable(v)}


# ============================================================================
# CORE ALGORITHMS
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
    """Merge over-segmented notes and remove glitches"""
    if len(notes) <= 1:
        return notes
    
    processed = []
    current_note = notes[0].copy()
    
    for next_note in notes[1:]:
        pitch_diff = abs(next_note['pitch'] - current_note['pitch'])
        time_gap = next_note['start'] - current_note['end']
        
        should_merge = (
            (pitch_diff <= config.merge_threshold and time_gap <= config.min_gap) or
            (pitch_diff <= config.merge_threshold and time_gap <= 0)
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
# VITERBI CONVERTER
# ============================================================================

class VocalToMIDIConverter_Viterbi:
    def __init__(self, config: ExperimentConfig, 
                 fmin=librosa.note_to_hz("C2"), 
                 fmax=librosa.note_to_hz("C7")):
        self.config = config
        self.fmin = fmin
        self.fmax = fmax
        self.midi_min = int(librosa.hz_to_midi(fmin))
        self.midi_max = int(librosa.hz_to_midi(fmax))
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
            hop_length=cfg.pyin_hop_length,
            win_length=cfg.pyin_frame_length
        )
        
        times = librosa.times_like(f0, sr=sr, hop_length=cfg.pyin_hop_length)
        voiced_ratio = np.sum(voiced_flag) / len(voiced_flag)
        print(f"    🎶 Voiced frame ratio: {voiced_ratio*100:.1f}%")
        
        return times, f0, voiced_probs

    def viterbi_decode(self, f0, voiced_probs):
        """Viterbi decoding with log probabilities"""
        n_frames = len(f0)
        n_states = self.n_states + 1
        
        log_A = np.log(self.transition_matrix + 1e-12)
        log_pi = np.log(self.start_prob + 1e-12)
        
        log_delta = np.zeros((n_frames, n_states))
        psi = np.zeros((n_frames, n_states), dtype=int)
        
        obs_midi = np.zeros_like(f0)
        valid_mask = (f0 > 0) & np.isfinite(f0)
        obs_midi[valid_mask] = librosa.hz_to_midi(f0[valid_mask])
        
        # Initialize
        log_delta[0] = log_pi + np.log(
            self.compute_emission_probabilities(obs_midi[0], voiced_probs[0]) + 1e-12
        )
        
        # Forward pass
        for t in range(1, n_frames):
            emission = np.log(
                self.compute_emission_probabilities(obs_midi[t], voiced_probs[t]) + 1e-12
            )
            for j in range(n_states):
                seq_probs = log_delta[t - 1] + log_A[:, j]
                psi[t, j] = np.argmax(seq_probs)
                log_delta[t, j] = np.max(seq_probs) + emission[j]
        
        # Backtrack
        states = np.zeros(n_frames, dtype=int)
        states[-1] = np.argmax(log_delta[-1])
        for t in range(n_frames - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]
        
        smoothed_midi = np.array(
            [self.midi_notes[s] if s < self.n_states else 0 for s in states]
        )
        
        return smoothed_midi

    def note_events_to_pretty_midi(self, notes, output_path="output.mid"):
        """Convert note events to MIDI file"""
        midi = pretty_midi.PrettyMIDI()
        inst = pretty_midi.Instrument(program=0)
        for n in notes:
            inst.notes.append(pretty_midi.Note(
                velocity=100,
                pitch=int(round(n["pitch"])),
                start=n["start"],
                end=n["end"]
            ))
        midi.instruments.append(inst)
        midi.write(output_path)

    def process_file(self, audio_path, output_midi="output.mid"):
        """Complete processing pipeline"""
        print(f"🎵 Processing: {audio_path}")
        times, f0, voiced_probs = self.extract_pitch_features(audio_path)
        
        print(f"Running Viterbi decoding on {len(f0)} frames...")
        smoothed_midi = self.viterbi_decode(f0, voiced_probs)
        smoothed_midi = scipy.signal.medfilt(
            smoothed_midi, 
            kernel_size=self.config.median_kernel_size
        )
        
        notes = improved_midi_to_note_events(times, smoothed_midi, self.config)
        notes = post_process_notes(notes, self.config)
        
        print(f"Predicted {len(notes)} notes")
        for n in notes[:5]:
            dur = n['end'] - n['start']
            print(f"    Note: pitch={n['pitch']:.1f}, dur={dur:.3f}s")
        
        self.note_events_to_pretty_midi(notes, output_midi)
        print(f"Saved MIDI to {output_midi}")
        
        return times, f0, smoothed_midi, notes


# ============================================================================
# EVALUATOR
# ============================================================================
import csv

def save_experiment_log(config: ExperimentConfig, avg_metrics: Dict, csv_path="experiment_log.csv"):
    import csv, os
    
    # Extract config attributes safely, including annotated ones
    config_dict = {
        k: getattr(config, k)
        for k in dir(config)
        if not k.startswith("_") and not callable(getattr(config, k))
    }

    config_dict["experiment_name"] = getattr(config, "experiment_name", "unvoiced stay")

    # Merge config + metrics
    row = {**config_dict, **avg_metrics}

    # Define base column ordering
    field_order = [
        "experiment_name","min_note_duration","pitch_threshold","vibrato_tolerance",
        "merge_threshold","min_gap","median_kernel_size","onset_tolerance",
        "pitch_tolerance","pyin_fmin","pyin_fmax","pyin_frame_length","pyin_hop_length",
        "self_trans_prob","neighbor_prob","unvoiced_stay",
        "avg_f_measure","avg_precision","avg_recall","avg_raw_pitch_accuracy","avg_raw_chroma_accuracy",
        "avg_overall_accuracy","avg_voicing_recall","avg_voicing_false_alarm"
        
    ]
    # Append new keys automatically
    for key in row.keys():
        if key not in field_order:
            field_order.append(key)

    # Write CSV
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=field_order)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"Experiment logged to {csv_path}")

class VocalMIDIEvaluator:
    """Evaluate vocal to MIDI conversion"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
    
    def load_ground_truth_from_json(self, json_path: str):
        """Load ground truth from GTSinger JSON format"""
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            ref_intervals = []
            ref_pitches = []
            
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
    
    def predictions_to_intervals(self, times, freqs, min_duration=0.05):
        """Convert frame-wise predictions to note intervals"""
        if len(times) == 0 or len(freqs) == 0:
            return np.array([]), np.array([])
        
        midi_seq = np.zeros_like(freqs)
        voiced_mask = freqs > 0
        if np.any(voiced_mask):
            midi_seq[voiced_mask] = librosa.hz_to_midi(freqs[voiced_mask])
        
        pred_intervals = []
        pred_pitches = []
        
        current_pitch = midi_seq[0]
        start_time = times[0]
        
        for i in range(1, len(midi_seq)):
            pitch_diff = abs(midi_seq[i] - current_pitch)
            
            if pitch_diff > 0.5:
                end_time = times[i]
                duration = end_time - start_time
                
                if current_pitch > 0 and duration >= min_duration:
                    pred_intervals.append([start_time, end_time])
                    pred_pitches.append(round(current_pitch))
                
                current_pitch = midi_seq[i]
                start_time = times[i]
        
        # Final note
        if len(times) > 1:
            frame_duration = times[-1] - times[-2]
            end_time = times[-1] + frame_duration
        else:
            end_time = times[-1]
        
        duration = end_time - start_time
        if current_pitch > 0 and duration >= min_duration:
            pred_intervals.append([start_time, end_time])
            pred_pitches.append(round(current_pitch))
        
        return np.array(pred_intervals), np.array(pred_pitches)
    
    def intervals_to_frames(self, intervals, pitches, times):
        """Convert note intervals to frame-level representation"""
        freqs = np.zeros_like(times)
        
        for (start, end), pitch in zip(intervals, pitches):
            mask = (times >= start) & (times < end)
            freqs[mask] = librosa.midi_to_hz(pitch)
        
        return freqs
    
    def evaluate(self, est_times, est_freqs, json_path):
        """Evaluate predictions against JSON ground truth"""
        ref_intervals, ref_pitches = self.load_ground_truth_from_json(json_path)
        
        if len(ref_pitches) == 0:
            return self._empty_metrics()
        
        pred_intervals, pred_pitches = self.predictions_to_intervals(est_times, est_freqs)
        print(f"Ground truth notes: {len(ref_pitches)}")
        print(f"Predicted notes: {len(pred_pitches)}")
        
        if len(pred_pitches) == 0:
            return self._empty_metrics(ref_count=len(ref_pitches))
        
        # Note-level metrics
        try:
            precision, recall, f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
                ref_intervals=ref_intervals,
                ref_pitches=ref_pitches,
                est_intervals=pred_intervals,
                est_pitches=pred_pitches,
                onset_tolerance=self.config.onset_tolerance,
                pitch_tolerance=self.config.pitch_tolerance / 100.0,
                offset_ratio=None
            )
        except Exception as e:
            print(f"Note-level evaluation error: {e}")
            precision = recall = f1 = 0.0
        
        # Frame-level metrics
        ref_freqs_frame = self.intervals_to_frames(ref_intervals, ref_pitches, est_times)
        ref_voicing = (ref_freqs_frame > 0).astype(float)
        est_voicing = (est_freqs > 0).astype(float)
        
        try:
            rpa = mir_eval.melody.raw_pitch_accuracy(
                ref_voicing, ref_freqs_frame,
                est_voicing, est_freqs,
                cent_tolerance=self.config.pitch_tolerance
            )
            rca = mir_eval.melody.raw_chroma_accuracy(
                ref_voicing, ref_freqs_frame,
                est_voicing, est_freqs,
                cent_tolerance=self.config.pitch_tolerance
            )
            oa = mir_eval.melody.overall_accuracy(
                ref_voicing, ref_freqs_frame,
                est_voicing, est_freqs,
                cent_tolerance=self.config.pitch_tolerance
            )
            vr = mir_eval.melody.voicing_recall(ref_voicing, est_voicing)
            vfa = mir_eval.melody.voicing_false_alarm(ref_voicing, est_voicing)
        except Exception as e:
            print(f"Frame-level evaluation error: {e}")
            rpa = rca = oa = vr = vfa = 0.0
        
        return {
            'precision': float(precision),
            'recall': float(recall),
            'f_measure': float(f1),
            'ref_count': int(len(ref_pitches)),
            'pred_count': int(len(pred_pitches)),
            'raw_pitch_accuracy': float(rpa),
            'raw_chroma_accuracy': float(rca),
            'overall_accuracy': float(oa),
            'voicing_recall': float(vr),
            'voicing_false_alarm': float(vfa)
        }
    
    def _empty_metrics(self, ref_count=0, pred_count=0):
        """Return empty metrics dict"""
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f_measure': 0.0,
            'ref_count': ref_count,
            'pred_count': pred_count,
            'raw_pitch_accuracy': 0.0,
            'raw_chroma_accuracy': 0.0,
            'overall_accuracy': 0.0,
            'voicing_recall': 0.0,
            'voicing_false_alarm': 0.0
        }


# ============================================================================
# BATCH PROCESSING
# ============================================================================

def process_gtsinger_dataset(english_folder_path: str, 
                             output_dir: str,
                             config: ExperimentConfig,
                             max_files: int = None,
                             audio_files: list = None):
    """Process GTSinger English dataset"""
    english_path = Path(english_folder_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    converter = VocalToMIDIConverter_Viterbi(config)
    evaluator = VocalMIDIEvaluator(config)
    
    if audio_files is None:
        audio_files = list(english_path.rglob("*.wav"))
        audio_files = [f for f in audio_files if "Paired_Speech_Group" not in f.parts]
    
    print(f"\n✓ Found {len(audio_files)} audio files")
    
    if max_files:
        audio_files = audio_files[:max_files]
        print(f"  Processing first {max_files} files...")
    
    all_results = []
    
    for idx, audio_file in enumerate(audio_files, 1):
        print(f"\n[{idx}/{len(audio_files)}] 🎵 {audio_file.name}")
        
        rel_path = audio_file.relative_to(english_path)
        output_subdir = output_dir / rel_path.parent
        output_subdir.mkdir(parents=True, exist_ok=True)
        
        midi_output = output_subdir / f"{audio_file.stem}.mid"
        
        try:
            times, f0, smoothed_midi, notes = converter.process_file(
                str(audio_file),
                output_midi=str(midi_output)
            )

            print(f'notes: {notes}')
            
            est_freqs = np.zeros_like(smoothed_midi, dtype=float)
            voiced_mask = smoothed_midi > 0
            if np.any(voiced_mask):
                est_freqs[voiced_mask] = librosa.midi_to_hz(smoothed_midi[voiced_mask])
            
            json_path = audio_file.with_suffix('.json')
            
            if json_path.exists():
                metrics = evaluator.evaluate(times, est_freqs, str(json_path))
                
                result = {
                    'file': str(rel_path),
                    'midi_output': str(midi_output.relative_to(output_dir)),
                    **metrics
                }
                all_results.append(result)
                
                print(f"    ✓ F1: {metrics['f_measure']:.3f} | "
                      f"P: {metrics['precision']:.3f} | "
                      f"R: {metrics['recall']:.3f}")
            else:
                result = {
                    'file': str(rel_path),
                    'status': 'no_ground_truth'
                }
                all_results.append(result)
                
        except Exception as e:
            print(f"    ✗ Error: {e}")
            result = {'file': str(rel_path), 'status': 'error', 'error': str(e)}
            all_results.append(result)
    
    # # Save results
    # results_file = output_dir / "evaluation_results.json"
    # with open(results_file, 'w') as f:
    #     json.dump(all_results, f, indent=2)
    
    # # Save config
    # config_file = output_dir / "experiment_config.json"
    # with open(config_file, 'w') as f:
    #     json.dump(config.to_dict(), f, indent=2)
    
    # Compute averages
    evaluated_results = [r for r in all_results if 'f_measure' in r]
    print(evaluated_results)
    if evaluated_results:
        avg_metrics = {
            'avg_f_measure': np.mean([r['f_measure'] for r in evaluated_results]),
            'avg_precision': np.mean([r['precision'] for r in evaluated_results]),
            'avg_recall': np.mean([r['recall'] for r in evaluated_results]),
            'avg_raw_pitch_accuracy': np.mean([r['raw_pitch_accuracy'] for r in evaluated_results]),
            'avg_raw_chroma_accuracy': np.mean([r['raw_chroma_accuracy'] for r in evaluated_results]),
            'avg_overall_accuracy': np.mean([r['overall_accuracy'] for r in evaluated_results]),
            'avg_voicing_recall': np.mean([r['voicing_recall'] for r in evaluated_results]),
            'avg_voicing_false_alarm': np.mean([r['voicing_false_alarm'] for r in evaluated_results]),
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
        
        return df, avg_metrics
    else:
        print("\nNo files were successfully evaluated")
        return None, None


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Configuration
    EXTRACT_TO = "./GTSinger_sample_50"
    OUTPUT_DIR = "./vocal_midi_output"
    MAX_FILES = 50
    
    # Create experiment config
    config = ExperimentConfig()
    
    if os.path.exists(EXTRACT_TO):
        english_folder = Path(EXTRACT_TO)
        print(f"\n✓ Using folder: {english_folder}")
        
        audio_files = [f for f in english_folder.rglob("*.wav")
                      if "Paired_Speech_Group" not in f.parts]
        print(f"✓ Found {len(audio_files)} WAV files")
        
        if MAX_FILES and len(audio_files) > MAX_FILES:
            audio_files = random.sample(audio_files, MAX_FILES)
        
        df_results, avg_metrics = process_gtsinger_dataset(
            str(english_folder),
            OUTPUT_DIR,
            config,
            max_files=MAX_FILES,
            audio_files=audio_files
        )
        
        if df_results is not None:
            print("\n Results saved to:", OUTPUT_DIR)
            display(df_results.head())
            # print(config)
            save_experiment_log(config, avg_metrics, csv_path="experiment_log.csv")
    else:
        print(f"Error: {EXTRACT_TO} not found")