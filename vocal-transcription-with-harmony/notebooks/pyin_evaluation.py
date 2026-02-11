import librosa
import numpy as np
import pretty_midi
from pathlib import Path
import json
from typing import Dict, Tuple, List
import mir_eval
import os
from IPython.display import display
import pandas as pd
import mir_eval.transcription
import scipy.signal
import matplotlib.pyplot as plt
import music21
#%matplotlib inline

class VocalToMIDIConverter_PYIN:
    """PYIN-based vocal to MIDI converter (from your original pipeline)"""
    
    def __init__(self, sr=16000, hop_length=512, fmin=65.4, fmax=2093):
        self.sr = sr
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax
    
    def extract_pitch_pyin(self, audio_path):
        """Extract pitch using PYIN algorithm from librosa."""
        y, sr = librosa.load(audio_path, sr=self.sr)
        
        # PYIN pitch detection
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
        """Convert pitch contour to discrete notes with onset/offset detection."""
        notes = []
        current_note = None
        min_frames = int(min_note_duration * self.sr / self.hop_length)
        
        for i, (pitch, is_voiced) in enumerate(zip(f0, voiced_flag)):
            time = times[i]
            
            if is_voiced and not np.isnan(pitch):
                midi_pitch = librosa.hz_to_midi(pitch)
                
                if current_note is None:
                    # Start new note
                    current_note = {
                        'start': time,
                        'pitch_sum': midi_pitch,
                        'pitch_count': 1,
                        'start_frame': i
                    }
                else:
                    # Continue current note
                    current_note['pitch_sum'] += midi_pitch
                    current_note['pitch_count'] += 1
            else:
                # End current note if it exists
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
        
        # Handle last note
        if current_note is not None:
            time = times[-1]
            avg_pitch = current_note['pitch_sum'] / current_note['pitch_count']
            notes.append({
                'start': current_note['start'],
                'end': time,
                'pitch': int(round(avg_pitch))
            })
        
        return notes
    
    def create_midi_from_notes(self, notes, output_path, tempo_bpm=120):
        """Create MIDI file from detected notes."""
        midi = pretty_midi.PrettyMIDI(initial_tempo=tempo_bpm)
        instrument = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano
        
        for note in notes:
            midi_note = pretty_midi.Note(
                velocity=80,
                pitch=note['pitch'],
                start=note['start'],
                end=note['end']
            )
            instrument.notes.append(midi_note)
        
        midi.instruments.append(instrument)
        midi.write(output_path)
        print(f"✅ MIDI saved to: {output_path}")
        
        return midi
    
    def process_file(self, audio_path, output_midi="output.mid", plot=False):
        """Full processing pipeline"""
        print(f"🎵 Processing: {audio_path}")
        
        # Extract pitch using PYIN
        times, f0, voiced_flag, voiced_probs, y, sr = self.extract_pitch_pyin(audio_path)
        print(f"✓ Pitch extracted: {len(f0)} frames")
        
        # Detect notes
        notes = self.detect_notes_from_pitch(f0, voiced_flag, times)
        print(f"✓ Detected {len(notes)} notes")
        
        # Create MIDI
        midi = self.create_midi_from_notes(notes, output_midi)
        
        # Convert to frequency array for evaluation
        est_freqs = np.zeros_like(f0)
        for i, (pitch, is_voiced) in enumerate(zip(f0, voiced_flag)):
            if is_voiced and not np.isnan(pitch):
                est_freqs[i] = pitch
        
        if plot:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(14, 5))
            
            # Plot PYIN output
            valid_f0 = f0.copy()
            valid_f0[~voiced_flag] = np.nan
            plt.plot(times, librosa.hz_to_midi(valid_f0),
                     'o', alpha=0.3, markersize=2, label="PYIN output")
            
            # Plot detected notes
            for note in notes:
                plt.plot([note['start'], note['end']], 
                        [note['pitch'], note['pitch']], 
                        linewidth=2, label="Detected notes" if note == notes[0] else "")
            
            plt.legend()
            plt.xlabel("Time (s)")
            plt.ylabel("MIDI pitch")
            plt.title("PYIN Pitch Tracking")
            plt.grid(alpha=0.3)
            plt.show()
        
        return times, est_freqs, notes
    
    def convert(self, audio_path: str, output_midi_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Backwards-compatible wrapper. Returns: (times, frequencies_in_Hz)"""
        times, est_freqs, notes = self.process_file(
            audio_path, output_midi=output_midi_path, plot=False
        )
        return times, est_freqs
    
class VocalMIDIEvaluator_PYIN_JSON:
    """Evaluate PYIN-based vocal to MIDI conversion using note-level and frame-level metrics."""
    
    def __init__(self, onset_tolerance=0.05, pitch_tolerance=50):
        self.onset_tolerance = onset_tolerance
        self.pitch_tolerance = pitch_tolerance
    
    def load_ground_truth_from_json(self, json_path: str):
        """Load ground truth from GTSinger JSON format."""
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
        """Convert frame-wise predictions to note intervals."""
        midi_seq = np.zeros_like(freqs)
        voiced_mask = freqs > 0
        if np.any(voiced_mask):
            midi_seq[voiced_mask] = librosa.hz_to_midi(freqs[voiced_mask])
        
        pred_intervals = []
        pred_pitches = []
        
        if len(midi_seq) == 0:
            return np.array(pred_intervals), np.array(pred_pitches)
        
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
        
        end_time = times[-1]
        duration = end_time - start_time
        if current_pitch > 0 and duration >= min_duration:
            pred_intervals.append([start_time, end_time])
            pred_pitches.append(round(current_pitch))
        
        return np.array(pred_intervals), np.array(pred_pitches)
    
    def intervals_to_frames(self, intervals, pitches, times):
        """Convert note intervals to frame-level representation."""
        freqs = np.zeros_like(times)
        
        for (start, end), pitch in zip(intervals, pitches):
            mask = (times >= start) & (times < end)
            freqs[mask] = librosa.midi_to_hz(pitch)
        
        return freqs
    
    def evaluate(self, est_times, est_freqs, json_path):
        """
        Evaluate predictions against JSON ground truth.
        Returns both note-level and frame-level metrics.
        """
        ref_intervals, ref_pitches = self.load_ground_truth_from_json(json_path)
        
        if len(ref_pitches) == 0:
            print(f"No ground truth notes")
            return self._empty_metrics()
        
        pred_intervals, pred_pitches = self.predictions_to_intervals(est_times, est_freqs)
        
        if len(pred_pitches) == 0:
            print(f"No notes predicted")
            return self._empty_metrics(ref_count=len(ref_pitches))
        
        # ===== NOTE-LEVEL METRICS =====
        ref_pitches_hz = librosa.midi_to_hz(ref_pitches)
        pred_pitches_hz = librosa.midi_to_hz(pred_pitches)
        
        try:
            precision, recall, f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
                ref_intervals=ref_intervals,
                ref_pitches=ref_pitches_hz,
                est_intervals=pred_intervals,
                est_pitches=pred_pitches_hz,
                onset_tolerance=self.onset_tolerance,
                pitch_tolerance=self.pitch_tolerance,
                offset_ratio=None
            )
        except Exception as e:
            print(f"Note-level evaluation error: {e}")
            precision = recall = f1 = 0.0
        
        # ===== FRAME-LEVEL METRICS =====
        ref_freqs_frame = self.intervals_to_frames(ref_intervals, ref_pitches, est_times)
        
        ref_voicing = (ref_freqs_frame > 0).astype(float)
        est_voicing = (est_freqs > 0).astype(float)
        
        try:
            rpa = mir_eval.melody.raw_pitch_accuracy(
                ref_voicing, ref_freqs_frame, 
                est_voicing, est_freqs,
                cent_tolerance=self.pitch_tolerance
            )
            
            rca = mir_eval.melody.raw_chroma_accuracy(
                ref_voicing, ref_freqs_frame,
                est_voicing, est_freqs,
                cent_tolerance=self.pitch_tolerance
            )
            
            oa = mir_eval.melody.overall_accuracy(
                ref_voicing, ref_freqs_frame,
                est_voicing, est_freqs,
                cent_tolerance=self.pitch_tolerance
            )
            
            vr = mir_eval.melody.voicing_recall(ref_voicing, est_voicing)
            
            vfa = mir_eval.melody.voicing_false_alarm(ref_voicing, est_voicing)
            
        except Exception as e:
            print(f"Frame-level evaluation error: {e}")
            rpa = rca = oa = vr = vfa = 0.0
        
       
        return {
            # Note-level metrics
            'precision': float(precision),
            'recall': float(recall),
            'f_measure': float(f1),
            'ref_count': int(len(ref_pitches)),
            'pred_count': int(len(pred_pitches)),
            
            # Frame-level melody metrics
            'raw_pitch_accuracy': float(rpa),
            'raw_chroma_accuracy': float(rca),
            'overall_accuracy': float(oa),
            'voicing_recall': float(vr),
            'voicing_false_alarm': float(vfa)
        }

    def _empty_metrics(self, ref_count=0, pred_count=0):
        """Return empty metrics dict."""
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

class VocalMIDIEvaluator_PYIN_musicxml:
    """Evaluate PYIN-based vocal to MIDI conversion using MusicXML as ground truth."""
    
    def __init__(self, onset_tolerance=0.05, pitch_tolerance=50):
        self.onset_tolerance = onset_tolerance
        self.pitch_tolerance = pitch_tolerance
    
    def load_ground_truth_from_musicxml(self, musicxml_path: str):
        """Load ground truth from MusicXML format."""
        try:
            # Parse MusicXML file
            score = music21.converter.parse(musicxml_path)
            
            ref_intervals = []
            ref_pitches = []
            
            # Extract all notes from all parts
            for part in score.parts:
                for note in part.flatten().notes:
                    if note.isNote:  # Single note (not chord)
                        start_time = float(note.offset)
                        duration = float(note.quarterLength)
                        end_time = start_time + duration
                        midi_pitch = note.pitch.midi
                        
                        ref_intervals.append([start_time, end_time])
                        ref_pitches.append(float(midi_pitch))
                    elif note.isChord:  # Handle chords by taking highest note
                        start_time = float(note.offset)
                        duration = float(note.quarterLength)
                        end_time = start_time + duration
                        # Take the highest pitch in the chord
                        midi_pitch = max([p.midi for p in note.pitches])
                        
                        ref_intervals.append([start_time, end_time])
                        ref_pitches.append(float(midi_pitch))
            
            return np.array(ref_intervals), np.array(ref_pitches)
            
        except Exception as e:
            print(f"Error loading MusicXML: {e}")
            return np.array([]), np.array([])
    
    def predictions_to_intervals(self, times, freqs, min_duration=0.05):
        """Convert frame-wise predictions to note intervals."""
        midi_seq = np.zeros_like(freqs)
        voiced_mask = freqs > 0
        if np.any(voiced_mask):
            midi_seq[voiced_mask] = librosa.hz_to_midi(freqs[voiced_mask])
        
        pred_intervals = []
        pred_pitches = []
        
        if len(midi_seq) == 0:
            return np.array(pred_intervals), np.array(pred_pitches)
        
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
        
        end_time = times[-1]
        duration = end_time - start_time
        if current_pitch > 0 and duration >= min_duration:
            pred_intervals.append([start_time, end_time])
            pred_pitches.append(round(current_pitch))
        
        return np.array(pred_intervals), np.array(pred_pitches)
    
    def intervals_to_frames(self, intervals, pitches, times):
        """Convert note intervals to frame-level representation."""
        freqs = np.zeros_like(times)
        
        for (start, end), pitch in zip(intervals, pitches):
            mask = (times >= start) & (times < end)
            freqs[mask] = librosa.midi_to_hz(pitch)
        
        return freqs
    
    def evaluate(self, est_times, est_freqs, musicxml_path):
        """
        Evaluate predictions against MusicXML ground truth.
        Returns both note-level and frame-level metrics.
        """
        ref_intervals, ref_pitches = self.load_ground_truth_from_musicxml(musicxml_path)
        
        if len(ref_pitches) == 0:
            print(f"No ground truth notes")
            return self._empty_metrics()
        
        pred_intervals, pred_pitches = self.predictions_to_intervals(est_times, est_freqs)
        
        if len(pred_pitches) == 0:
            print(f"No notes predicted")
            return self._empty_metrics(ref_count=len(ref_pitches))
        
        # ===== NOTE-LEVEL METRICS =====
        ref_pitches_hz = librosa.midi_to_hz(ref_pitches)
        pred_pitches_hz = librosa.midi_to_hz(pred_pitches)
        
        try:
            precision, recall, f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
                ref_intervals=ref_intervals,
                ref_pitches=ref_pitches_hz,
                est_intervals=pred_intervals,
                est_pitches=pred_pitches_hz,
                onset_tolerance=self.onset_tolerance,
                pitch_tolerance=self.pitch_tolerance,
                offset_ratio=None
            )
        except Exception as e:
            print(f"Note-level evaluation error: {e}")
            precision = recall = f1 = 0.0
        
        # ===== FRAME-LEVEL METRICS =====
        ref_freqs_frame = self.intervals_to_frames(ref_intervals, ref_pitches, est_times)
        
        ref_voicing = (ref_freqs_frame > 0).astype(float)
        est_voicing = (est_freqs > 0).astype(float)
        
        try:
            rpa = mir_eval.melody.raw_pitch_accuracy(
                ref_voicing, ref_freqs_frame, 
                est_voicing, est_freqs,
                cent_tolerance=self.pitch_tolerance
            )
            
            rca = mir_eval.melody.raw_chroma_accuracy(
                ref_voicing, ref_freqs_frame,
                est_voicing, est_freqs,
                cent_tolerance=self.pitch_tolerance
            )
            
            oa = mir_eval.melody.overall_accuracy(
                ref_voicing, ref_freqs_frame,
                est_voicing, est_freqs,
                cent_tolerance=self.pitch_tolerance
            )
            
            vr = mir_eval.melody.voicing_recall(ref_voicing, est_voicing)
            
            vfa = mir_eval.melody.voicing_false_alarm(ref_voicing, est_voicing)
            
        except Exception as e:
            print(f"Frame-level evaluation error: {e}")
            rpa = rca = oa = vr = vfa = 0.0
        
        # ===== RETURN METRICS =====
        return {
            # Note-level metrics
            'precision': float(precision),
            'recall': float(recall),
            'f_measure': float(f1),
            'ref_count': int(len(ref_pitches)),
            'pred_count': int(len(pred_pitches)),
            
            # Frame-level melody metrics
            'raw_pitch_accuracy': float(rpa),
            'raw_chroma_accuracy': float(rca),
            'overall_accuracy': float(oa),
            'voicing_recall': float(vr),
            'voicing_false_alarm': float(vfa)
        }

    def _empty_metrics(self, ref_count=0, pred_count=0):
        """Return empty metrics dict."""
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
    
# Modified batch processing function that converts once and evaluates twice
def test_batch_processing_combined():
    """
    Convert audio to MIDI once, then evaluate against both JSON and MusicXML ground truth.
    """
    batch_dir = Path("batch_results_gtsinger50_combined")
    batch_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Combined Batch Processing (GTSinger Sample 50) ===")
    print("Converting audio → MIDI once, then evaluating against JSON & MusicXML\n")
    
    converter = VocalToMIDIConverter_PYIN()
    evaluator_json = VocalMIDIEvaluator_PYIN_JSON()
    evaluator_musicxml = VocalMIDIEvaluator_PYIN_musicxml()
    
    # Find all .wav files
    data_dir = Path("./GTSinger_sample_50")
    
    if not data_dir.exists():
        print(f"❌ Directory not found: {data_dir}")
        return None, None
    
    wav_files = list(data_dir.rglob("*.wav"))
    
    if not wav_files:
        print(f"❌ No .wav files found in {data_dir}")
        return None, None
    
    print(f"🎵 Found {len(wav_files)} audio files")
    
    # Build file pairs
    audio_files = []
    for wav_path in wav_files:
        json_path = wav_path.with_suffix('.json')
        musicxml_path = wav_path.with_suffix('.musicxml')
        
        has_json = json_path.exists()
        has_musicxml = musicxml_path.exists()
        
        if has_json or has_musicxml:
            audio_files.append({
                'wav': str(wav_path),
                'json': str(json_path) if has_json else None,
                'musicxml': str(musicxml_path) if has_musicxml else None
            })
    
    print(f"✅ Found {len(audio_files)} audio files with ground truth\n")
    
    results_json = []
    results_musicxml = []
    
    for i, file_info in enumerate(audio_files, 1):
        audio_path = file_info['wav']
        json_path = file_info['json']
        musicxml_path = file_info['musicxml']
        
        print(f"[{i}/{len(audio_files)}] Processing: {Path(audio_path).name}")
        
        # Create unique filename
        wav_path = Path(audio_path)
        relative_path = wav_path.relative_to(data_dir)
        unique_filename = str(relative_path.with_suffix('')).replace('/', '_').replace('\\', '_')
        output_midi = batch_dir / f"{unique_filename}_pyin.mid"
        
        try:
            # ===== CONVERT ONCE =====
            times, freqs, notes = converter.process_file(audio_path, str(output_midi))
            print(f"  ✅ MIDI saved to: {output_midi}")
            
            # ===== EVALUATE AGAINST JSON =====
            if json_path:
                metrics_json = evaluator_json.evaluate(times, freqs, json_path)
                metrics_json['file'] = audio_path
                metrics_json['output_midi'] = str(output_midi)
                metrics_json['ground_truth'] = json_path
                results_json.append(metrics_json)
                print(f"  📊 JSON - F-measure: {metrics_json['f_measure']:.4f}, RPA: {metrics_json['raw_pitch_accuracy']:.4f}")
            
            # ===== EVALUATE AGAINST MUSICXML =====
            if musicxml_path:
                metrics_musicxml = evaluator_musicxml.evaluate(times, freqs, musicxml_path)
                metrics_musicxml['file'] = audio_path
                metrics_musicxml['output_midi'] = str(output_midi)
                metrics_musicxml['ground_truth'] = musicxml_path
                results_musicxml.append(metrics_musicxml)
                print(f"  📊 MusicXML - F-measure: {metrics_musicxml['f_measure']:.4f}, RPA: {metrics_musicxml['raw_pitch_accuracy']:.4f}")
            
        except Exception as e:
            print(f"  ❌ Error processing {unique_filename}: {e}")
            continue
    
    # ===== SAVE JSON RESULTS =====
    if results_json:
        df_json = pd.DataFrame(results_json)
        csv_path_json = batch_dir / "evaluation_results_json.csv"
        df_json.to_csv(csv_path_json, index=False)
        print(f"\n✅ JSON results saved to: {csv_path_json}")
        
        summary_path_json = batch_dir / "summary_statistics_json.txt"
        metric_cols = ['precision', 'recall', 'f_measure', 'raw_pitch_accuracy', 
                       'raw_chroma_accuracy', 'overall_accuracy', 'voicing_recall', 
                       'voicing_false_alarm']
        
        with open(summary_path_json, 'w') as f:
            f.write("📊 Summary Statistics (JSON Ground Truth):\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total files processed: {len(results_json)}\n\n")
            f.write(df_json[metric_cols].describe().to_string())
        
        print(f"✅ JSON summary saved to: {summary_path_json}")
    else:
        df_json = None
        print("⚠️  No JSON evaluations completed")
    
    # ===== SAVE MUSICXML RESULTS =====
    if results_musicxml:
        df_musicxml = pd.DataFrame(results_musicxml)
        csv_path_musicxml = batch_dir / "evaluation_results_musicxml.csv"
        df_musicxml.to_csv(csv_path_musicxml, index=False)
        print(f"\n✅ MusicXML results saved to: {csv_path_musicxml}")
        
        summary_path_musicxml = batch_dir / "summary_statistics_musicxml.txt"
        
        with open(summary_path_musicxml, 'w') as f:
            f.write("📊 Summary Statistics (MusicXML Ground Truth):\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total files processed: {len(results_musicxml)}\n\n")
            f.write(df_musicxml[metric_cols].describe().to_string())
        
        print(f"✅ MusicXML summary saved to: {summary_path_musicxml}")
    else:
        df_musicxml = None
        print("⚠️  No MusicXML evaluations completed")
    
    print(f"\n🎉 Processing complete!")
    print(f"   JSON evaluations: {len(results_json)}")
    print(f"   MusicXML evaluations: {len(results_musicxml)}")
    
    return df_json, df_musicxml

# Run combined batch processing
df_json, df_musicxml = test_batch_processing_combined()


# Display results
if df_json is not None:
    print("\n" + "="*60)
    print("📊 JSON Ground Truth - Mean Metrics:")
    print("="*60)
    metric_cols = ['precision', 'recall', 'f_measure', 'raw_pitch_accuracy', 
                   'raw_chroma_accuracy', 'overall_accuracy', 'voicing_recall', 
                   'voicing_false_alarm']
    
    for col in metric_cols:
        print(f"  {col:30s}: {df_json[col].mean():.6f}")