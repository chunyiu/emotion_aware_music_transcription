"""
Shared note segmentation methods.
All return lists of dicts with keys: start, end, pitch_midi, pitch_hz, confidence.
Compatible with TranscribedNote schema.
"""

import numpy as np
import librosa
from scipy.signal import medfilt


def segment_notes_simple(f0, voiced_flag, times, sr=16000, hop_length=512,
                          min_note_duration=0.1):
    """
    Simple note segmentation from pYIN output.
    Groups voiced frames into notes by averaging pitch.

    Returns:
        List of note dicts with start, end, pitch_midi, pitch_hz, confidence.
    """
    notes = []
    current_note = None
    min_frames = int(min_note_duration * sr / hop_length)

    for i, (pitch, is_voiced) in enumerate(zip(f0, voiced_flag)):
        time = times[i]
        if is_voiced and not np.isnan(pitch):
            midi_pitch = librosa.hz_to_midi(pitch)
            if current_note is None:
                current_note = {
                    'start': time,
                    'pitch_sum': midi_pitch,
                    'hz_sum': float(pitch),
                    'pitch_count': 1,
                    'start_frame': i
                }
            else:
                current_note['pitch_sum'] += midi_pitch
                current_note['hz_sum'] += float(pitch)
                current_note['pitch_count'] += 1
        else:
            if current_note is not None:
                duration_frames = i - current_note['start_frame']
                if duration_frames >= min_frames:
                    avg_midi = current_note['pitch_sum'] / current_note['pitch_count']
                    avg_hz = current_note['hz_sum'] / current_note['pitch_count']
                    notes.append({
                        'start': current_note['start'],
                        'end': time,
                        'pitch_midi': int(round(avg_midi)),
                        'pitch_hz': float(avg_hz),
                        'confidence': 1.0,
                    })
                current_note = None

    # Handle trailing note
    if current_note is not None:
        avg_midi = current_note['pitch_sum'] / current_note['pitch_count']
        avg_hz = current_note['hz_sum'] / current_note['pitch_count']
        notes.append({
            'start': current_note['start'],
            'end': times[-1],
            'pitch_midi': int(round(avg_midi)),
            'pitch_hz': float(avg_hz),
            'confidence': 1.0,
        })

    return notes


def segment_notes_hmm(times, smoothed_midi, config=None):
    """
    Note segmentation from HMM-smoothed MIDI sequence.
    Uses adaptive pitch threshold with vibrato tolerance.

    Args:
        times: array of frame times
        smoothed_midi: array of MIDI pitch values (0 = unvoiced)
        config: object with min_note_duration, pitch_threshold,
                vibrato_tolerance, merge_threshold, min_gap

    Returns:
        List of note dicts.
    """
    if config is None:
        from common.pitch_detectors import _default_hmm_config
        config = _default_hmm_config()

    notes = _improved_midi_to_note_events(times, smoothed_midi, config)
    notes = _post_process_notes(notes, config)
    return notes


def segment_notes_crepe(times, f0, conf, conf_threshold=0.45,
                         median_filter_size=5, min_note_sec=0.15,
                         gap_join_sec=0.20, pitch_change_threshold=0.5):
    """
    Note segmentation from CREPE/TorchCrepe output.
    Applies confidence threshold, median filter, and pitch-change detection.

    Returns:
        List of note dicts.
    """
    # Apply confidence threshold
    f0_filt = f0.copy()
    f0_filt[conf < conf_threshold] = 0.0

    # Median filtering
    f0_smooth = medfilt(f0_filt, kernel_size=median_filter_size)

    # Convert to MIDI
    midi = np.zeros_like(f0_smooth)
    mask_voiced = (f0_smooth > 0)
    midi[mask_voiced] = librosa.hz_to_midi(f0_smooth[mask_voiced])

    # Segment into notes
    notes = []
    in_note = False
    note_start_idx = None
    note_pitches = []
    note_hz_values = []

    for i in range(len(midi)):
        if midi[i] > 0:
            if not in_note:
                in_note = True
                note_start_idx = i
                note_pitches = [midi[i]]
                note_hz_values = [f0_smooth[i]]
            else:
                if note_pitches:
                    pitch_diff = abs(midi[i] - np.median(note_pitches))
                    if pitch_diff > pitch_change_threshold:
                        # End current note, start new
                        note_end_idx = i - 1
                        duration = times[note_end_idx] - times[note_start_idx]
                        if duration >= min_note_sec:
                            median_midi = np.median(note_pitches)
                            median_hz = np.median(note_hz_values)
                            avg_conf = float(np.mean(conf[note_start_idx:i]))
                            notes.append({
                                'start': float(times[note_start_idx]),
                                'end': float(times[note_end_idx]),
                                'pitch_midi': int(round(median_midi)),
                                'pitch_hz': float(median_hz),
                                'confidence': avg_conf,
                            })
                        note_start_idx = i
                        note_pitches = [midi[i]]
                        note_hz_values = [f0_smooth[i]]
                    else:
                        note_pitches.append(midi[i])
                        note_hz_values.append(f0_smooth[i])
        else:
            if in_note:
                note_end_idx = i - 1
                duration = times[note_end_idx] - times[note_start_idx]
                if duration >= min_note_sec:
                    median_midi = np.median(note_pitches)
                    median_hz = np.median(note_hz_values)
                    avg_conf = float(np.mean(conf[note_start_idx:i]))
                    notes.append({
                        'start': float(times[note_start_idx]),
                        'end': float(times[note_end_idx]),
                        'pitch_midi': int(round(median_midi)),
                        'pitch_hz': float(median_hz),
                        'confidence': avg_conf,
                    })
                in_note = False
                note_pitches = []
                note_hz_values = []

    # Handle final note
    if in_note and note_pitches:
        duration = times[-1] - times[note_start_idx]
        if duration >= min_note_sec:
            median_midi = np.median(note_pitches)
            median_hz = np.median(note_hz_values)
            avg_conf = float(np.mean(conf[note_start_idx:len(midi)]))
            notes.append({
                'start': float(times[note_start_idx]),
                'end': float(times[-1]),
                'pitch_midi': int(round(median_midi)),
                'pitch_hz': float(median_hz),
                'confidence': avg_conf,
            })

    # Merge close notes with similar pitch
    if len(notes) > 1:
        merged = [notes[0]]
        for n in notes[1:]:
            prev = merged[-1]
            gap = n['start'] - prev['end']
            pitch_diff = abs(n['pitch_midi'] - prev['pitch_midi'])
            if gap <= gap_join_sec and pitch_diff <= 0.5:
                prev['end'] = n['end']
                prev['pitch_midi'] = int(round(
                    (prev['pitch_midi'] + n['pitch_midi']) / 2
                ))
                prev['pitch_hz'] = (prev['pitch_hz'] + n['pitch_hz']) / 2
                prev['confidence'] = (prev['confidence'] + n['confidence']) / 2
            else:
                merged.append(n)
        notes = merged

    return notes


# ---------------------------------------------------------------------------
# Internal helpers for HMM-based segmentation
# ---------------------------------------------------------------------------

def _improved_midi_to_note_events(times, midi_seq, config):
    """Two-pass note detection with local pitch stability analysis."""
    notes = []
    if len(midi_seq) == 0:
        return notes

    def round_time(t, step=0.01):
        return float(np.round(t / step) * step)

    # Pass 1: local pitch stability
    window_size = 5
    pitch_variance = np.zeros_like(midi_seq, dtype=float)
    for i in range(len(midi_seq)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(midi_seq), i + window_size // 2 + 1)
        window = midi_seq[start_idx:end_idx]
        voiced_window = window[window > 0]
        if len(voiced_window) > 1:
            pitch_variance[i] = np.std(voiced_window)

    # Pass 2: segment with stability awareness
    current_pitch = midi_seq[0]
    start_time = times[0]
    pitch_accumulator = [current_pitch] if current_pitch > 0 else []

    for i in range(1, len(midi_seq)):
        current_val = midi_seq[i]
        prev_val = midi_seq[i - 1]
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
            if pitch_accumulator:
                stable_pitch = np.median(pitch_accumulator)
                if stable_pitch > 0 and duration >= config.min_note_duration:
                    hz_val = librosa.midi_to_hz(stable_pitch)
                    notes.append({
                        'start': round_time(start_time),
                        'end': round_time(end_time),
                        'pitch_midi': int(round(stable_pitch)),
                        'pitch_hz': float(hz_val),
                        'confidence': 1.0,
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
    if pitch_accumulator:
        stable_pitch = np.median(pitch_accumulator)
        if stable_pitch > 0 and duration >= config.min_note_duration:
            hz_val = librosa.midi_to_hz(stable_pitch)
            notes.append({
                'start': round_time(start_time),
                'end': round_time(end_time),
                'pitch_midi': int(round(stable_pitch)),
                'pitch_hz': float(hz_val),
                'confidence': 1.0,
            })

    return notes


def _post_process_notes(notes, config):
    """Merge similar adjacent notes."""
    if len(notes) <= 1:
        return notes

    merged = []
    current = notes[0].copy()
    for next_note in notes[1:]:
        pitch_diff = abs(next_note['pitch_midi'] - current['pitch_midi'])
        time_gap = next_note['start'] - current['end']
        if pitch_diff <= config.merge_threshold and time_gap <= config.min_gap:
            current['end'] = next_note['end']
            current['pitch_midi'] = int(round(
                (current['pitch_midi'] + next_note['pitch_midi']) / 2
            ))
            current['pitch_hz'] = (current['pitch_hz'] + next_note['pitch_hz']) / 2
        else:
            merged.append(current)
            current = next_note.copy()
    merged.append(current)
    return merged
