import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import librosa
from crepe import predict
from mir_eval.melody import evaluate as melody_eval
from music21 import converter, note, stream, key as m21key, clef, meter, tempo
from scipy.signal import medfilt
from dtw import dtw
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')  # disable GPU use for CREPE

# === CONFIG ===
root_dir = "./GTSinger_sample_50"
output_dir = "./predicted_crepe_xml_folder"
frame_rate = 100
median_filter_size = 5
MIN_NOTE_SEC = 0.12
GAP_JOIN_SEC = 0.4
ONSET_TOL_SEC = 0.30
OFFSET_TOL_SEC = 0.35
PITCH_TOL_SEMITONES = 0.75
CONFIDENCE_THRESHOLD = 0.5

os.makedirs(output_dir, exist_ok=True)

# === Helper ===
def ensure_unique_path(path):
    base, ext = os.path.splitext(path)
    counter = 1
    while os.path.exists(path):
        path = f"{base}_v{counter}{ext}"
        counter += 1
    return path

def hz_to_midi(hz):
    return 69 + 12 * np.log2(hz / 440.0) if hz > 0 else None

# === Harmony Generation ===
HARMONY_MIN_MIDI = 55   # G3
HARMONY_MAX_MIDI = 81   # A5
DEFAULT_BPM = 120

def detect_key_from_score(sc):
    try:
        k = sc.analyze('key')
        if isinstance(k, m21key.Key):
            return k
    except Exception:
        pass
    ks = sc.flat.getElementsByClass('KeySignature')
    if ks:
        try:
            return ks[0].asKey()
        except Exception:
            pass
    return m21key.Key('C')

def build_scale_pitch_list(k, low='C1', high='C7'):
    sca = k.getScale()
    tonic = k.tonic.name
    low, high = f"{tonic}1", f"{tonic}7"
    return sca.getPitches(low, high)

def diatonic_step(p, k, steps):
    plist = build_scale_pitch_list(k)
    midis = [pp.midi for pp in plist]
    m = p.midi
    idx = min(range(len(midis)), key=lambda i: abs(midis[i] - m))
    idx_target = int(np.clip(idx + steps, 0, len(plist) - 1))
    return plist[idx_target]

def generate_harmony_part_from_melody(melody_part, k, prefer_third_below=True):
    harm = stream.Part()
    harm.insert(0, tempo.MetronomeMark(number=DEFAULT_BPM))
    harm.insert(0, meter.TimeSignature("4/4"))
    harm.insert(0, clef.TrebleClef())
    for n in melody_part.recurse().notes:
        if not isinstance(n, note.Note):
            continue
        src_pitch = n.pitch
        candidate = diatonic_step(src_pitch, k, steps=-2 if prefer_third_below else +2)
        midi_val = candidate.midi
        if midi_val < HARMONY_MIN_MIDI or midi_val > HARMONY_MAX_MIDI:
            candidate = diatonic_step(src_pitch, k, steps=+5)
            midi_val = candidate.midi
        midi_val = min(max(midi_val, HARMONY_MIN_MIDI), HARMONY_MAX_MIDI)
        hnote = note.Note(int(round(midi_val)))
        hnote.duration = n.duration
        harm.append(hnote)
    return harm

def write_harmonized_score_from_xml(melody_xml_path):
    sc = converter.parse(melody_xml_path)
    melody_part = sc.parts[0] if sc.parts else sc.flatten().parts[0]
    k = detect_key_from_score(sc)
    harm_part = generate_harmony_part_from_melody(melody_part, k, prefer_third_below=True)
    new_sc = stream.Score()
    for el in melody_part.getElementsByClass((tempo.MetronomeMark, meter.TimeSignature, clef.Clef)):
        new_sc.insert(0, el)
    melody_part.id, harm_part.id = "Melody", "Harmony"
    new_sc.insert(0, melody_part)
    new_sc.insert(0, harm_part)
    out = os.path.splitext(melody_xml_path)[0] + "_harmonized.musicxml"
    new_sc.write('musicxml', fp=out)
    print(f"🎶 Saved harmonized score: {out}")
    return out

# === Convert MusicXML to pitch sequence ===
def musicxml_to_pitch_sequence(xml_file, sr=frame_rate):
    try:
        score = converter.parse(xml_file)
    except Exception as e:
        print(f" Error parsing {xml_file}: {e}")
        return None, None

    notes = score.flat.notes
    if not notes:
        print(f" No notes found in {xml_file}")
        return None, None

    end_time = max(n.offset + n.quarterLength for n in notes)
    times = np.arange(0, end_time, 1 / sr)
    freqs = np.zeros_like(times)

    for n in notes:
        if isinstance(n, note.Note):
            hz = n.pitch.frequency
            start_idx = int(n.offset * sr)
            end_idx = int((n.offset + n.quarterLength) * sr)
            end_idx = min(end_idx, len(freqs))
            freqs[start_idx:end_idx] = hz

    return times, freqs

# === Convert MusicXML to note events ===
def musicxml_to_note_events(xml_file):
    try:
        sc = converter.parse(xml_file)
    except Exception as e:
        print(f" Error parsing {xml_file}: {e}")
        return []
    evs = []
    for n in sc.flat.notes:
        if isinstance(n, note.Note):
            on, off = float(n.offset), float(n.offset + n.quarterLength)
            midi_val = n.pitch.midi
            evs.append((on, off, midi_val))
    return evs

# === Pitch processing ===
def smooth_pitch(frequencies, window=median_filter_size):
    return medfilt(frequencies, kernel_size=window)

def correct_octave_errors(freqs, max_jump_semitones=6):
    midi_vals = np.array([hz_to_midi(f) if f > 0 else 0 for f in freqs])
    corrected = np.copy(midi_vals)
    for i in range(1, len(midi_vals)):
        if midi_vals[i] == 0 or corrected[i-1] == 0:
            continue
        diff = midi_vals[i] - corrected[i-1]
        if abs(abs(diff) - 12) < 2:
            corrected[i] = corrected[i-1]
        elif abs(diff) > max_jump_semitones:
            candidates = [midi_vals[i], midi_vals[i] - 12, midi_vals[i] + 12]
            distances = [abs(c - corrected[i-1]) for c in candidates]
            corrected[i] = candidates[np.argmin(distances)]
        else:
            corrected[i] = midi_vals[i]
    return np.array([librosa.midi_to_hz(m) if m > 0 else 0 for m in corrected])

def extract_pitch(y, sr, conf_threshold=CONFIDENCE_THRESHOLD):
    time, frequency, confidence, activation = predict(y, sr, viterbi=True)
    frequency[confidence < conf_threshold] = 0.0
    frequency = correct_octave_errors(frequency)
    frequency = smooth_pitch(frequency)
    return time, frequency

# === Frame-to-note segmentation ===
def frames_to_notes_advanced(times, freqs, min_note_sec=MIN_NOTE_SEC):
    if len(freqs) == 0:
        return []
    midi_vals = np.array([hz_to_midi(f) if f > 0 else 0 for f in freqs])
    is_voiced = midi_vals > 0
    pitch_stable = np.ones(len(midi_vals), dtype=bool)
    for i in range(1, len(midi_vals) - 1):
        if is_voiced[i]:
            prev_stable = not is_voiced[i-1] or abs(midi_vals[i] - midi_vals[i-1]) < 0.4
            next_stable = not is_voiced[i+1] or abs(midi_vals[i] - midi_vals[i+1]) < 0.4
            pitch_stable[i] = prev_stable and next_stable
    notes = []
    in_note = False
    note_start_idx = 0
    note_pitches = []
    for i in range(len(midi_vals)):
        if is_voiced[i] and pitch_stable[i]:
            if not in_note:
                in_note = True
                note_start_idx = i
                note_pitches = [midi_vals[i]]
            else:
                if abs(midi_vals[i] - np.median(note_pitches)) < 0.8:
                    note_pitches.append(midi_vals[i])
                else:
                    if len(note_pitches) > 0:
                        onset = times[note_start_idx]
                        offset = times[i-1]
                        if offset - onset >= min_note_sec:
                            notes.append([onset, offset, np.median(note_pitches)])
                    note_start_idx = i
                    note_pitches = [midi_vals[i]]
        else:
            if in_note and len(note_pitches) > 0:
                onset = times[note_start_idx]
                offset = times[i-1]
                if offset - onset >= min_note_sec:
                    notes.append([onset, offset, np.median(note_pitches)])
            in_note = False
            note_pitches = []
    if in_note and len(note_pitches) > 0:
        onset = times[note_start_idx]
        offset = times[-1]
        if offset - onset >= min_note_sec:
            notes.append([onset, offset, np.median(note_pitches)])
    if not notes:
        return []
    merged = [notes[0]]
    for onset, offset, pitch in notes[1:]:
        last_onset, last_offset, last_pitch = merged[-1]
        if abs(pitch - last_pitch) <= 0.8 and (onset - last_offset) <= GAP_JOIN_SEC:
            merged[-1][1] = offset
            merged[-1][2] = (pitch + last_pitch) / 2.0
        else:
            merged.append([onset, offset, pitch])
    return merged

# === Convert to MusicXML ===
def notes_to_musicxml(notes_list, output_path, min_quarter_length=0.125):
    if not notes_list:
        print(f" No notes to save for {output_path}")
        return None
    s = stream.Stream()
    for onset, offset, midi_val in notes_list:
        n = note.Note(int(round(midi_val)))
        duration_sec = max(0.0, offset - onset)
        quarter_len = duration_sec * 2.0
        if quarter_len < min_quarter_length:
            continue
        n.quarterLength = max(min_quarter_length, round(quarter_len * 8) / 8)
        s.append(n)
    if len(s.notes) == 0:
        print(f" No valid notes created for {output_path}")
        return None
    output_path = ensure_unique_path(output_path)
    s.write("musicxml", fp=output_path)
    print(f" Saved predicted MusicXML: {output_path} ({len(s.notes)} notes)")
    return output_path

# === Evaluation ===
def evaluate_notes(pred_notes, ref_notes):
    if not pred_notes or not ref_notes:
        return {"precision": 0, "recall": 0, "f1": 0}
    pred_notes = [(on, off, round(m)) for on, off, m in pred_notes]
    ref_notes = [(on, off, round(m)) for on, off, m in ref_notes]
    tp, ref_used = 0, np.zeros(len(ref_notes), dtype=bool)
    for (pon, poff, pm) in pred_notes:
        for j, (ron, roff, rm) in enumerate(ref_notes):
            if ref_used[j]:
                continue
            if (abs(pon - ron) <= ONSET_TOL_SEC and
                abs(poff - roff) <= OFFSET_TOL_SEC and
                abs(pm - rm) <= PITCH_TOL_SEMITONES):
                tp += 1
                ref_used[j] = True
                break
    fp = len(pred_notes) - tp
    fn = len(ref_notes) - tp
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0
    return {"precision": prec, "recall": rec, "f1": f1}

# === Main Loop ===
results_frame_raw, results_frame_dtw, results_note = [], [], []
processed, skipped = 0, 0

for root, _, files in os.walk(root_dir):
    wav_files = [f for f in files if f.endswith(".wav")]
    for wav_file in wav_files:
        wav_path = os.path.join(root, wav_file)
        xml_path = os.path.join(root, wav_file.replace(".wav", ".musicxml"))
        if not os.path.exists(xml_path):
            print(f"Skipping {wav_path}, no matching MusicXML found")
            skipped += 1
            continue
        print(f"\n Processing {wav_file}...")
        rel_path = os.path.relpath(root, root_dir)
        dest_dir = os.path.join(output_dir, rel_path)
        os.makedirs(dest_dir, exist_ok=True)
        pred_xml_path = os.path.join(dest_dir, wav_file.replace(".wav", "_predicted.musicxml"))
        try:
            y, sr = librosa.load(wav_path, sr=None)
            y, _ = librosa.effects.trim(y, top_db=20)
            y_harmonic, _ = librosa.effects.hpss(y)
            est_time, est_freq = extract_pitch(y_harmonic, sr)
            pred_notes = frames_to_notes_advanced(est_time, est_freq)
            pred_xml_path = notes_to_musicxml(pred_notes, pred_xml_path)
            if not pred_xml_path:
                skipped += 1
                continue
            # Generate harmony
            write_harmonized_score_from_xml(pred_xml_path)
            # Frame-level eval
            ref_time, ref_freq = musicxml_to_pitch_sequence(xml_path)
            if ref_time is None:
                skipped += 1
                continue
            max_t = min(est_time[-1], ref_time[-1])
            t_common = np.arange(0, max_t, 1 / frame_rate)
            est_freq_interp = np.interp(t_common, est_time, est_freq)
            ref_freq_interp = np.interp(t_common, ref_time, ref_freq)
            raw_scores = melody_eval(t_common, ref_freq_interp, t_common, est_freq_interp)
            results_frame_raw.append(raw_scores)
            ref_v, est_v = ref_freq_interp[ref_freq_interp > 0], est_freq_interp[est_freq_interp > 0]
            if len(ref_v) > 2 and len(est_v) > 2:
                ali = dtw(ref_v.reshape(-1, 1), est_v.reshape(-1, 1), keep_internals=True)
                ar, ae = ref_v[ali.index1s], est_v[ali.index2s]
                t_ar, t_ae = np.arange(len(ar))/frame_rate, np.arange(len(ae))/frame_rate
                aligned_scores = melody_eval(t_ar, ar, t_ae, ae)
                results_frame_dtw.append(aligned_scores)
            ref_notes = musicxml_to_note_events(xml_path)
            note_scores = evaluate_notes(pred_notes, ref_notes)
            results_note.append(note_scores)
            processed += 1
            print(f" Done: {wav_file}")
        except Exception as e:
            print(f" Error processing {wav_file}: {e}")
            import traceback
            traceback.print_exc()
            skipped += 1
            continue

def summarize_frame(results, title):
    if not results:
        print(f"\n No valid {title} results.")
        return
    avg = {f"avg_{k.replace(' ', '_').lower()}": np.mean([r[k] for r in results]) for k in results[0]}
    print(f"\n Frame-Level Metrics ({title}):")
    for k, v in avg.items():
        print(f"{k}: {v:.4f}")

def summarize_notes(nmetrics):
    if not nmetrics:
        print("\n No valid Note-level results.")
        return
    P = np.mean([m['precision'] for m in nmetrics])
    R = np.mean([m['recall'] for m in nmetrics])
    F = np.mean([m['f1'] for m in nmetrics])
    print("\n Note-Level Metrics:")
    print(f"avg_precision : {P:.4f}")
    print(f"avg_recall    : {R:.4f}")
    print(f"avg_f_measure : {F:.4f}")

summarize_frame(results_frame_raw, "CREPE Raw")
summarize_frame(results_frame_dtw, "CREPE DTW-Aligned")
summarize_notes(results_note)

print(f"\n Summary: Processed {processed} files, Skipped {skipped} files.")
print(f" Predicted XML root: {output_dir}")