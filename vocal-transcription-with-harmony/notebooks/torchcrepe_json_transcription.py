import os
import json
import numpy as np
import librosa
import torch
import torchcrepe
from scipy.signal import medfilt
from mir_eval.melody import evaluate as melody_eval
from dtw import dtw
from music21 import stream, note, tempo, meter, clef, converter, key as m21key

# =========================== CONFIG ===========================
root_dir   = "./GTSinger_sample_50"
output_dir = "./predicted_torchcrepe_json_folder"
os.makedirs(output_dir, exist_ok=True)

FRAME_RATE          = 100
MEDIAN_FILTER_SIZE  = 9
FMIN_HZ, FMAX_HZ    = 50.0, 1100.0
MIN_NOTE_SEC        = 0.15
GAP_JOIN_SEC        = 0.3
MIN_Q_LEN_BEATS     = 0.25
DEFAULT_BPM         = 120
PITCH_TOL_SEMITONES = 1.0
ONSET_TOL_SEC       = 0.25
OFFSET_TOL_SEC      = 0.3
CHUNK_SECONDS       = 10
CONFIDENCE_THRESHOLD = 0.5  # Periodicity confidence threshold (0.0-1.0)

# Harmony generation limits
HARMONY_MIN_MIDI = 55   # G3
HARMONY_MAX_MIDI = 81   # A5
# =============================================================

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# -------------------------- Utils ----------------------------
def ensure_unique_path(path):
    base, ext = os.path.splitext(path)
    k, out = 1, path
    while os.path.exists(out):
        out = f"{base}_v{k}{ext}"
        k += 1
    return out

def trim_and_normalize(y):
    y, _ = librosa.effects.trim(y, top_db=25)
    return y / (np.max(np.abs(y)) + 1e-9)

def smooth_pitch(freqs, window=MEDIAN_FILTER_SIZE):
    return medfilt(freqs, kernel_size=window) if window > 1 else freqs

def hz_to_midi(hz):
    return 69 + 12 * np.log2(hz / 440.0) if hz > 0 else None


# ---------------- Enhanced Audio Preprocessing ----------------
def preprocess_audio(y, sr):
    """Enhanced preprocessing for better pitch tracking."""
    from scipy.signal import butter, filtfilt
    y, _ = librosa.effects.trim(y, top_db=20)
    b, a = butter(5, 80, btype='high', fs=sr)
    y = filtfilt(b, a, y)
    y = y / (np.max(np.abs(y)) + 1e-9) * 0.95
    return y


# ---------------- JSON → Notes & Pitch Sequences -------------
def json_to_notes_direct(json_file):
    """Extract notes directly from JSON without frame conversion."""
    try:
        with open(json_file, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f" Error reading {json_file}: {e}")
        return []

    if not isinstance(data, list) or len(data) == 0:
        return []

    note_events = []
    for entry in data:
        if "note_start" in entry and "note_end" in entry and "note" in entry:
            for s, e, p in zip(entry["note_start"], entry["note_end"], entry["note"]):
                if p > 0 and e > s:
                    note_events.append([float(s), float(e), float(p)])
    if not note_events:
        return []
    note_events.sort(key=lambda x: x[0])

    merged = [note_events[0]]
    for s, e, p in note_events[1:]:
        last_s, last_e, last_p = merged[-1]
        if abs(p - last_p) <= 1.0 and s - last_e <= 0.1:
            merged[-1][1] = e
            merged[-1][2] = (p + last_p) / 2.0
        else:
            merged.append([s, e, p])
    return merged

def json_to_pitch_sequence(json_file, sr=FRAME_RATE):
    """Convert JSON to frame-level pitch sequence for frame-level metrics."""
    notes = json_to_notes_direct(json_file)
    if not notes:
        return None, None
    end_time = max(e for _, e, _ in notes)
    times = np.arange(0, end_time, 1 / sr)
    freqs = np.zeros_like(times)
    for s, e, p in notes:
        start_idx = int(s * sr)
        end_idx = min(int(e * sr), len(freqs))
        freqs[start_idx:end_idx] = librosa.midi_to_hz(p)
    return times, freqs


# ---------------- Octave Error Correction ---------------------
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


# ---------------- Pitch Extraction ----------------------------
def extract_pitch_torchcrepe(y, sr, model='full'):
    hop_length = int(sr / FRAME_RATE)
    audio = torch.tensor(y, dtype=torch.float32).unsqueeze(0).to(device)
    max_chunk = int(sr * CHUNK_SECONDS)
    pitch_pieces, periodicity_pieces = [], []
    with torch.no_grad():
        for i in range(0, audio.shape[1], max_chunk):
            chunk = audio[:, i:i + max_chunk]
            torch.cuda.empty_cache()
            pitch, periodicity = torchcrepe.predict(
                chunk, sample_rate=sr, hop_length=hop_length,
                fmin=FMIN_HZ, fmax=FMAX_HZ, model=model,
                device=device, pad=True, batch_size=64,
                return_periodicity=True, decoder=torchcrepe.decode.viterbi
            )
            pitch_pieces.append(pitch.cpu())
            periodicity_pieces.append(periodicity.cpu())
    pitch_hz = torch.cat(pitch_pieces, dim=1).squeeze(0).numpy()
    periodicity = torch.cat(periodicity_pieces, dim=1).squeeze(0).numpy()
    pitch_hz[periodicity < CONFIDENCE_THRESHOLD] = 0
    pitch_hz = correct_octave_errors(pitch_hz)
    pitch_hz = smooth_pitch(pitch_hz)
    times = np.arange(len(pitch_hz)) * (hop_length / sr)
    return times, pitch_hz


# ---------------- Note Segmentation ---------------------
def frames_to_notes_advanced(times, freqs, min_note_sec=0.15):
    if len(freqs) == 0:
        return []
    midi_vals = np.array([hz_to_midi(f) if f > 0 else 0 for f in freqs])
    is_voiced = midi_vals > 0
    pitch_stable = np.ones(len(midi_vals), dtype=bool)
    for i in range(1, len(midi_vals) - 1):
        if is_voiced[i]:
            prev_stable = not is_voiced[i-1] or abs(midi_vals[i] - midi_vals[i-1]) < 0.5
            next_stable = not is_voiced[i+1] or abs(midi_vals[i] - midi_vals[i+1]) < 0.5
            pitch_stable[i] = prev_stable and next_stable
    notes = []
    in_note, note_start_idx, note_pitches = False, 0, []
    for i in range(len(midi_vals)):
        if is_voiced[i] and pitch_stable[i]:
            if not in_note:
                in_note = True
                note_start_idx = i
                note_pitches = [midi_vals[i]]
            else:
                if abs(midi_vals[i] - np.median(note_pitches)) < 1.0:
                    note_pitches.append(midi_vals[i])
                else:
                    onset, offset = times[note_start_idx], times[i-1]
                    if offset - onset >= min_note_sec:
                        notes.append([onset, offset, np.median(note_pitches)])
                    note_start_idx = i
                    note_pitches = [midi_vals[i]]
        else:
            if in_note and len(note_pitches) > 0:
                onset, offset = times[note_start_idx], times[i-1]
                if offset - onset >= min_note_sec:
                    notes.append([onset, offset, np.median(note_pitches)])
            in_note, note_pitches = False, []
    if in_note and len(note_pitches) > 0:
        onset, offset = times[note_start_idx], times[-1]
        if offset - onset >= min_note_sec:
            notes.append([onset, offset, np.median(note_pitches)])
    merged = [notes[0]] if notes else []
    for onset, offset, pitch in notes[1:]:
        last_onset, last_offset, last_pitch = merged[-1]
        if abs(pitch - last_pitch) <= 1.0 and (onset - last_offset) <= GAP_JOIN_SEC:
            merged[-1][1] = offset
            merged[-1][2] = (pitch + last_pitch) / 2.0
        else:
            merged.append([onset, offset, pitch])
    return merged


# ---------------- MusicXML Writer -----------------------------
def write_notes_to_musicxml(notes_list, output_path, bpm=DEFAULT_BPM):
    sc, p = stream.Score(), stream.Part()
    p.insert(0, tempo.MetronomeMark(number=bpm))
    p.insert(0, meter.TimeSignature("4/4"))
    p.insert(0, clef.TrebleClef())
    sec_per_beat = 60.0 / bpm
    for onset, offset, midi_val in notes_list:
        dur_sec = max(0.0, offset - onset)
        ql = max(MIN_Q_LEN_BEATS, round((dur_sec / sec_per_beat) / (1/32)) * (1/32))
        n = note.Note(int(round(midi_val)))
        n.duration.quarterLength = ql
        p.append(n)
    sc.insert(0, p)
    out = ensure_unique_path(output_path)
    sc.write("musicxml", fp=out)
    print(f"🎼 Saved predicted MusicXML: {out}")
    return out


# ---------------- Harmony Generation ---------------------
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


# ---------------- Evaluation ---------------------
def evaluate_notes(pred_notes, ref_notes):
    if not pred_notes or not ref_notes:
        return {"precision": 0, "recall": 0, "f1": 0}
    pred_notes = [(on, off, round(m)) for on, off, m in pred_notes]
    ref_notes  = [(on, off, round(m)) for on, off, m in ref_notes]
    tp, ref_used = 0, np.zeros(len(ref_notes), dtype=bool)
    for (pon, poff, pm) in pred_notes:
        for j, (ron, roff, rm) in enumerate(ref_notes):
            if ref_used[j]: continue
            if (abs(pon - ron) <= ONSET_TOL_SEC and
                abs(poff - roff) <= OFFSET_TOL_SEC and
                abs(pm - rm) <= PITCH_TOL_SEMITONES):
                tp += 1
                ref_used[j] = True
                break
    fp, fn = len(pred_notes) - tp, len(ref_notes) - tp
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec  = tp / (tp + fn) if (tp + fn) else 0
    f1   = (2*prec*rec)/(prec+rec) if (prec+rec) else 0
    return {"precision": prec, "recall": rec, "f1": f1}


# ---------------- Main Loop ---------------------
results_raw, results_aligned, note_metrics_all = [], [], []
processed, skipped = 0, 0

for root, _, files in os.walk(root_dir):
    wav_files = [f for f in files if f.endswith(".wav")]
    for wav_file in wav_files:
        wav_path = os.path.join(root, wav_file)
        json_path = os.path.join(root, wav_file.replace(".wav", ".json"))
        if not os.path.exists(json_path):
            skipped += 1
            continue
        print(f"\n Processing {wav_file}...")
        try:
            y, sr = librosa.load(wav_path, sr=None)
            y = preprocess_audio(y, sr)
            y_harmonic, _ = librosa.effects.hpss(y)
            est_t, est_f = extract_pitch_torchcrepe(y_harmonic, sr)
            pred_notes = frames_to_notes_advanced(est_t, est_f, min_note_sec=MIN_NOTE_SEC)

            # Preserve GTSinger structure
            rel_dir = os.path.relpath(root, root_dir)
            out_subdir = os.path.join(output_dir, rel_dir)
            os.makedirs(out_subdir, exist_ok=True)

            # Save predicted XML
            out_path = os.path.join(out_subdir, wav_file.replace(".wav", "_predicted.musicxml"))
            saved_xml_path = write_notes_to_musicxml(pred_notes, out_path)

            # Generate harmony
            write_harmonized_score_from_xml(saved_xml_path)

            # Evaluate
            ref_t, ref_f = json_to_pitch_sequence(json_path, sr=FRAME_RATE)
            if ref_t is None:
                skipped += 1
                continue
            max_t = min(est_t[-1], ref_t[-1])
            t_common = np.arange(0, max_t, 1.0 / FRAME_RATE)
            est_interp = np.interp(t_common, est_t, est_f)
            ref_interp = np.interp(t_common, ref_t, ref_f)
            results_raw.append(melody_eval(t_common, ref_interp, t_common, est_interp))

            ref_v, est_v = ref_interp[ref_interp > 0], est_interp[est_interp > 0]
            if len(ref_v) > 2 and len(est_v) > 2:
                ali = dtw(ref_v.reshape(-1,1), est_v.reshape(-1,1), keep_internals=True)
                ar, ae = ref_v[ali.index1s], est_v[ali.index2s]
                t_ar, t_ae = np.arange(len(ar))/FRAME_RATE, np.arange(len(ae))/FRAME_RATE
                results_aligned.append(melody_eval(t_ar, ar, t_ae, ae))

            ref_notes = json_to_notes_direct(json_path)
            note_metrics_all.append(evaluate_notes(pred_notes, ref_notes))
            processed += 1
        except Exception as e:
            print(f" Error processing {wav_file}: {e}")
            skipped += 1
            continue


# ---------------- Summaries ---------------------
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
    print("\n🎼 Note-Level Metrics:")
    print(f"avg_precision : {P:.4f}")
    print(f"avg_recall    : {R:.4f}")
    print(f"avg_f_measure : {F:.4f}")

summarize_frame(results_raw, "RAW")
summarize_frame(results_aligned, "DTW-Aligned")
summarize_notes(note_metrics_all)

print(f"\n Summary: Processed {processed}, Skipped {skipped}.")
print(f" Predicted XML root: {output_dir}")
