import os
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
output_dir = "./predicted_torchcrepe_xml_folder"
os.makedirs(output_dir, exist_ok=True)

FRAME_RATE          = 100
MEDIAN_FILTER_SIZE  = 5
FMIN_HZ, FMAX_HZ    = 50.0, 1100.0
CONF_THRESHOLD      = 0.45
MIN_NOTE_SEC        = 0.15
GAP_JOIN_SEC        = 0.20
MIN_Q_LEN_BEATS     = 0.25
DEFAULT_BPM         = 120
PITCH_TOL_SEMITONES = 0.75
ONSET_TOL_SEC       = 0.25
OFFSET_TOL_SEC      = 0.30
CHUNK_SECONDS       = 10

# Harmony range
HARMONY_MIN_MIDI = 55   # G3
HARMONY_MAX_MIDI = 81   # A5
# =============================================================

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# -------------------------- Utils ----------------------------
def ensure_unique_path(path):
    base, ext = os.path.splitext(path)
    k = 1
    out = path
    while os.path.exists(out):
        out = f"{base}_v{k}{ext}"
        k += 1
    return out

def preprocess_audio(y, sr):
    """Enhanced preprocessing for better pitch tracking."""
    from scipy.signal import butter, filtfilt
    # More aggressive silence trimming
    y, _ = librosa.effects.trim(y, top_db=20)
    # High-pass filter to remove low-frequency rumble
    b, a = butter(5, 80, btype='high', fs=sr)
    y = filtfilt(b, a, y)
    # Normalize with headroom
    y = y / (np.max(np.abs(y)) + 1e-9) * 0.95
    return y

def smooth_pitch(freqs, window=MEDIAN_FILTER_SIZE):
    if window > 1:
        return medfilt(freqs, kernel_size=window)
    return freqs

def hz_to_midi(hz):
    if hz <= 0:
        return None
    return 69 + 12 * np.log2(hz / 440.0)

def get_score_bpm(score):
    try:
        mm = next((mm for _, mm in score.metronomeMarkBoundaries() if mm is not None), None)
        if mm and hasattr(mm, 'number') and mm.number:
            return float(mm.number)
    except Exception:
        pass
    return float(DEFAULT_BPM)

# === Reference XML to frame pitch ===
def musicxml_to_pitch_sequence(xml_file, sr=FRAME_RATE):
    score = converter.parse(xml_file)
    bpm   = get_score_bpm(score)
    sec_per_beat = 60.0 / bpm
    notes = score.flatten().notes
    if not notes:
        return None, None
    end_beats = max(n.offset + n.quarterLength for n in notes)
    end_time  = end_beats * sec_per_beat
    times = np.arange(0, end_time, 1.0 / sr)
    freqs = np.zeros_like(times)
    for n in notes:
        if isinstance(n, note.Note):
            hz = n.pitch.frequency
            start_sec = n.offset * sec_per_beat
            end_sec   = (n.offset + n.quarterLength) * sec_per_beat
            start_idx = int(start_sec * sr)
            end_idx   = min(int(end_sec * sr), len(freqs))
            if end_idx > start_idx:
                freqs[start_idx:end_idx] = hz
    return times, freqs

# ================= Pitch extraction (separate tracks) =================
def extract_f0_conf(y, sr, model='full'):
    """Return times, continuous f0 (confidence-gated), and confidence."""
    hop_length = int(sr / FRAME_RATE)
    audio = torch.tensor(y, dtype=torch.float32).unsqueeze(0).to(device)
    max_chunk = int(sr * CHUNK_SECONDS)
    T = audio.shape[1]
    num_chunks = max(1, int(np.ceil(T / max_chunk)))

    f0_list, conf_list = [], []
    with torch.no_grad():
        for i in range(num_chunks):
            s, e = i * max_chunk, min((i + 1) * max_chunk, T)
            chunk = audio[:, s:e]
            torch.cuda.empty_cache()
            f0, conf = torchcrepe.predict(
                chunk, sample_rate=sr, hop_length=hop_length,
                fmin=FMIN_HZ, fmax=FMAX_HZ, model=model,
                device=device, pad=True, batch_size=64,
                return_periodicity=True
            )
            f0_list.append(f0.cpu())
            conf_list.append(conf.cpu())

    f0_hz = torch.cat(f0_list, dim=1).squeeze(0).numpy()
    conf  = torch.cat(conf_list, dim=1).squeeze(0).numpy()

    # light smoothing for stability
    f0_hz = medfilt(f0_hz, kernel_size=3)
    conf  = medfilt(conf,   kernel_size=3)

    # confidence-based voicing for FRAME metrics
    voiced = (conf >= CONF_THRESHOLD) & (f0_hz >= FMIN_HZ)
    f0_cont = np.where(voiced, f0_hz, 0.0)

    times = np.arange(len(f0_cont)) * (hop_length / sr)
    return times, f0_cont, conf

def snap_to_semitone_only_for_notes(f0_cont):
    """Use snapped f0 ONLY for note segmentation; keep continuous for frame eval."""
    f = f0_cont.copy()
    voiced = f > 0
    if np.any(voiced):
        f[voiced] = 440 * 2 ** (np.round(12 * np.log2(f[voiced] / 440)) / 12)
    return f

# === Convert frames → notes (your enhanced version) ===
def frames_to_notes(times, freqs, min_note_sec=MIN_NOTE_SEC, gap_join_sec=GAP_JOIN_SEC):
    if len(freqs) == 0:
        return []
    voiced = freqs > 0
    frame_dur = np.median(np.diff(times))
    gap_frames = int(0.05 / frame_dur)
    for i in range(1, len(voiced) - gap_frames):
        if not voiced[i] and np.any(voiced[i-gap_frames:i]) and np.any(voiced[i+1:i+gap_frames]):
            voiced[i] = True
    freqs = freqs * voiced

    events = []
    start_idx, last_pitch = None, None
    for i in range(1, len(freqs) + 1):
        cur = freqs[i-1] if i-1 < len(freqs) else 0.0
        nxt = freqs[i]   if i   < len(freqs) else 0.0
        cur_note = hz_to_midi(cur) if cur > 0 else None
        nxt_note = hz_to_midi(nxt) if nxt > 0 else None

        if cur_note is not None and start_idx is None:
            start_idx, last_pitch = i - 1, cur_note

        change = (
            (cur_note is None and last_pitch is not None) or
            (cur_note is not None and nxt_note is None) or
            (cur_note is not None and nxt_note is not None and abs(cur_note - nxt_note) >= 0.75)
        )
        if start_idx is not None and change:
            end_idx = i
            onset, offset = times[start_idx], times[end_idx - 1]
            dur = max(0.0, offset - onset)
            if dur >= min_note_sec:
                seg_midis = [hz_to_midi(f) for f in freqs[start_idx:end_idx] if f > 0]
                if seg_midis:
                    midi_val = float(np.median(seg_midis))
                    events.append([onset, offset, midi_val])
            start_idx, last_pitch = None, None

    merged = []
    if events:
        merged = [events[0]]
        for (on2, off2, m2) in events[1:]:
            on1, off1, m1 = merged[-1]
            if abs(m1 - m2) <= 0.5 and (on2 - off1) <= gap_join_sec:
                merged[-1][1] = off2
                merged[-1][2] = (m1 + m2) / 2.0
            else:
                merged.append([on2, off2, m2])
    return merged

# === MusicXML Writing & Metrics ===
def write_notes_to_musicxml(notes_list, output_path, bpm=DEFAULT_BPM):
    sc, p = stream.Score(), stream.Part()
    p.insert(0, tempo.MetronomeMark(number=bpm))
    p.insert(0, meter.TimeSignature("4/4"))
    p.insert(0, clef.TrebleClef())
    sec_per_beat = 60.0 / bpm
    for onset, offset, midi_val in notes_list:
        dur_sec = max(0.0, offset - onset)
        if dur_sec < MIN_NOTE_SEC:
            continue
        ql = max(MIN_Q_LEN_BEATS, round((dur_sec / sec_per_beat) / (1/32)) * (1/32))
        n = note.Note(int(round(midi_val)))
        n.duration.quarterLength = ql
        p.append(n)
    sc.insert(0, p)
    out = ensure_unique_path(output_path)
    sc.write("musicxml", fp=out)
    print(f"🎼 Saved predicted MusicXML: {out}")
    return out

def evaluate_notes(pred_notes, ref_notes):
    if not pred_notes and not ref_notes:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred_notes:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    if not ref_notes:
        return {"precision": 0.0, "recall": 1.0, "f1": 0.0}
    ref_used = np.zeros(len(ref_notes), dtype=bool)
    tp = 0
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
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = (2*prec*rec)/(prec+rec) if (prec+rec) else 0.0
    return {"precision": prec, "recall": rec, "f1": f1}

def musicxml_to_note_events(xml_file):
    sc = converter.parse(xml_file)
    bpm = get_score_bpm(sc)
    spb = 60.0 / bpm
    evs = []
    for n in sc.flatten().notes:
        if isinstance(n, note.Note):
            on, off = n.offset * spb, (n.offset + n.quarterLength) * spb
            midi_val = n.pitch.midi
            if off > on:
                evs.append((on, off, midi_val))
    return evs

# ---------------- Harmony generation ----------------
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
    low = f"{tonic}1"
    high = f"{tonic}7"
    return sca.getPitches(low, high)

def diatonic_step(p, k, steps):
    plist = build_scale_pitch_list(k)
    midis = [pp.midi for pp in plist]
    m = p.midi
    # nearest index
    idx = min(range(len(midis)), key=lambda i: abs(midis[i] - m))
    idx_target = int(np.clip(idx + steps, 0, len(plist) - 1))
    return plist[idx_target]

def generate_harmony_part_from_melody(melody_part, k, prefer_third_below=True):
    harm = stream.Part()
    harm.insert(0, tempo.MetronomeMark(number=120))
    harm.insert(0, meter.TimeSignature("4/4"))
    harm.insert(0, clef.TrebleClef())
    for n in melody_part.recurse().notes:
        if not isinstance(n, note.Note):
            continue
        src_pitch = n.pitch
        # 3rd below by default; if out of range, try 6th above
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

def write_harmonized_score_from_xml(melody_xml_path, out_path):
    sc = converter.parse(melody_xml_path)
    melody_part = sc.parts[0] if sc.parts else sc.flatten().parts[0]
    k = detect_key_from_score(sc)
    harm_part = generate_harmony_part_from_melody(melody_part, k, prefer_third_below=True)
    new_sc = stream.Score()
    for el in melody_part.getElementsByClass((tempo.MetronomeMark, meter.TimeSignature, clef.Clef)):
        new_sc.insert(0, el)
    melody_part.id = "Melody"
    harm_part.id = "Harmony"
    new_sc.insert(0, melody_part)
    new_sc.insert(0, harm_part)
    out = ensure_unique_path(out_path)
    new_sc.write('musicxml', fp=out)
    print(f"🎶 Saved harmonized score: {out}")
    return out

# ===================== MAIN LOOP =====================
results_raw, results_aligned, note_metrics_all = [], [], []
processed, skipped = 0, 0

for root, _, files in os.walk(root_dir):
    wav_files = [f for f in files if f.endswith(".wav")]
    if not wav_files:
        continue
    rel_path = os.path.relpath(root, root_dir)
    dest_dir = os.path.join(output_dir, rel_path)
    os.makedirs(dest_dir, exist_ok=True)

    for wav_file in wav_files:
        wav_path = os.path.join(root, wav_file)
        xml_path = os.path.join(root, wav_file.replace(".wav", ".musicxml"))
        out_path = os.path.join(dest_dir, wav_file.replace(".wav", "_predicted.musicxml"))
        if not os.path.exists(xml_path):
            print(f"Skipping {wav_path}, no MusicXML reference.")
            skipped += 1
            continue

        print(f"\n Processing {wav_file}...")
        try:
            # ----- Enhanced preprocessing + harmonic separation -----
            y, sr = librosa.load(wav_path, sr=None)
            y = preprocess_audio(y, sr)
            
            # Extract harmonic component for cleaner pitch tracking
            y_harmonic, _ = librosa.effects.hpss(y)
            
            # ----- Extract f0 (continuous for frames; snapped for notes) -----
            est_t, f0_cont, conf = extract_f0_conf(y_harmonic, sr)
            f0_note = snap_to_semitone_only_for_notes(f0_cont)

            # ----- Notes + XML -----
            pred_notes = frames_to_notes(est_t, f0_note)
            saved_xml_path = write_notes_to_musicxml(pred_notes, out_path, bpm=DEFAULT_BPM)

            # Harmony XML
            harm_out_path = os.path.splitext(saved_xml_path)[0] + "_harmonized.musicxml"
            write_harmonized_score_from_xml(saved_xml_path, harm_out_path)

            # ----- Frame metrics (use continuous f0) -----
            ref_t, ref_f = musicxml_to_pitch_sequence(xml_path, sr=FRAME_RATE)
            if ref_t is None or len(ref_t) < 2 or len(est_t) < 2:
                skipped += 1
                continue
            max_t = min(est_t[-1], ref_t[-1])
            if max_t <= 0:
                skipped += 1
                continue
            t_common = np.arange(0, max_t, 1.0/FRAME_RATE)
            est_interp = np.interp(t_common, est_t, f0_cont)   # <— continuous track here
            ref_interp = np.interp(t_common, ref_t, ref_f)
            raw_scores = melody_eval(t_common, ref_interp, t_common, est_interp)
            results_raw.append(raw_scores)

            # DTW-aligned frame metrics (still on continuous f0)
            ref_v, est_v = ref_interp[ref_interp > 0], est_interp[est_interp > 0]
            if len(ref_v) > 2 and len(est_v) > 2:
                ali = dtw(ref_v.reshape(-1,1), est_v.reshape(-1,1), keep_internals=True)
                ar, ae = ref_v[ali.index1s], est_v[ali.index2s]
                t_ar, t_ae = np.arange(len(ar))/FRAME_RATE, np.arange(len(ae))/FRAME_RATE
                aligned_scores = melody_eval(t_ar, ar, t_ae, ae)
                results_aligned.append(aligned_scores)

            # ----- Note metrics (same as your original) -----
            ref_notes = musicxml_to_note_events(xml_path)
            pred_notes_eval = [(on, off, round(m)) for on, off, m in pred_notes]
            ref_notes_eval  = [(on, off, round(m)) for on, off, m in ref_notes]
            nm = evaluate_notes(pred_notes_eval, ref_notes_eval)
            note_metrics_all.append(nm)
            processed += 1
            print(f" Done: {wav_file}")

        except Exception as e:
            print(f" Error processing {wav_file}: {e}")
            skipped += 1
            continue

# ===================== Summaries =====================
def summarize_frame(results, title):
    if not results:
        print(f"\n No valid {title} results.")
        return
    avg = {f"avg_{k.replace(' ', '_').lower()}": np.mean([r[k] for r in results]) for k in results[0]}
    print(f"\n GTSinger Evaluation Results ({title}):")
    for k, v in avg.items():
        print(f"{k}: {v:.4f}")

def summarize_notes(nmetrics):
    if not nmetrics:
        print("\n No valid Note-level results.")
        return
    P = np.mean([m['precision'] for m in nmetrics])
    R = np.mean([m['recall'] for m in nmetrics])
    F = np.mean([m['f1'] for m in nmetrics])
    print("\n Note-Level Metrics (overall):")
    print(f"avg_precision          : {P:.4f}")
    print(f"avg_recall             : {R:.4f}")
    print(f"avg_f_measure (F1)     : {F:.4f}")

summarize_frame(results_raw, "Frame RAW")
summarize_frame(results_aligned, "Frame DTW-Aligned")
summarize_notes(note_metrics_all)

print(f"\n Summary: Processed {processed}, Skipped {skipped}.")
print(f" Predicted XML root: {output_dir}")