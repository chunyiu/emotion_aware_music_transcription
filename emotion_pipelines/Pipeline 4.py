"""
Pipeline 4: Librosa + CREPE + Music21 Harmony with Emotion Classification
Uses regular CREPE with Music21 harmony generation
Includes before/after emotion analysis
"""

import numpy as np
import librosa
import crepe
from scipy.signal import medfilt
from pathlib import Path
from music21 import stream, note, chord, tempo, meter, clef, key as m21key, roman, analysis, metadata
import warnings
import pickle
import json
import sys
from music21 import converter as m21_converter 
from music21 import chord as m21chord, note as m21note
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
CREPE_MODEL = 'full'
CREPE_STEP_SIZE = 10
MEDIAN_FILTER_SIZE = 5
CONF_THRESHOLD = 0.45
MIN_NOTE_SEC = 0.15
GAP_JOIN_SEC = 0.20
MIN_Q_LEN_BEATS = 0.25
DEFAULT_BPM = 120


# ============================================================================
# CREPE PITCH DETECTION
# ============================================================================
def extract_f0_conf_crepe(audio_path, model=CREPE_MODEL, step_size=CREPE_STEP_SIZE):
    """Extract F0 and confidence using CREPE"""
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    
    time, frequency, confidence, activation = crepe.predict(
        y, sr,
        model_capacity=model,
        step_size=step_size,
        viterbi=True
    )
    
    return time, frequency, confidence


# ============================================================================
# NOTE SEGMENTATION
# ============================================================================
def segment_notes(times, f0, conf, conf_threshold=CONF_THRESHOLD, min_note_sec=MIN_NOTE_SEC, gap_join_sec=GAP_JOIN_SEC):
    """Segment pitch into notes"""
    f0_filt = f0.copy()
    f0_filt[conf < conf_threshold] = 0.0
    
    f0_smooth = medfilt(f0_filt, kernel_size=MEDIAN_FILTER_SIZE)
    
    midi = np.zeros_like(f0_smooth)
    mask_voiced = (f0_smooth > 0)
    midi[mask_voiced] = librosa.hz_to_midi(f0_smooth[mask_voiced])
    
    notes = []
    in_note = False
    note_start_idx = None
    note_pitches = []
    
    for i in range(len(midi)):
        if midi[i] > 0:
            if not in_note:
                in_note = True
                note_start_idx = i
                note_pitches = [midi[i]]
            else:
                if len(note_pitches) > 0:
                    pitch_diff = abs(midi[i] - np.median(note_pitches))
                    if pitch_diff > 0.5:
                        note_end_idx = i - 1
                        duration = times[note_end_idx] - times[note_start_idx]
                        if duration >= min_note_sec:
                            notes.append({
                                'start': times[note_start_idx],
                                'end': times[note_end_idx],
                                'pitch': np.median(note_pitches)
                            })
                        note_start_idx = i
                        note_pitches = [midi[i]]
                    else:
                        note_pitches.append(midi[i])
        else:
            if in_note:
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
    
    if in_note and len(note_pitches) > 0:
        duration = times[-1] - times[note_start_idx]
        if duration >= min_note_sec:
            notes.append({
                'start': times[note_start_idx],
                'end': times[-1],
                'pitch': np.median(note_pitches)
            })
    
    # Merge close notes
    if len(notes) > 1:
        merged = [notes[0]]
        for n in notes[1:]:
            prev = merged[-1]
            gap = n['start'] - prev['end']
            pitch_diff = abs(n['pitch'] - prev['pitch'])
            
            if gap <= gap_join_sec and pitch_diff <= 0.5:
                prev['end'] = n['end']
                prev['pitch'] = (prev['pitch'] + n['pitch']) / 2
            else:
                merged.append(n)
        notes = merged
    
    return notes


# ============================================================================
# HARMONY GENERATION
# ============================================================================
def get_melody_notes_at_measure(melody_part, measure_num, beats_per_measure=4.0):
    """Extract pitch classes in a measure"""
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
    """Score chord fit to melody"""
    try:
        rn = roman.RomanNumeral(chord_rn, key_obj)
        chord_pcs = {p.pitchClass for p in rn.pitches}
    except:
        return -100
    
    if not melody_pcs:
        return 0
    
    score = 0
    notes_in_chord = len(melody_pcs & chord_pcs)
    score += notes_in_chord * 10
    
    notes_not_in_chord = len(melody_pcs - chord_pcs)
    score -= notes_not_in_chord * 5
    
    if chord_rn in ["I", "i"]:
        score += 2
    elif chord_rn in ["IV", "iv", "V", "v"]:
        score += 1
    
    return score


def choose_best_chord(melody_pcs, key_obj, available_chords, prev_chord=None):
    """Choose best chord"""
    best_chord = available_chords[0]
    best_score = -1000
    
    for chord_rn in available_chords:
        score = score_chord_fit(chord_rn, melody_pcs, key_obj)
        
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
    """Generate harmony with Music21"""
    if est_key is None:
        melody_score = stream.Score()
        melody_score.insert(0, melody_part)
        analyzer = analysis.discrete.KrumhanslSchmuckler()
        est_key = analyzer.getSolution(melody_score)
    
    if not isinstance(est_key, m21key.Key):
        est_key = m21key.Key('C')
    
    if est_key.mode == 'minor':
        diatonic_rn = ["i", "iio", "III", "iv", "v", "VI", "VII"]
    else:
        diatonic_rn = ["I", "ii", "iii", "IV", "V", "vi", "viio"]
    
    harmony_part = stream.Part(id="Harmony")
    harmony_part.partName = "Harmony"   # <-- add this line
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
            rn = roman.RomanNumeral("I" if est_key.mode != 'minor' else "i", est_key)
            ch = chord.Chord(rn.pitches)
            ch.quarterLength = 4.0
            harmony_part.append(ch)
    
    return harmony_part, est_key


# ============================================================================
# MUSICXML EXPORT
# ============================================================================
def create_score_with_harmony(notes, output_path, title="CREPE + Harmony"):
    """Create score with harmony"""
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
    
    harmony_part, est_key = generate_harmony(melody)
    
    score = stream.Score()
    score.insert(0, metadata.Metadata())
    score.metadata.title = title
    score.metadata.composer = f"Pipeline 4: CREPE + Music21 | Key: {est_key}"
    
    score.append(melody)
    score.append(harmony_part)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    score.write('musicxml', fp=str(output_path))
    
    return len(notes), est_key




def extract_harmony_only_xml(full_xml_path, harmony_only_xml_path):
    try:
        score = m21_converter.parse(full_xml_path)

        harmony_score = stream.Score()

        if score.metadata is not None:
            harmony_score.insert(0, score.metadata)

        if score.parts:
            first_part = score.parts[0]
            ts = first_part.recurse().getElementsByClass(meter.TimeSignature).first()
            mm = first_part.recurse().getElementsByClass(tempo.MetronomeMark).first()
            if ts is not None:
                harmony_score.insert(0, ts)
            if mm is not None:
                harmony_score.insert(0, mm)

        # collect harmony parts
        harmony_parts = []
        for p in score.parts:
            pid_raw = getattr(p, "id", "")
            pname_raw = getattr(p, "partName", "")
            pid = str(pid_raw).lower()
            pname = str(pname_raw).lower()
            if "harm" in pid or "harm" in pname:
                harmony_parts.append(p)

        if not harmony_parts:
            print("  [HARMONY] No harmony part detected in MusicXML.")
            return False

        # Build a *monophonic* part called "Melody" from the harmony chords
        mono_part = stream.Part(id="Melody")
        mono_part.partName = "Melody (Harmony-only)"

        # take the first harmony part only (or merge more if you want)
        harmony_part = harmony_parts[0]

        for el in harmony_part.recurse():
            if isinstance(el, m21chord.Chord):
                # choose root or first note as a single note
                root_pitch = el.root() if el.root() is not None else el.pitches[0]
                n = m21note.Note(root_pitch)
                n.quarterLength = el.quarterLength
                # keep offset by placing with .offset if needed
                mono_part.insert(el.offset, n)
            elif isinstance(el, m21note.Note):
                # already monophonic content
                mono_part.insert(el.offset, el)
            # ignore other element types for simple synthesis

        harmony_score.append(mono_part)

        harmony_only_xml_path.parent.mkdir(parents=True, exist_ok=True)
        harmony_score.write("musicxml", fp=str(harmony_only_xml_path))
        print(f"  [HARMONY] Harmony-only MusicXML written to: {harmony_only_xml_path}")
        return True

    except Exception as e:
        print(f"  [HARMONY] Failed to extract harmony-only MusicXML: {e}")
        return False



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
            print(f"  [PROCESS] Extracting pitch with CREPE...")
            times, f0, conf = extract_f0_conf_crepe(audio_path)
            
            # Segment notes
            notes = segment_notes(times, f0, conf)
            
            # Generate score with harmony
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


            # ---------------------------------------------------------
            # HARMONY-ONLY EMOTION: extract harmony, render, classify
            # ---------------------------------------------------------
            harmony_emotion = None
            harmony_xml_path = musicxml_dir / f"{stem}_harmony_only.musicxml"
            harmony_wav_path = wav_after_dir / f"{stem}_harmony_only.wav"

            try:
                ok_harmony_xml = extract_harmony_only_xml(musicxml_path, harmony_xml_path)
                if ok_harmony_xml:
                    print(f"  [HARMONY] Converting harmony-only MusicXML to WAV...")
                    harm_success = convert_musicxml_to_wav(harmony_xml_path, harmony_wav_path, method='simple')

                    if harm_success and harmony_wav_path.exists():
                        harmony_emotion = emotion_classifier.predict(str(harmony_wav_path))
                        if harmony_emotion:
                            print(
                                f"    [HARMONY] Top: {harmony_emotion['top_emotion']} "
                                f"({harmony_emotion['top_confidence']:.2%})"
                            )
                    else:
                        print("    [HARMONY] Failed to generate harmony-only WAV.")
                else:
                    print("    [HARMONY] Skipping harmony-only emotion (no harmony part).")
            except Exception as e:
                print(f"    [HARMONY] Exception during harmony-only processing: {e}")
                import traceback
                traceback.print_exc()

            # ---------------------------------------------------------
            # Write separate JSON: harmony vs ORIGINAL (before) emotion
            # ---------------------------------------------------------
            harmony_json = {
                "file": audio_path.name,
                "melody_top1": emotion_before["top_emotion"] if emotion_before else None,
                "melody_confidence": float(emotion_before["top_confidence"]) if emotion_before else None,
                "harmony_top1": harmony_emotion["top_emotion"] if harmony_emotion else None,
                "harmony_confidence": float(harmony_emotion["top_confidence"]) if harmony_emotion else None,
                "match": (
                    emotion_before is not None
                    and harmony_emotion is not None
                    and emotion_before["top_emotion"] == harmony_emotion["top_emotion"]
                ),
                "musicxml_full": str(musicxml_path.relative_to(output_dir)),
                "musicxml_harmony_only": (
                    str(harmony_xml_path.relative_to(output_dir))
                    if harmony_xml_path.exists() else None
                ),
                "wav_harmony_only": (
                    str(harmony_wav_path.relative_to(output_dir))
                    if harmony_wav_path.exists() else None
                ),
                "note": "Harmony emotion compared against ORIGINAL (before) emotion.",
            }

            harmony_json_path = emotion_dir / f"{stem}_harmony_emotion.json"
            with open(harmony_json_path, "w") as f:
                json.dump(harmony_json, f, indent=2)
            print(f"  [HARMONY] Saved harmony-only emotion JSON to {harmony_json_path.name}")

            
            
            # Save comparison results
            result = {
                'file': audio_path.name,
                'notes': num_notes,
                'key': str(est_key),
                'musicxml': str(musicxml_path.relative_to(output_dir)),
                'wav_after': str(wav_after_path.relative_to(output_dir)) if success else None,

                # Existing emotions
                'emotion_before': emotion_before,
                'emotion_after': emotion_after,

                # NEW — harmony-only emotion
                'harmony_emotion': harmony_emotion,

                # NEW — Quick comparison info
                'harmony_matches_original':
                    (
                        emotion_before is not None and
                        harmony_emotion is not None and
                        emotion_before["top_emotion"] == harmony_emotion["top_emotion"]
                    ),

                # NEW — Paths for reference
                'harmony_xml': str(harmony_xml_path.relative_to(output_dir)) 
                            if harmony_xml_path.exists() else None,

                'harmony_wav': str(harmony_wav_path.relative_to(output_dir))
                            if harmony_wav_path.exists() else None,

                # Metadata
                'note': 'Includes AFTER (melody+harmony) and harmony-only emotion analysis.',
                'success': True
            }

            
            result_json_path = emotion_dir / f"{stem}_emotion_comparison.json"
            with open(result_json_path, 'w') as f:
                json.dump(result, f, indent=2)
            
            results.append(result)
            print(f"  ✓ Saved to {result_json_path.name}\n")
            
        except Exception as e:
            print(f"  ✗ Error: {e}\n")
            import traceback
            traceback.print_exc()
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
    
    summary_path = output_dir / "pipeline_4_summary.json"
    summary = {
        'pipeline': 'Pipeline 4: CREPE + Music21 Harmony',
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
    OUTPUT_DIR = Path(__file__).parent / "output_pipeline_4_emotion"
    MODEL_DIR = BASE_DIR / "results" / "emotion_model"
    MAX_FILES = None
    
    print("="*60)
    print("PIPELINE 4: CREPE + HARMONY + EMOTION")
    print("="*60)
    print(f"Input:  {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Model:  {CREPE_MODEL}")
    print(f"Emotion Model: {MODEL_DIR}")
    print("="*60)
    print()
    
    process_dataset(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        model_dir=MODEL_DIR,
        max_files=MAX_FILES
    )
