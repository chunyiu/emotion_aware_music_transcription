"""
Four distinct harmony generation methods for Stage 2 pipelines.

Each function takes a music21 melody Part and returns a harmony Part.
"""

import numpy as np
from music21 import stream, note, chord, meter, instrument, clef, roman, analysis
from music21 import key as m21key


# ============================================================================
# Pipeline 1: Simple interval transposition
# ============================================================================

def harmony_transpose(melody_part, interval_str='-M3'):
    """
    Generate harmony by transposing the melody by a fixed interval.

    Args:
        melody_part: music21 stream.Part with the melody
        interval_str: Interval string (e.g., '-M3' for major third down)

    Returns:
        music21 stream.Part containing the harmony
    """
    harmony_part = melody_part.transpose(interval_str)
    harmony_part.id = 'Harmony'
    harmony_part.partName = "Harmony"
    harmony_part.insert(0, instrument.fromString('Voice'))
    harmony_part.insert(0, clef.TenorClef())
    return harmony_part


# ============================================================================
# Pipeline 2: Mingus algorithmic chord analysis
# ============================================================================

def harmony_mingus_chord(melody_part):
    """
    Generate harmony using mingus chord determination.
    Analyzes note content per measure and generates a bass root note.

    Returns:
        music21 stream.Part containing bass harmony
    """
    import re
    try:
        import mingus.core.chords as m_chords
    except ImportError:
        print("Warning: mingus not available, returning empty harmony")
        return stream.Part(id='Harmony')

    # Ensure measures exist
    temp_part = melody_part.makeNotation()

    harmony_part = stream.Part(id='Harmony')
    harmony_part.partName = "Harmony"
    harmony_part.insert(0, instrument.fromString('Acoustic Bass'))

    for m in temp_part.getElementsByClass('Measure'):
        measure_notes = []
        for n in m.notes:
            if n.isNote:
                measure_notes.append(n.pitch.name)
            elif n.isChord:
                measure_notes.append(n.root().name)

        if not measure_notes:
            harmony_part.insert(m.offset, note.Rest(
                quarterLength=m.duration.quarterLength
            ))
            continue

        # Determine chord using mingus
        try:
            determined = m_chords.determine(measure_notes, shorthand=True)
            if determined:
                raw_chord = determined[0]
                match = re.match(r'^([A-Ga-g][#b-]?)', raw_chord)
                root_name = match.group(1).upper() if match else measure_notes[0]
            else:
                root_name = measure_notes[0]
        except Exception:
            root_name = measure_notes[0]

        harmony_note = note.Note(f"{root_name}3")
        harmony_note.duration = m.duration
        harmony_part.insert(m.offset, harmony_note)

    return harmony_part


# ============================================================================
# Pipeline 3: Diatonic Roman numeral chords (Krumhansl-Schmuckler)
# ============================================================================

def harmony_diatonic_roman(melody_part, est_key=None):
    """
    Generate harmony using Krumhansl-Schmuckler key detection and
    diatonic Roman numeral chord selection per measure.

    Returns:
        (harmony_part, est_key) tuple
    """
    if est_key is None:
        est_key = _detect_key(melody_part)

    if not isinstance(est_key, m21key.Key):
        est_key = m21key.Key('C')

    # Diatonic chords
    if est_key.mode == 'minor':
        diatonic_rn = ["i", "iio", "III", "iv", "v", "VI", "VII"]
    else:
        diatonic_rn = ["I", "ii", "iii", "IV", "V", "vi", "viio"]

    harmony_part = stream.Part(id="Harmony")
    harmony_part.partName = "Harmony"
    harmony_part.append(meter.TimeSignature('4/4'))

    q_len = melody_part.duration.quarterLength
    num_measures = int(np.ceil(q_len / 4.0))

    prev_chord_rn = None
    for measure_num in range(1, num_measures + 1):
        melody_pcs = _get_melody_pcs_at_measure(melody_part, measure_num)
        best_chord_rn = _choose_best_diatonic_chord(
            melody_pcs, est_key, diatonic_rn, prev_chord_rn
        )

        try:
            rn = roman.RomanNumeral(best_chord_rn, est_key)
            ch = chord.Chord(rn.pitches)
            ch.quarterLength = 4.0
            harmony_part.append(ch)
            prev_chord_rn = best_chord_rn
        except Exception:
            fallback = "I" if est_key.mode != 'minor' else "i"
            rn = roman.RomanNumeral(fallback, est_key)
            ch = chord.Chord(rn.pitches)
            ch.quarterLength = 4.0
            harmony_part.append(ch)

    return harmony_part, est_key


# ============================================================================
# Pipeline 4: Circle-of-fifths voice leading with secondary dominants
# ============================================================================

def harmony_circle_of_fifths(melody_part, est_key=None):
    """
    Generate harmony using circle-of-fifths progressions with secondary dominants.

    Key differences from Pipeline 3 (diatonic_roman):
    - Expanded chord vocabulary including secondary dominants (V/V, V/vi, V/IV)
    - Look-ahead scoring: chords that resolve down a fifth to the next are preferred
    - Richer chromatic color through borrowed dominants

    Returns:
        (harmony_part, est_key) tuple
    """
    if est_key is None:
        est_key = _detect_key(melody_part)

    if not isinstance(est_key, m21key.Key):
        est_key = m21key.Key('C')

    # Expanded chord set: diatonic + secondary dominants
    if est_key.mode == 'minor':
        base_chords = ["i", "iio", "III", "iv", "v", "VI", "VII"]
        secondary_dominants = ["V/III", "V/iv", "V/v", "V/VI"]
    else:
        base_chords = ["I", "ii", "iii", "IV", "V", "vi", "viio"]
        secondary_dominants = ["V/ii", "V/iii", "V/IV", "V/V", "V/vi"]

    all_chords = base_chords + secondary_dominants

    harmony_part = stream.Part(id="Harmony")
    harmony_part.partName = "Harmony"
    harmony_part.append(meter.TimeSignature('4/4'))

    q_len = melody_part.duration.quarterLength
    num_measures = int(np.ceil(q_len / 4.0))

    # Collect melody pitch classes per measure
    measure_pcs = []
    for m_num in range(1, num_measures + 1):
        measure_pcs.append(_get_melody_pcs_at_measure(melody_part, m_num))

    # Forward pass: score chords with look-ahead
    chosen_chords = []
    prev_chord = None

    for m_idx in range(num_measures):
        melody_pcs_here = measure_pcs[m_idx]
        next_pcs = measure_pcs[m_idx + 1] if m_idx + 1 < num_measures else set()

        best_chord = all_chords[0]
        best_score = -1000

        for chord_rn in all_chords:
            score = _score_chord_fit(chord_rn, melody_pcs_here, est_key)

            # Voice leading from previous chord
            if prev_chord:
                score += _voice_leading_score(prev_chord, chord_rn, est_key)

            # Circle-of-fifths look-ahead: does THIS chord resolve nicely
            # to any chord that fits the NEXT measure?
            if next_pcs:
                lookahead_bonus = _fifths_lookahead(
                    chord_rn, next_pcs, est_key, all_chords
                )
                score += lookahead_bonus * 0.5

            # Prefer secondary dominants resolving to their targets
            score += _secondary_dominant_bonus(chord_rn, prev_chord, est_key)

            if score > best_score:
                best_score = score
                best_chord = chord_rn

        chosen_chords.append(best_chord)
        prev_chord = best_chord

    # Build the harmony part from chosen chords
    for chord_rn in chosen_chords:
        try:
            rn = roman.RomanNumeral(chord_rn, est_key)
            ch = chord.Chord(rn.pitches)
            ch.quarterLength = 4.0
            harmony_part.append(ch)
        except Exception:
            fallback = "I" if est_key.mode != 'minor' else "i"
            rn = roman.RomanNumeral(fallback, est_key)
            ch = chord.Chord(rn.pitches)
            ch.quarterLength = 4.0
            harmony_part.append(ch)

    return harmony_part, est_key


# ============================================================================
# Internal helpers
# ============================================================================

def _detect_key(melody_part):
    """Detect key using Krumhansl-Schmuckler algorithm."""
    melody_score = stream.Score()
    melody_score.insert(0, melody_part)
    analyzer = analysis.discrete.KrumhanslSchmuckler()
    return analyzer.getSolution(melody_score)


def _get_melody_pcs_at_measure(melody_part, measure_num, beats_per_measure=4.0):
    """Extract pitch classes of melody notes in a specific measure."""
    notes_in_measure = []
    measure_start = (measure_num - 1) * beats_per_measure
    measure_end = measure_start + beats_per_measure

    for n in melody_part.flatten().notes:
        if hasattr(n, 'pitch'):
            note_offset = float(n.offset)
            if measure_start <= note_offset < measure_end:
                notes_in_measure.append(n.pitch.pitchClass)

    return set(notes_in_measure)


def _score_chord_fit(chord_rn, melody_pcs, key_obj):
    """Score how well a chord fits the melody notes."""
    try:
        rn = roman.RomanNumeral(chord_rn, key_obj)
        chord_pcs = {p.pitchClass for p in rn.pitches}
    except Exception:
        return -100

    if not melody_pcs:
        return 0

    score = 0
    notes_in_chord = len(melody_pcs & chord_pcs)
    score += notes_in_chord * 10
    notes_not_in_chord = len(melody_pcs - chord_pcs)
    score -= notes_not_in_chord * 5

    # Prefer stable chords
    if chord_rn in ("I", "i"):
        score += 2
    elif chord_rn in ("IV", "iv", "V", "v"):
        score += 1

    return score


def _choose_best_diatonic_chord(melody_pcs, key_obj, available_chords, prev_chord=None):
    """Choose best chord for Pipeline 3 (diatonic only)."""
    best_chord = available_chords[0]
    best_score = -1000

    for chord_rn in available_chords:
        score = _score_chord_fit(chord_rn, melody_pcs, key_obj)

        if prev_chord:
            if chord_rn == prev_chord:
                score += 3
            try:
                prev_rn = roman.RomanNumeral(prev_chord, key_obj)
                curr_rn = roman.RomanNumeral(chord_rn, key_obj)
                interval_dist = abs(prev_rn.scaleDegree - curr_rn.scaleDegree)
                if interval_dist in [3, 4, 5]:
                    score += 2
            except Exception:
                pass

        if score > best_score:
            best_score = score
            best_chord = chord_rn

    return best_chord


def _voice_leading_score(prev_chord_rn, curr_chord_rn, key_obj):
    """Score voice leading smoothness between two chords."""
    try:
        prev_rn = roman.RomanNumeral(prev_chord_rn, key_obj)
        curr_rn = roman.RomanNumeral(curr_chord_rn, key_obj)
    except Exception:
        return 0

    score = 0
    # Same chord = continuity
    if prev_chord_rn == curr_chord_rn:
        score += 2

    # Circle of fifths motion (root moves down a fifth / up a fourth)
    try:
        root_interval = abs(prev_rn.scaleDegree - curr_rn.scaleDegree) % 7
        if root_interval == 4:  # down a fifth (e.g., V -> I)
            score += 4
        elif root_interval == 3:  # down a fourth
            score += 2
        elif root_interval in (1, 6):  # step motion
            score += 1
    except Exception:
        pass

    return score


def _fifths_lookahead(chord_rn, next_measure_pcs, key_obj, all_chords):
    """
    Score how well this chord sets up a fifths-resolution
    to a chord that fits the next measure.
    """
    best_bonus = 0

    try:
        curr_rn = roman.RomanNumeral(chord_rn, key_obj)
        curr_root_pc = curr_rn.root().pitchClass
    except Exception:
        return 0

    for next_chord_rn in all_chords[:7]:  # check diatonic chords only for targets
        fit = _score_chord_fit(next_chord_rn, next_measure_pcs, key_obj)
        if fit <= 0:
            continue

        try:
            next_rn = roman.RomanNumeral(next_chord_rn, key_obj)
            next_root_pc = next_rn.root().pitchClass
        except Exception:
            continue

        # Check if current root is a fifth above next root (circle of fifths)
        interval = (curr_root_pc - next_root_pc) % 12
        if interval == 7:  # perfect fifth
            best_bonus = max(best_bonus, 3)
        elif interval == 5:  # perfect fourth (plagal)
            best_bonus = max(best_bonus, 1)

    return best_bonus


def _secondary_dominant_bonus(chord_rn, prev_chord, key_obj):
    """
    Bonus for secondary dominants that properly resolve.
    E.g., V/V should resolve to V.
    """
    if "/" not in chord_rn or not chord_rn.startswith("V/"):
        return 0

    # The target of V/X is X
    target = chord_rn.split("/")[1]

    # If previous chord was V/X and this is X, that's a good resolution
    if prev_chord and prev_chord.startswith("V/"):
        prev_target = prev_chord.split("/")[1]
        if chord_rn.startswith(prev_target):
            return 3

    # A secondary dominant gets a slight bonus just for variety
    return 1
