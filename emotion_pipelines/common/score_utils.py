"""
Music21 score construction utilities shared across pipelines.
"""

import numpy as np
from fractions import Fraction
from music21 import stream, note, chord, tempo, meter, metadata, duration, tie, clef


DEFAULT_BPM = 120
MIN_Q_LEN_BEATS = 0.25

# Legal quarter lengths for clean MusicXML export
LEGAL_QL = (2.0, 1.5, 1.0, 0.75, 0.5, 0.375, 0.25, 0.125)


def notes_to_music21_part(notes, bpm=DEFAULT_BPM, part_id="Melody"):
    """
    Convert a list of note dicts to a music21 Part.

    Args:
        notes: List of dicts with start, end, pitch_midi keys.
        bpm: Tempo in beats per minute.
        part_id: Part identifier.

    Returns:
        music21 stream.Part
    """
    sec_per_beat = 60.0 / bpm
    part = stream.Part(id=part_id)
    part.append(meter.TimeSignature('4/4'))
    part.append(tempo.MetronomeMark(number=bpm))

    for n in notes:
        pitch_midi = int(round(n['pitch_midi']))
        duration_sec = n['end'] - n['start']
        ql = max(MIN_Q_LEN_BEATS,
                 round(duration_sec / sec_per_beat / MIN_Q_LEN_BEATS) * MIN_Q_LEN_BEATS)

        music_note = note.Note()
        music_note.pitch.midi = pitch_midi
        music_note.quarterLength = ql
        part.append(music_note)

    return part


def create_score(melody_part, harmony_part=None, title="", composer=""):
    """
    Combine melody and optional harmony parts into a Score.

    Returns:
        music21 stream.Score
    """
    score = stream.Score()
    score.insert(0, metadata.Metadata())
    score.metadata.title = title
    score.metadata.composer = composer

    score.append(melody_part)
    if harmony_part is not None:
        score.append(harmony_part)

    return score


def extract_harmony_only(score):
    """
    Extract harmony part(s) from a score.

    Returns:
        music21 stream.Score with only harmony parts, or None if not found.
    """
    harmony_parts = []
    for p in score.parts:
        pid = (getattr(p, "id", "") or "").lower()
        pname = (getattr(p, "partName", "") or "").lower()
        if "harm" in pid or "harm" in pname:
            harmony_parts.append(p)

    # Fallback: if no explicit harmony, use non-first parts
    if not harmony_parts and len(score.parts) >= 2:
        harmony_parts = list(score.parts)[1:]

    if not harmony_parts:
        return None

    harmony_score = stream.Score()
    for hp in harmony_parts:
        harmony_score.append(hp)
    return harmony_score


def export_musicxml_safely(score, fp, ts='4/4', denom_limit=8):
    """
    Export a score to MusicXML with duration cleanup.
    Handles inexpressible durations and tiny notes.
    """
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

    # Fix bad durations
    bad = []
    for part in cleaned_score.parts:
        for n in part.recurse().notesAndRests:
            t = getattr(n.duration, 'type', None)
            if t in ('inexpressible', 'complex') or n.duration.quarterLength <= 0:
                bad.append(n)

    if bad:
        for n in bad:
            n.duration = duration.Duration(1.0)
            if hasattr(n, 'tie'):
                n.tie = None
        cleaned_score.makeMeasures(inPlace=True)
        cleaned_score.makeNotation(inPlace=True)
        cleaned_score.stripTies(inPlace=True)
        cleaned_score.sort()

    cleaned_score.write('musicxml', fp=fp)
    return cleaned_score


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _nearest_legal_ql(ql, legal=LEGAL_QL):
    return min(legal, key=lambda x: abs(x - ql))


def _split_into_legal_segments(ql, legal=LEGAL_QL):
    chunks = []
    remaining = float(ql)
    eps = 1e-9
    while remaining > eps:
        candidates = [l for l in legal if l <= remaining + eps]
        if not candidates:
            if chunks:
                chunks[-1] += remaining
            else:
                chunks.append(_nearest_legal_ql(remaining, legal))
            break
        l = max(candidates)
        if remaining - l < eps * 10:
            chunks.append(l)
            break
        chunks.append(l)
        remaining -= l
    return chunks


def _retime_part_flat(part, denom_limit=8):
    part_flat = part.flatten().sorted()
    cleaned = stream.Part()
    cleaned.id = part.id
    cleaned.partName = getattr(part, 'partName', None)

    cur_offset = 0.0
    min_ql = 0.125

    for el in part_flat.notesAndRests:
        raw_ql = float(el.duration.quarterLength)
        if raw_ql <= 0:
            raw_ql = 1.0 / denom_limit
        if raw_ql < min_ql:
            raw_ql = min_ql

        ql = float(Fraction(raw_ql).limit_denominator(denom_limit))
        segs = _split_into_legal_segments(ql, LEGAL_QL)

        def _clone_like(e):
            if isinstance(e, note.Note):
                c = note.Note(e.pitch)
                for ly in getattr(e, 'lyrics', []) or []:
                    t = getattr(ly, 'text', None)
                    if t:
                        c.addLyric(t)
                return c
            elif isinstance(e, chord.Chord):
                c = chord.Chord(e.pitches)
                for ly in getattr(e, 'lyrics', []) or []:
                    t = getattr(ly, 'text', None)
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
                    new_el.tie = tie.Tie('start')
                elif i == len(segs) - 1:
                    new_el.tie = tie.Tie('stop')
                else:
                    new_el.tie = tie.Tie('continue')
            cur_offset += seg

    return cleaned
