"""
Compute emotion preservation rate for each Stage 1 pipeline (A–H).

Two rates are reported:
  • before_rate: emotion_before.top_emotion == gt_emotion  (baseline: same for all pipelines)
  • after_rate:  emotion_after.top_emotion  == gt_emotion  (meaningful: differs per pipeline)
                 Only shown when emotion_after has been populated by add_emotion_after.py.

gt_emotion is read from:
  1. The individual *_notes.json file (field 'gt_emotion'), if present.
  2. Fallback: gtsinger_english_emotions.csv, matched via source_audio path.
"""

import glob
import json
import sys
from pathlib import Path

BASE_DIR  = Path(__file__).parent
CSV_PATH  = BASE_DIR.parent / "emotion_pipelines" / "gtsinger_english_emotions.csv"
PIPELINES = sorted([d.name for d in BASE_DIR.iterdir()
                    if d.is_dir() and d.name.startswith("pipeline_")])

sys.path.insert(0, str(BASE_DIR.parent / "emotion_pipelines"))
from common.csv_emotions import load_csv_emotions, source_audio_to_csv_key


def main():
    csv_emotions = load_csv_emotions(CSV_PATH)
    print(f"Loaded {len(csv_emotions):,} entries from GTSinger CSV\n")

    results = {}

    for pipeline_dir_name in PIPELINES:
        pipeline_dir = BASE_DIR / pipeline_dir_name
        label = pipeline_dir_name.replace("pipeline_", "Pipeline ").upper()

        note_files = sorted(glob.glob(str(pipeline_dir / "*_notes.json")))

        total          = 0
        before_preserved = 0
        after_preserved  = 0
        after_available  = 0
        missing_gt     = 0
        csv_fallback   = 0

        for fp in note_files:
            with open(fp, encoding="utf-8") as f:
                data = json.load(f)

            # ── Ground-truth emotion ──────────────────────────────────────
            gt = (data.get("gt_emotion") or "").strip().lower()
            if not gt:
                key = source_audio_to_csv_key(data.get("source_audio", ""))
                gt = csv_emotions.get(key, "").strip().lower()
                if gt:
                    csv_fallback += 1
            if not gt:
                missing_gt += 1
                continue

            total += 1

            # ── Emotion before (source audio) ─────────────────────────────
            eb = data.get("emotion_before") or {}
            if eb.get("top_emotion", "").strip().lower() == gt:
                before_preserved += 1

            # ── Emotion after (synthesized transcription audio) ───────────
            ea = data.get("emotion_after") or {}
            if ea:
                after_available += 1
                if ea.get("top_emotion", "").strip().lower() == gt:
                    after_preserved += 1

        before_rate = before_preserved / total if total > 0 else 0.0
        after_rate  = after_preserved / after_available if after_available > 0 else None

        results[pipeline_dir_name] = {
            "label":            label,
            "total":            total,
            "missing_gt":       missing_gt,
            "csv_fallback":     csv_fallback,
            "before_preserved": before_preserved,
            "before_rate":      before_rate,
            "after_available":  after_available,
            "after_preserved":  after_preserved,
            "after_rate":       after_rate,
        }

    # ── Check whether emotion_after is populated ──────────────────────────────
    has_after = any(r["after_available"] > 0 for r in results.values())

    # ── Print table ───────────────────────────────────────────────────────────
    if has_after:
        print(f"{'Pipeline':<12} {'Total':>7}  {'Before rate':>12}  {'After rate':>11}  {'After N':>8}")
        print("-" * 60)
        for pd_name in PIPELINES:
            r = results[pd_name]
            after_str = f"{r['after_rate']:>10.2%}" if r["after_rate"] is not None else "        N/A"
            print(f"{r['label']:<12} {r['total']:>7,}  {r['before_rate']:>11.2%}  {after_str}  {r['after_available']:>8,}")
        print("\n  'Before rate' = emotion_before vs gt_emotion (source audio, same for all pipelines)")
        print("  'After rate'  = emotion_after  vs gt_emotion (synthesized transcription audio)")
    else:
        print(f"{'Pipeline':<12} {'Preserved':>10} {'Total':>8} {'Rate (before)':>14}")
        print("-" * 50)
        for pd_name in PIPELINES:
            r = results[pd_name]
            print(f"{r['label']:<12} {r['before_preserved']:>10,} {r['total']:>8,} {r['before_rate']:>13.2%}")
        print("\n  Run add_emotion_after.py to populate emotion_after and get per-pipeline rates.")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out_path = BASE_DIR / "emotion_preservation_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to: {out_path}")


if __name__ == "__main__":
    main()
