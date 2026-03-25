"""
Patch existing stage1_transcription JSON files for pipelines B, D, F, H
to add the missing 'gt_emotion' field sourced from gtsinger_english_emotions.csv.

Pipelines B/D/F/H use MusicXML ground truth, which has no emotion field,
so gt_emotion was never stored during the original pipeline run.
"""

import glob
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent
CSV_PATH = BASE_DIR.parent / "emotion_pipelines" / "gtsinger_english_emotions.csv"
MUSICXML_PIPELINES = ["pipeline_B", "pipeline_D", "pipeline_F", "pipeline_H"]

sys.path.insert(0, str(BASE_DIR.parent / "emotion_pipelines"))
from common.csv_emotions import load_csv_emotions, source_audio_to_csv_key


def main():
    csv_emotions = load_csv_emotions(CSV_PATH)
    print(f"Loaded {len(csv_emotions):,} entries from GTSinger CSV\n")

    for pipeline_name in MUSICXML_PIPELINES:
        pipeline_dir = BASE_DIR / pipeline_name
        note_files = sorted(glob.glob(str(pipeline_dir / "*_notes.json")))

        patched = 0
        skipped_already_has = 0
        skipped_no_key = 0

        for fp in note_files:
            with open(fp, encoding="utf-8") as f:
                data = json.load(f)

            # Skip files that already have a non-empty gt_emotion
            if data.get("gt_emotion"):
                skipped_already_has += 1
                continue

            csv_key = source_audio_to_csv_key(data.get("source_audio", ""))
            gt_emotion = csv_emotions.get(csv_key, "")

            if not gt_emotion:
                skipped_no_key += 1
                continue

            data["gt_emotion"] = gt_emotion

            with open(fp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            patched += 1

        print(f"{pipeline_name}: patched={patched:,}  already_set={skipped_already_has:,}  no_csv_match={skipped_no_key:,}")

    print("\nDone. Re-run compute_emotion_preservation.py to see updated results.")


if __name__ == "__main__":
    main()
