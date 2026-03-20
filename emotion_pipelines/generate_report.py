"""
Generate a 32-combination comparison report from all pipeline outputs.

Usage:
    python generate_report.py
"""

import json
import csv
from pathlib import Path
from config import OUTPUT_DIR

STAGE1_IDS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
STAGE2_IDS = ['1', '2', '3', '4']

PITCH_METHODS = {
    'A': 'Simple pYIN', 'B': 'Simple pYIN',
    'C': 'pYIN+HMM', 'D': 'pYIN+HMM',
    'E': 'CREPE', 'F': 'CREPE',
    'G': 'TorchCrepe', 'H': 'TorchCrepe',
}

GT_FORMATS = {
    'A': 'JSON', 'B': 'MusicXML',
    'C': 'JSON', 'D': 'MusicXML',
    'E': 'JSON', 'F': 'MusicXML',
    'G': 'JSON', 'H': 'MusicXML',
}

HARMONY_METHODS = {
    '1': 'Music21 Transpose',
    '2': 'Mingus Chords',
    '3': 'Diatonic Roman',
    '4': 'Circle-of-Fifths',
}


def generate_report():
    output_dir = Path(OUTPUT_DIR)
    summaries_dir = output_dir / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for s1 in STAGE1_IDS:
        # Load Stage 1 summary
        s1_summary_path = output_dir / "stage1_transcription" / f"pipeline_{s1}" / f"pipeline_{s1.lower()}_summary.json"
        s1_data = {}
        if s1_summary_path.exists():
            with open(s1_summary_path) as f:
                s1_data = json.load(f)

        s1_results = s1_data.get('results', [])
        avg_f1 = 0.0
        if s1_results:
            f1s = [r.get('gt_metrics', {}).get('f1', 0) for r in s1_results if r.get('gt_metrics')]
            avg_f1 = sum(f1s) / len(f1s) if f1s else 0.0

        for s2 in STAGE2_IDS:
            combo = f"{s1}{s2}"
            s2_summary_path = output_dir / "stage2_harmony" / combo / f"{combo}_summary.json"

            s2_data = {}
            if s2_summary_path.exists():
                with open(s2_summary_path) as f:
                    s2_data = json.load(f)

            s2_results = s2_data.get('results', [])
            total = s2_data.get('total', 0)
            successful = s2_data.get('successful', 0)

            # Emotion preservation rate
            preserved_count = sum(1 for r in s2_results if r.get('emotion_preserved'))
            comparable = sum(1 for r in s2_results if r.get('emotion_preserved') is not None)
            preservation_rate = (preserved_count / comparable * 100) if comparable > 0 else 0.0

            # Emotion distributions
            before_emotions = {}
            after_emotions = {}
            for r in s2_results:
                eb = r.get('emotion_before', {})
                ea = r.get('emotion_after', {})
                if eb and eb.get('top_emotion'):
                    e = eb['top_emotion']
                    before_emotions[e] = before_emotions.get(e, 0) + 1
                if ea and ea.get('top_emotion'):
                    e = ea['top_emotion']
                    after_emotions[e] = after_emotions.get(e, 0) + 1

            rows.append({
                'Combo': combo,
                'Pitch Method': PITCH_METHODS.get(s1, '?'),
                'GT Format': GT_FORMATS.get(s1, '?'),
                'Harmony Method': HARMONY_METHODS.get(s2, '?'),
                'Stage1 Avg F1': f'{avg_f1:.3f}',
                'Total Files': total,
                'Successful': successful,
                'Emotion Preservation %': f'{preservation_rate:.1f}',
                'Top Before Emotion': max(before_emotions, key=before_emotions.get) if before_emotions else 'N/A',
                'Top After Emotion': max(after_emotions, key=after_emotions.get) if after_emotions else 'N/A',
            })

    # Write CSV
    csv_path = summaries_dir / "full_matrix_32.csv"
    if rows:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"Report saved to {csv_path}")
    else:
        print("No data found to generate report.")

    # Write JSON summary
    json_path = summaries_dir / "full_matrix_32.json"
    with open(json_path, 'w') as f:
        json.dump(rows, f, indent=2)
    print(f"JSON report saved to {json_path}")

    # Print summary table
    print(f"\n{'Combo':<8} {'Pitch':<14} {'GT':<10} {'Harmony':<18} {'F1':<8} {'Preserve%':<10}")
    print("-" * 70)
    for row in rows:
        print(f"{row['Combo']:<8} {row['Pitch Method']:<14} {row['GT Format']:<10} "
              f"{row['Harmony Method']:<18} {row['Stage1 Avg F1']:<8} {row['Emotion Preservation %']:<10}")

    return rows


if __name__ == "__main__":
    generate_report()
