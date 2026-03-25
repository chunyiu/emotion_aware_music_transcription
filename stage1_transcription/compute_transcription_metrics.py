"""
Aggregate F1, RPA, RCA, OA per Stage 1 pipeline (A-H) from individual
*_notes.json files. Metrics were already computed against the correct
ground truth (JSON for A/C/E/G, MusicXML for B/D/F/H) during the pipeline run.
"""

import glob
import json
from pathlib import Path

BASE_DIR = Path(__file__).parent

GT_FORMAT = {
    'pipeline_A': 'JSON',    'pipeline_B': 'MusicXML',
    'pipeline_C': 'JSON',    'pipeline_D': 'MusicXML',
    'pipeline_E': 'JSON',    'pipeline_F': 'MusicXML',
    'pipeline_G': 'JSON',    'pipeline_H': 'MusicXML',
}

PITCH_METHOD = {
    'pipeline_A': 'Simple pYIN',    'pipeline_B': 'Simple pYIN',
    'pipeline_C': 'pYIN+HMM',       'pipeline_D': 'pYIN+HMM',
    'pipeline_E': 'CREPE',          'pipeline_F': 'CREPE',
    'pipeline_G': 'TorchCrepe',     'pipeline_H': 'TorchCrepe',
}

METRICS = ['f1', 'rpa', 'rca', 'oa']


def main():
    pipelines = sorted([d.name for d in BASE_DIR.iterdir()
                        if d.is_dir() and d.name.startswith('pipeline_')])
    results = {}

    for pipeline_name in pipelines:
        note_files = glob.glob(str(BASE_DIR / pipeline_name / '*_notes.json'))

        totals = {m: 0.0 for m in METRICS}
        counts = {m: 0   for m in METRICS}

        for fp in note_files:
            with open(fp, encoding='utf-8') as f:
                data = json.load(f)
            gm = data.get('ground_truth_metrics') or {}
            for m in METRICS:
                if m in gm:
                    totals[m] += gm[m]
                    counts[m] += 1

        averages = {m: (totals[m] / counts[m] if counts[m] > 0 else None)
                    for m in METRICS}

        results[pipeline_name] = {
            'pipeline':     pipeline_name.replace('pipeline_', 'Pipeline ').upper(),
            'pitch_method': PITCH_METHOD.get(pipeline_name, '?'),
            'gt_format':    GT_FORMAT.get(pipeline_name, '?'),
            'total_files':  len(note_files),
            **{m: round(averages[m], 4) if averages[m] is not None else None
               for m in METRICS},
            'file_counts':  counts,
        }

    # Print table
    print(f"{'Pipeline':<12} {'Pitch Method':<14} {'GT':<10} {'F1':>7} {'RPA':>7} {'RCA':>7} {'OA':>7}")
    print('-' * 66)
    for name in pipelines:
        r = results[name]
        def fmt(v): return f'{v:.4f}' if v is not None else '   N/A'
        print(f"{r['pipeline']:<12} {r['pitch_method']:<14} {r['gt_format']:<10} "
              f"{fmt(r['f1']):>7} {fmt(r['rpa']):>7} {fmt(r['rca']):>7} {fmt(r['oa']):>7}")

    # Save
    out_path = BASE_DIR / 'transcription_metrics_summary.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved to: {out_path}')


if __name__ == '__main__':
    main()
