"""
Orchestrator for all 32 pipeline combinations (8 Stage 1 x 4 Stage 2).

Usage:
    python run_all.py                          # Run everything
    python run_all.py --stage 1                # Stage 1 only (A-H)
    python run_all.py --stage 2                # Stage 2 only (1-4 x A-H)
    python run_all.py --stage1-pipelines A B   # Only run pipelines A and B
    python run_all.py --stage2-pipelines 1 3   # Only harmony methods 1 and 3
    python run_all.py --max-files 5            # Limit to 5 files per pipeline
"""

import argparse
import json
import time
from pathlib import Path

from config import OUTPUT_DIR

STAGE1_IDS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
STAGE2_IDS = ['1', '2', '3', '4']

# Map pipeline IDs to their modules
STAGE1_MODULES = {
    'A': 'stage1.pipeline_a',
    'B': 'stage1.pipeline_b',
    'C': 'stage1.pipeline_c',
    'D': 'stage1.pipeline_d',
    'E': 'stage1.pipeline_e',
    'F': 'stage1.pipeline_f',
    'G': 'stage1.pipeline_g',
    'H': 'stage1.pipeline_h',
}

STAGE2_MODULES = {
    '1': 'stage2.pipeline_1',
    '2': 'stage2.pipeline_2',
    '3': 'stage2.pipeline_3',
    '4': 'stage2.pipeline_4',
}


def run_stage1(pipeline_ids=None, max_files=None, input_dir=None, model_dir=None):
    """Run Stage 1 pipelines (A-H)."""
    import importlib

    pipeline_ids = pipeline_ids or STAGE1_IDS

    for pid in pipeline_ids:
        print(f"\n{'='*70}")
        print(f"RUNNING STAGE 1: Pipeline {pid}")
        print(f"{'='*70}")

        module = importlib.import_module(STAGE1_MODULES[pid])
        start_time = time.time()

        kwargs = {}
        if max_files:
            kwargs['max_files'] = max_files
        if input_dir:
            kwargs['input_dir'] = input_dir
        if model_dir:
            kwargs['model_dir'] = model_dir

        module.run_pipeline(**kwargs)

        elapsed = time.time() - start_time
        print(f"Pipeline {pid} completed in {elapsed:.1f}s")


def run_stage2(stage1_ids=None, stage2_ids=None, model_dir=None):
    """Run Stage 2 pipelines (1-4) against Stage 1 outputs."""
    import importlib

    stage1_ids = stage1_ids or STAGE1_IDS
    stage2_ids = stage2_ids or STAGE2_IDS

    total = len(stage1_ids) * len(stage2_ids)
    count = 0

    for s1 in stage1_ids:
        for s2 in stage2_ids:
            count += 1
            combo = f"{s1}{s2}"
            print(f"\n{'='*70}")
            print(f"RUNNING STAGE 2: Combination {combo} [{count}/{total}]")
            print(f"{'='*70}")

            module = importlib.import_module(STAGE2_MODULES[s2])
            start_time = time.time()

            kwargs = {'stage1_pipeline_id': s1}
            if model_dir:
                kwargs['model_dir'] = model_dir

            module.run_harmony(**kwargs)

            elapsed = time.time() - start_time
            print(f"Combination {combo} completed in {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Run emotion-aware music transcription pipelines")
    parser.add_argument('--stage', choices=['1', '2', 'all'], default='all',
                        help='Which stage(s) to run')
    parser.add_argument('--stage1-pipelines', nargs='+', choices=STAGE1_IDS,
                        help='Subset of Stage 1 pipelines to run')
    parser.add_argument('--stage2-pipelines', nargs='+', choices=STAGE2_IDS,
                        help='Subset of Stage 2 pipelines to run')
    parser.add_argument('--max-files', type=int, default=None,
                        help='Maximum files per Stage 1 pipeline')
    parser.add_argument('--input-dir', type=str, default=None,
                        help='Override input dataset directory')
    parser.add_argument('--model-dir', type=str, default=None,
                        help='Override emotion model directory')

    args = parser.parse_args()

    s1_ids = args.stage1_pipelines or STAGE1_IDS
    s2_ids = args.stage2_pipelines or STAGE2_IDS

    print("=" * 70)
    print("EMOTION-AWARE MUSIC TRANSCRIPTION PIPELINE ORCHESTRATOR")
    print("=" * 70)

    if args.stage in ('1', 'all'):
        print(f"\nStage 1 pipelines: {', '.join(s1_ids)}")
        if args.max_files:
            print(f"Max files per pipeline: {args.max_files}")

    if args.stage in ('2', 'all'):
        combos = [f"{s1}{s2}" for s1 in s1_ids for s2 in s2_ids]
        print(f"\nStage 2 combinations: {', '.join(combos)} ({len(combos)} total)")

    print()

    # Run Stage 1
    if args.stage in ('1', 'all'):
        run_stage1(s1_ids, args.max_files, args.input_dir, args.model_dir)

    # Run Stage 2
    if args.stage in ('2', 'all'):
        run_stage2(s1_ids, s2_ids, args.model_dir)

    print(f"\n{'='*70}")
    print("ALL PIPELINES COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
