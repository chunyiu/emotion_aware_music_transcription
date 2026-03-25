"""
Orchestrator for all 32 pipeline combinations (8 Stage 1 x 4 Stage 2).

Usage:
    python run_all.py                          # Run everything (auto device: CUDA > MPS > CPU)
    python run_all.py --stage 1                # Stage 1 only (A-H)
    python run_all.py --stage 2                # Stage 2 only (1-4 x A-H)
    python run_all.py --stage1-pipelines A B   # Only run pipelines A and B
    python run_all.py --stage2-pipelines 1 3   # Only harmony methods 1 and 3
    python run_all.py --max-files 5            # Limit to 5 files per pipeline
    python run_all.py --device cuda            # Force CUDA (overrides auto)
    python run_all.py --device mps             # Force MPS
    python run_all.py --device cpu             # Force CPU
"""

import argparse
import os
import time
from pathlib import Path

import torch
from tqdm import tqdm

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


def get_best_device(requested: str = "auto") -> torch.device:
    """
    Select the best available device with priority: cuda > mps > cpu
    Supports forcing a specific device with fallback warning if unavailable.
    """
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    # Forced device
    dev = torch.device(requested)

    if requested == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Falling back to best available device.")
        return get_best_device("auto")
    if requested == "mps" and not torch.backends.mps.is_available():
        print("Warning: MPS requested but not available. Falling back to best available device.")
        return get_best_device("auto")

    return dev


def run_stage1(
    pipeline_ids=None,
    max_files=None,
    input_dir=None,
    model_dir=None,
    device: torch.device = None,
):
    """Run Stage 1 pipelines (A-H) with clean tqdm progress bar."""
    import importlib

    pipeline_ids = pipeline_ids or STAGE1_IDS

    pbar = tqdm(pipeline_ids, desc="Stage 1 Pipelines", unit="pipeline", leave=True)
    for pid in pbar:
        pbar.set_description(f"Stage 1 -> Pipeline {pid}")

        module = importlib.import_module(STAGE1_MODULES[pid])
        start_time = time.time()

        kwargs = {}
        if max_files:
            kwargs['max_files'] = max_files
        if input_dir:
            kwargs['input_dir'] = input_dir
        if model_dir:
            kwargs['model_dir'] = model_dir
        if device is not None:
            kwargs['device'] = device

        try:
            module.run_pipeline(**kwargs)
        except TypeError as e:
            if "device" in str(e):
                print(f"  Warning: Pipeline {pid} does not accept 'device' argument yet.")
            else:
                raise

        elapsed = time.time() - start_time
        pbar.set_postfix({"time": f"{elapsed:.1f}s"})


def run_stage2(
    stage1_ids=None,
    stage2_ids=None,
    model_dir=None,
    device: torch.device = None,
):
    """Run Stage 2 pipelines (1-4) against Stage 1 outputs with clean tqdm progress bar."""
    import importlib

    stage1_ids = stage1_ids or STAGE1_IDS
    stage2_ids = stage2_ids or STAGE2_IDS

    combinations = [(s1, s2) for s1 in stage1_ids for s2 in stage2_ids]

    pbar = tqdm(combinations, desc="Stage 2 Combinations", unit="combo", leave=True)
    for s1, s2 in pbar:
        combo = f"{s1}{s2}"
        pbar.set_description(f"Stage 2 -> {combo}")

        module = importlib.import_module(STAGE2_MODULES[s2])
        start_time = time.time()

        kwargs = {'stage1_pipeline_id': s1}
        if model_dir:
            kwargs['model_dir'] = model_dir
        if device is not None:
            kwargs['device'] = device

        try:
            module.run_harmony(**kwargs)
        except TypeError as e:
            if "device" in str(e):
                print(f"  Warning: Harmony {combo} does not accept 'device' argument yet.")
            else:
                raise

        elapsed = time.time() - start_time
        pbar.set_postfix({"time": f"{elapsed:.1f}s"})


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
    parser.add_argument('--device', type=str, choices=['auto', 'cuda', 'mps', 'cpu'], default='auto',
                        help='Device selection (default: auto = CUDA > MPS > CPU)')

    args = parser.parse_args()

    # Device selection
    selected_device = get_best_device(args.device)

    # Enable MPS fallback only when using MPS
    if selected_device.type == "mps":
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    s1_ids = args.stage1_pipelines or STAGE1_IDS
    s2_ids = args.stage2_pipelines or STAGE2_IDS

    print("=" * 70)
    print("EMOTION-AWARE MUSIC TRANSCRIPTION PIPELINE ORCHESTRATOR")
    print("=" * 70)
    print(f"SELECTED DEVICE: {selected_device}")
    print(f"  (priority: CUDA -> MPS -> CPU)")

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
        run_stage1(
            s1_ids,
            args.max_files,
            args.input_dir,
            args.model_dir,
            device=selected_device
        )

    # Run Stage 2
    if args.stage in ('2', 'all'):
        run_stage2(
            s1_ids,
            s2_ids,
            args.model_dir,
            device=selected_device
        )

    print(f"\n{'='*70}")
    print("ALL PIPELINES COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()