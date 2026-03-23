"""Convenience script to run a selected Stage 1 transcription pipeline
on the GTSinger dataset and report aggregate note‑level metrics.

Usage examples (from the emotion_pipelines directory):

    # Run pipeline A on all files, auto‑select device
    python run_stage1.py --pipeline A

    # Run pipeline C on at most 100 files, force CPU
    python run_stage1.py --pipeline C --max-files 100 --device cpu

This simply forwards to the existing stage1 `run_pipeline` functions,
which already perform transcription and compare against GTSinger ground
truth using `compare_notes`.
"""

import argparse
from typing import Dict, Any, List

from sklearn.metrics import f1_score

# Import the individual Stage 1 pipeline modules
from stage1 import pipeline_a, pipeline_b, pipeline_c, pipeline_d
from stage1 import pipeline_e, pipeline_f, pipeline_g, pipeline_h


PIPELINE_MODULES = {
    "A": pipeline_a,
    "B": pipeline_b,
    "C": pipeline_c,
    "D": pipeline_d,
    "E": pipeline_e,
    "F": pipeline_f,
    "G": pipeline_g,
    "H": pipeline_h,
}


def compute_aggregate_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute simple averages of note and melody metrics over all files.

    Each result dict is expected to contain a `gt_metrics` entry like:
        {"precision": ..., "recall": ..., "f1": ..., "rpa": ..., ...}
    as produced by common.ground_truth.compare_notes and
    compute_melody_frame_metrics.
    """

    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []
    rpas: List[float] = []
    rcas: List[float] = []
    oas: List[float] = []
    emo_true: List[str] = []
    emo_pred: List[str] = []

    for r in results:
        m = r.get("gt_metrics") or {}
        if not m:
            continue
        if "precision" in m:
            precisions.append(m["precision"])
        if "recall" in m:
            recalls.append(m["recall"])
        if "f1" in m:
            f1s.append(m["f1"])
        if "rpa" in m:
            rpas.append(m["rpa"])
        if "rca" in m:
            rcas.append(m["rca"])
        if "oa" in m:
            oas.append(m["oa"])

        # Emotion classification vs ground-truth emotion (JSON GT only)
        gt_emotion = r.get("gt_emotion")
        emo_before = r.get("emotion_before") or {}
        pred_label = None
        if isinstance(emo_before, dict):
            pred_label = emo_before.get("top_emotion")
        # Only count if both labels are present
        if gt_emotion and pred_label:
            emo_true.append(str(gt_emotion))
            emo_pred.append(str(pred_label))

    def avg(values: List[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    metrics: Dict[str, float] = {
        "precision": avg(precisions),
        "recall": avg(recalls),
        "f1": avg(f1s),
        "rpa": avg(rpas),
        "rca": avg(rcas),
        "oa": avg(oas),
    }

    # Emotion metrics (Stage 1 SER vs ground-truth emotion)
    if emo_true:
        # Simple accuracy and macro F1 over emotion labels
        correct = sum(1 for t, p in zip(emo_true, emo_pred) if t == p)
        metrics["emotion_accuracy"] = correct / len(emo_true)
        try:
            metrics["emotion_f1_macro"] = f1_score(emo_true, emo_pred, average="macro")
        except Exception:
            metrics["emotion_f1_macro"] = 0.0

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a Stage 1 transcription pipeline (A–H) on GTSinger and "
            "summarise note‑level metrics against ground truth."
        )
    )

    parser.add_argument(
        "--pipeline",
        "-p",
        type=str,
        required=True,
        choices=list(PIPELINE_MODULES.keys()),
        help="Which Stage 1 pipeline to run (A–H).",
    )

    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional limit on number of files to process.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help=(
            "Optional device override, e.g. 'cuda', 'mps', or 'cpu'. "
            "If omitted, each pipeline's own auto‑detection is used."
        ),
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Override dataset root (defaults to config.DATASET_DIR).",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output root (defaults to config.OUTPUT_DIR).",
    )

    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Override model root (defaults to config.MODEL_DIR).",
    )

    args = parser.parse_args()

    pipeline_id = args.pipeline.upper()
    module = PIPELINE_MODULES[pipeline_id]

    print(f"\n=== Running Stage 1 Pipeline {pipeline_id} ===")

    # Some pipeline modules accept `device` as a torch.device or str;
    # here we pass through the raw string and let each module handle it.
    run_kwargs = {
        "input_dir": args.input_dir,
        "output_dir": args.output_dir,
        "model_dir": args.model_dir,
        "max_files": args.max_files,
    }

    # Only include device if explicitly provided, to preserve auto‑detection
    # behaviour in the individual pipeline modules.
    if args.device is not None:
        run_kwargs["device"] = args.device

    results = module.run_pipeline(**run_kwargs)

    if not results:
        print("\nNo results produced (no files or all failed).")
        return

    metrics = compute_aggregate_metrics(results)

    print("\n=== Aggregate note‑level metrics (predicted vs GTSinger GT) ===")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall:    {metrics['recall']:.3f}")
    print(f"F1 score:  {metrics['f1']:.3f}")

    print("\n=== Aggregate melody metrics (frame‑level, mir_eval.melody) ===")
    print(f"RPA:       {metrics['rpa']:.3f}")
    print(f"RCA:       {metrics['rca']:.3f}")
    print(f"OA:        {metrics['oa']:.3f}")

    if "emotion_accuracy" in metrics:
        print("\n=== Stage 1 emotion metrics (SER vs GT emotion) ===")
        print(f"Accuracy:  {metrics['emotion_accuracy']:.3f}")
        print(f"F1 macro:  {metrics['emotion_f1_macro']:.3f}")


if __name__ == "__main__":
    main()
