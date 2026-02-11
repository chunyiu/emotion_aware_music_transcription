"""
Analysis script to identify top-performing files based on emotion classification.
Ranks files by confidence and selects best candidates for manual listening test.
"""
import json
import pandas as pd
from pathlib import Path
import numpy as np

# --- CONFIG ---
PREDICTIONS_PATH = Path(__file__).resolve().parent.parent.parent / "results" / "predictions" / "predictions.json"
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "emotion_model" / "analysis"


def load_predictions():
    """Load prediction results."""
    with open(PREDICTIONS_PATH, 'r') as f:
        return json.load(f)


def rank_files_by_confidence(predictions, top_n=50):
    """Rank files by prediction confidence."""
    df = pd.DataFrame([{
        "filename": p["filename"],
        "file_path": p["file"],
        "predicted_emotion": p["predicted_emotion"],
        "confidence": p["confidence"],
        "top2_emotion": p["top3_predictions"][1][0] if len(p["top3_predictions"]) > 1 else "",
        "top2_confidence": p["top3_predictions"][1][1] if len(p["top3_predictions"]) > 1 else 0.0,
        "confidence_gap": p["confidence"] - (p["top3_predictions"][1][1] if len(p["top3_predictions"]) > 1 else 0.0)
    } for p in predictions])
    
    # Sort by confidence (descending)
    df_sorted = df.sort_values(by="confidence", ascending=False)
    
    return df_sorted.head(top_n)


def select_best_per_emotion(predictions, n_per_emotion=3):
    """Select top N files per emotion class."""
    df = pd.DataFrame([{
        "filename": p["filename"],
        "file_path": p["file"],
        "predicted_emotion": p["predicted_emotion"],
        "confidence": p["confidence"],
    } for p in predictions])
    
    best_per_emotion = df.groupby("predicted_emotion").apply(
        lambda x: x.nlargest(n_per_emotion, "confidence")
    ).reset_index(drop=True)
    
    return best_per_emotion


def analyze_predictions(predictions):
    """Compute summary statistics."""
    df = pd.DataFrame([{
        "predicted_emotion": p["predicted_emotion"],
        "confidence": p["confidence"],
    } for p in predictions])
    
    stats = {
        "total_files": len(df),
        "emotion_distribution": df["predicted_emotion"].value_counts().to_dict(),
        "avg_confidence_per_emotion": df.groupby("predicted_emotion")["confidence"].mean().to_dict(),
        "overall_avg_confidence": float(df["confidence"].mean()),
        "overall_std_confidence": float(df["confidence"].std())
    }
    
    return stats


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load predictions
    print("Loading predictions...")
    predictions = load_predictions()
    print(f"Loaded {len(predictions)} predictions")
    
    # 1. Rank top 50 by confidence
    print("\n=== Top 50 files by confidence ===")
    top50 = rank_files_by_confidence(predictions, top_n=50)
    print(top50.head(10))
    
    top50_path = OUTPUT_DIR / "top50_by_confidence.csv"
    top50.to_csv(top50_path, index=False)
    print(f"Top 50 saved to: {top50_path}")
    
    # 2. Select top 3 per emotion
    print("\n=== Top 3 per emotion ===")
    best_per_emotion = select_best_per_emotion(predictions, n_per_emotion=3)
    print(best_per_emotion)
    
    best_per_emotion_path = OUTPUT_DIR / "top3_per_emotion.csv"
    best_per_emotion.to_csv(best_per_emotion_path, index=False)
    print(f"Top 3 per emotion saved to: {best_per_emotion_path}")
    
    # 3. Summary statistics
    print("\n=== Prediction Statistics ===")
    stats = analyze_predictions(predictions)
    print(json.dumps(stats, indent=2))
    
    stats_path = OUTPUT_DIR / "prediction_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Stats saved to: {stats_path}")
    
    # 4. Select 3 best overall (for manual listening)
    print("\n=== Recommended files for manual listening (top 3 overall) ===")
    top3_overall = top50.head(3)
    print(top3_overall[["filename", "predicted_emotion", "confidence", "confidence_gap"]])
    
    listening_test_path = OUTPUT_DIR / "listening_test_candidates.csv"
    top3_overall.to_csv(listening_test_path, index=False)
    print(f"\nListening test candidates saved to: {listening_test_path}")
    print("\nNext steps:")
    print("1. Listen to these 3 files")
    print("2. Verify predicted emotion matches perceived emotion")
    print("3. Document findings in manual evaluation report")
    
    # Generate listening test report template
    report_md = f"""# Emotion Classification - Manual Listening Test

## Top 3 Files Selected for Evaluation

"""
    for idx, row in top3_overall.iterrows():
        report_md += f"""
### File {idx+1}: {row['filename']}
- **Predicted Emotion**: {row['predicted_emotion']}
- **Confidence**: {row['confidence']:.4f}
- **Confidence Gap** (vs 2nd choice): {row['confidence_gap']:.4f}
- **File Path**: `{row['file_path']}`

**Manual Evaluation**:
- [ ] Listened to file
- [ ] Perceived emotion matches prediction: YES / NO
- [ ] Notes: _____________________

---
"""
    
    report_path = OUTPUT_DIR / "listening_test_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_md)
    print(f"Listening test report template saved to: {report_path}")


if __name__ == "__main__":
    main()
