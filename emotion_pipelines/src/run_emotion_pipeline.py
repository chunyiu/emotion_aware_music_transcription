"""
Master script to run complete emotion classification pipeline:
1. Train model on RAVDESS (80/20 split)
2. Predict on unseen pipeline outputs
3. Analyze and select top files for manual evaluation
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from emotion_classification import train_emotion_classifier
from emotion_classification import predict_emotion
from emotion_classification import analyze_best_files


def main():
    print("=" * 60)
    print("EMOTION CLASSIFICATION PIPELINE")
    print("=" * 60)
    
    # Step 1: Train model
    print("\n" + "=" * 60)
    print("STEP 1: Training emotion classifier on RAVDESS dataset")
    print("=" * 60)
    try:
        train_emotion_classifier.main()
        print("\n✓ Training completed successfully")
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        print("Please ensure RAVDESS dataset is available at the configured path.")
        return
    
    # Step 2: Predict on unseen data
    print("\n" + "=" * 60)
    print("STEP 2: Predicting emotions on 50 sample files")
    print("=" * 60)
    try:
        # Process the sample files in the data folder
        INPUT_FOLDER = Path(__file__).resolve().parent.parent / "data" / "raw" / "archive"
        print(f"Processing audio files from: {INPUT_FOLDER}")
        predict_emotion.main(input_dir=INPUT_FOLDER)
        print("\n✓ Prediction completed successfully")
    except Exception as e:
        print(f"\n✗ Prediction failed: {e}")
        return
    
    # Step 3: Analyze results
    print("\n" + "=" * 60)
    print("STEP 3: Analyzing results and selecting top files")
    print("=" * 60)
    try:
        analyze_best_files.main()
        print("\n✓ Analysis completed successfully")
    except Exception as e:
        print(f"\n✗ Analysis failed: {e}")
        return
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Review top 50 files: results/analysis/top50_by_confidence.csv")
    print("2. Listen to top 3 candidates: results/analysis/listening_test_candidates.csv")
    print("3. Fill out manual evaluation: results/analysis/listening_test_report.md")
    print("4. Compare with test set metrics: results/emotion_model/test_results.json")


if __name__ == "__main__":
    main()
