"""
Emotion prediction script for unseen audio files (pipeline outputs).
Loads trained model and predicts emotion with confidence scores.
"""
import os
import numpy as np
import librosa
import pickle
import json
from pathlib import Path
import pandas as pd

# --- CONFIG ---
MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "emotion_model"
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "predictions"
SR = 22050
N_MFCC = 40
N_MELS = 128
DURATION_SEC = 3.0


def extract_features(audio_path, sr=SR, duration=DURATION_SEC):
    """Extract features (same as training)."""
    try:
        y, sr_actual = librosa.load(audio_path, sr=sr, duration=duration, mono=True)
        
        target_length = int(sr * duration)
        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)), mode='constant')
        
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
        mel_mean = np.mean(mel, axis=1)
        mel_std = np.std(mel, axis=1)
        
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
        
        features = np.concatenate([
            mfcc_mean, mfcc_std,
            mel_mean, mel_std,
            [spectral_centroid, spectral_rolloff, zero_crossing_rate]
        ])
        
        return features
    except Exception as e:
        print(f"Error extracting features from {audio_path}: {e}")
        return None


def load_model_artifacts():
    """Load trained model, scaler, and label encoder."""
    model_path = MODEL_DIR / "emotion_classifier.pkl"
    scaler_path = MODEL_DIR / "scaler.pkl"
    le_path = MODEL_DIR / "label_encoder.pkl"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Run train_emotion_classifier.py first.")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(le_path, 'rb') as f:
        le = pickle.load(f)
    
    return model, scaler, le


def predict_emotion_batch(audio_files, model, scaler, le):
    """
    Predict emotion for a batch of audio files.
    Returns list of dicts with predictions.
    """
    results = []
    
    for audio_file in audio_files:
        audio_path = Path(audio_file)
        if not audio_path.exists():
            print(f"File not found: {audio_path}")
            continue
        
        features = extract_features(audio_path)
        if features is None:
            continue
        
        # Scale and predict
        features_scaled = scaler.transform(features.reshape(1, -1))
        pred_encoded = model.predict(features_scaled)[0]
        pred_proba = model.predict_proba(features_scaled)[0]
        
        emotion = le.inverse_transform([pred_encoded])[0]
        confidence = float(pred_proba[pred_encoded])
        
        # Get top-3 predictions
        top3_idx = np.argsort(pred_proba)[::-1][:3]
        top3_emotions = [(le.classes_[i], float(pred_proba[i])) for i in top3_idx]
        
        results.append({
            "file": str(audio_path),
            "filename": audio_path.name,
            "predicted_emotion": emotion,
            "confidence": confidence,
            "top3_predictions": top3_emotions,
            "all_probabilities": {le.classes_[i]: float(pred_proba[i]) for i in range(len(le.classes_))}
        })
    
    return results


def main(input_dir=None, input_files=None):
    """
    Predict emotions for unseen audio files.
    Provide either input_dir (scan for .wav/.mp3) or input_files (list of paths).
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("Loading trained model...")
    model, scaler, le = load_model_artifacts()
    print(f"Model loaded. Classes: {le.classes_}")
    
    # Collect audio files
    if input_files is None:
        if input_dir is None:
            raise ValueError("Provide either input_dir or input_files")
        input_dir = Path(input_dir)
        
        # Scan for audio files (.wav, .mp3) - exclude .mid files
        audio_files = list(input_dir.rglob("*.wav")) + list(input_dir.rglob("*.mp3"))
        
        # Filter out any system/temp files
        audio_files = [f for f in audio_files if not f.name.startswith('.') and not f.name.startswith('~')]
        
        print(f"Found {len(audio_files)} audio files in {input_dir}")
        
        if len(audio_files) > 0:
            print("Files to process:")
            for f in audio_files[:10]:  # Show first 10
                print(f"  - {f.relative_to(input_dir)}")
            if len(audio_files) > 10:
                print(f"  ... and {len(audio_files) - 10} more")
    else:
        audio_files = input_files
        print(f"Processing {len(audio_files)} provided files")
    
    if len(audio_files) == 0:
        print("No audio files to process.")
        return
    
    # Predict
    print("\nPredicting emotions...")
    results = predict_emotion_batch(audio_files, model, scaler, le)
    
    if len(results) == 0:
        print("No valid predictions generated.")
        return
    
    # Save results
    results_json_path = OUTPUT_DIR / "predictions.json"
    with open(results_json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nPredictions saved to: {results_json_path}")
    
    # Save as CSV
    df = pd.DataFrame([{
        "filename": r["filename"],
        "predicted_emotion": r["predicted_emotion"],
        "confidence": r["confidence"],
        "top2_emotion": r["top3_predictions"][1][0] if len(r["top3_predictions"]) > 1 else "",
        "top2_confidence": r["top3_predictions"][1][1] if len(r["top3_predictions"]) > 1 else 0.0
    } for r in results])
    
    csv_path = OUTPUT_DIR / "predictions.csv"
    df.to_csv(csv_path, index=False)
    print(f"Predictions CSV saved to: {csv_path}")
    
    # Summary
    print("\n=== Prediction Summary ===")
    print(df.groupby("predicted_emotion").size())
    
    return results


if __name__ == "__main__":
    # Predict on the 50 sample files in vocal-to-score-demo/Input folder
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    SAMPLE_FOLDER = BASE_DIR / "vocal-to-score-demo"
    
    # Process the 50 sample files in Input folder
    INPUT_FOLDER = SAMPLE_FOLDER / "Input"
    print(f"Processing 50 sample files from: {INPUT_FOLDER}")
    main(input_dir=INPUT_FOLDER)
    
    # Alternative options (uncomment if needed):
    # Option A: Process all audio in vocal-to-score-demo folder (including outputs):
    # main(input_dir=SAMPLE_FOLDER)
    
    # Option B: Process only outputs:
    # main(input_dir=SAMPLE_FOLDER / "outputs")
    
    # Option C: Process specific files:
    # specific_files = [INPUT_FOLDER / "file1.wav", INPUT_FOLDER / "file2.mp3"]
    # main(input_files=specific_files)
