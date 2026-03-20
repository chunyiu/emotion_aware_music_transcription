"""
Emotion classification training script for RAVDESS dataset.
Trains on 80% of data, validates on 20%.
"""
import os
import numpy as np
import librosa
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import json

# --- CONFIG ---
RAVDESS_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "archive (1)"
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "emotion_model"
SR = 22050
N_MFCC = 40
N_MELS = 128
DURATION_SEC = 3.0  # Fixed audio duration for consistency

# RAVDESS emotion mapping (filename format: 03-01-XX-01-01-01-XX.wav)
# Position 3 (index 2) encodes emotion
EMOTION_MAP = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}


def parse_ravdess_emotion(filename):
    """Extract emotion label from RAVDESS filename."""
    parts = filename.split('-')
    if len(parts) >= 3:
        emotion_code = parts[2]
        return EMOTION_MAP.get(emotion_code, None)
    return None


def extract_features(audio_path, sr=SR, duration=DURATION_SEC):
    """
    Extract MFCC + mel-spectrogram statistics from audio.
    Returns feature vector (fixed length).
    """
    try:
        # Load audio with fixed duration
        y, sr_actual = librosa.load(audio_path, sr=sr, duration=duration, mono=True)
        
        # Pad if too short
        target_length = int(sr * duration)
        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)), mode='constant')
        
        # MFCC features (mean, std over time)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        
        # Mel-spectrogram features
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
        mel_mean = np.mean(mel, axis=1)
        mel_std = np.std(mel, axis=1)
        
        # Spectral features
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
        
        # Concatenate all features
        features = np.concatenate([
            mfcc_mean, mfcc_std,
            mel_mean, mel_std,
            [spectral_centroid, spectral_rolloff, zero_crossing_rate]
        ])
        
        return features
    except Exception as e:
        print(f"Error extracting features from {audio_path}: {e}")
        return None


def load_ravdess_dataset(data_path):
    """
    Load RAVDESS dataset and extract features.
    Returns X (features), y (emotion labels), filenames.
    """
    X = []
    y = []
    filenames = []
    
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"RAVDESS path not found: {data_path}")
    
    # RAVDESS structure: Actor_XX folders containing .wav files
    wav_files = list(data_path.rglob("*.wav"))
    
    if len(wav_files) == 0:
        raise ValueError(f"No .wav files found in {data_path}")
    
    print(f"Found {len(wav_files)} audio files in RAVDESS dataset")
    
    for i, wav_file in enumerate(wav_files):
        if i % 50 == 0:
            print(f"Processing {i}/{len(wav_files)}...")
        
        emotion = parse_ravdess_emotion(wav_file.name)
        if emotion is None:
            continue
        
        features = extract_features(wav_file)
        if features is not None:
            X.append(features)
            y.append(emotion)
            filenames.append(str(wav_file))
    
    print(f"Successfully extracted features from {len(X)} files")
    return np.array(X), np.array(y), filenames


def train_classifier(X_train, y_train, model_type='rf'):
    """
    Train emotion classifier.
    model_type: 'rf' (RandomForest) or 'svm'
    """
    if model_type == 'svm':
        model = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
    else:  # random forest
        model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
    
    print(f"Training {model_type.upper()} classifier...")
    model.fit(X_train, y_train)
    print("Training complete.")
    return model


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Load dataset
    print("Loading RAVDESS dataset...")
    X, y, filenames = load_ravdess_dataset(RAVDESS_PATH)
    
    # 2. Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    print(f"\nEmotion distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for emotion, count in zip(unique, counts):
        print(f"  {emotion}: {count}")
    
    # 3. Split 80/20
    X_train, X_test, y_train, y_test, files_train, files_test = train_test_split(
        X, y_encoded, filenames, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # 4. Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 5. Train model (Random Forest as baseline)
    model = train_classifier(X_train_scaled, y_train, model_type='rf')
    
    # 6. Evaluate on test set
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n=== Test Set Performance ===")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(conf_matrix)
    
    # 7. Save model, scaler, label encoder
    model_path = OUTPUT_DIR / "emotion_classifier.pkl"
    scaler_path = OUTPUT_DIR / "scaler.pkl"
    le_path = OUTPUT_DIR / "label_encoder.pkl"
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    with open(le_path, 'wb') as f:
        pickle.dump(le, f)
    
    print(f"\nModel saved to: {model_path}")
    print(f"Scaler saved to: {scaler_path}")
    print(f"Label encoder saved to: {le_path}")
    
    # 8. Save test set results
    test_results = {
        "accuracy": float(accuracy),
        "classification_report": classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True),
        "confusion_matrix": conf_matrix.tolist(),
        "test_files": files_test,
        "test_predictions": [le.classes_[p] for p in y_pred],
        "test_probabilities": y_pred_proba.tolist()
    }
    
    results_path = OUTPUT_DIR / "test_results.json"
    with open(results_path, 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print(f"Test results saved to: {results_path}")


if __name__ == "__main__":
    main()
