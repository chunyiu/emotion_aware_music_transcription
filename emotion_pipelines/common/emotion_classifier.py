"""
Shared RAVDESS-trained emotion classifier.
Standardized to 339 features: 40 MFCCs (mean+std), 128 mel-specs (mean+std),
3 spectral (centroid, rolloff, zero_crossing_rate) — must match train_emotion_classifier.py.
"""

import numpy as np
import librosa
import pickle
from pathlib import Path

class EmotionClassifier:
    """RAVDESS-trained emotion classifier with standardized output schema."""

    def __init__(self, model_dir, device='cpu'):
        """
        Args:
            model_dir (str or Path): Directory containing model, scaler, label encoder.
            device (str): 'cpu' or 'cuda', reserved for future PyTorch support.
        """
        self.device = device
        model_dir = Path(model_dir)

        with open(model_dir / 'emotion_classifier.pkl', 'rb') as f:
            self.model = pickle.load(f)
        with open(model_dir / 'scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
        with open(model_dir / 'label_encoder.pkl', 'rb') as f:
            self.le = pickle.load(f)

        # If PyTorch model, move to device (future-proofing)
        try:
            import torch
            if isinstance(self.model, torch.nn.Module):
                self.model.to(self.device)
        except ImportError:
            pass

    def extract_features(self, audio_path, sr=22050, duration=3.0):
        """Extract 339 audio features for emotion classification."""
        try:
            y, _ = librosa.load(audio_path, sr=sr, duration=duration, mono=True)
            target_length = int(sr * duration)
            if len(y) < target_length:
                y = np.pad(y, (0, target_length - len(y)), mode='constant')

            # MFCCs (40 mean + 40 std = 80)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)

            # Mel spectrogram (128 mean + 128 std = 256)
            mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            mel_mean = np.mean(mel, axis=1)
            mel_std = np.std(mel, axis=1)

            # Spectral features (3)
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))

            features = np.concatenate([
                mfcc_mean, mfcc_std,
                mel_mean, mel_std,
                [spectral_centroid, spectral_rolloff, zero_crossing_rate]
            ])  # Total: 339 features

            return features

        except Exception as e:
            print(f"Error extracting features: {e}")
            return None

    def predict(self, audio_path):
        """Predict emotion, returning standardized output with top 2 emotions."""
        features = self.extract_features(audio_path)
        if features is None:
            return None

        features_scaled = self.scaler.transform(features.reshape(1, -1))
        proba = self.model.predict_proba(features_scaled)[0]

        top2_idx = np.argsort(proba)[-2:][::-1]
        top2_emotions = [
            {
                'emotion': self.le.classes_[idx],
                'confidence': float(proba[idx])
            }
            for idx in top2_idx
        ]

        return {
            'top_emotion': top2_emotions[0]['emotion'],
            'top_confidence': top2_emotions[0]['confidence'],
            'second_emotion': top2_emotions[1]['emotion'],
            'second_confidence': top2_emotions[1]['confidence'],
            'top2': top2_emotions
        }