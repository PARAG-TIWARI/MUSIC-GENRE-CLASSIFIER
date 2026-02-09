import librosa
import numpy as np
import joblib
import sys

model = joblib.load("model/genre_model.pkl")
scaler = joblib.load("model/scaler.pkl")

def extract_features(file_path):
    signal, sr = librosa.load(file_path, duration=30)
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
    return np.mean(mfcc, axis=1)

def predict_genre(file_path):
    features = extract_features(file_path)
    features = scaler.transform([features])
    return model.predict(features)[0]

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py <audio.wav>")
        sys.exit()

    print("🎵 Predicted Genre:", predict_genre(sys.argv[1]))
