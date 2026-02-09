import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

DATASET_PATH = "Data/genres_original"
OUTPUT_FILE = "features.csv"

def extract_features(file_path):
    signal, sr = librosa.load(file_path, duration=30, mono=True)
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40)

    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_var = np.var(mfcc, axis=1)

    return np.concatenate((mfcc_mean, mfcc_var))

data = []

print("🔄 Extracting enhanced MFCC features...")

for genre in tqdm(os.listdir(DATASET_PATH)):
    genre_path = os.path.join(DATASET_PATH, genre)
    if not os.path.isdir(genre_path):
        continue

    for file in os.listdir(genre_path):
        if file.endswith(".wav"):
            try:
                features = extract_features(os.path.join(genre_path, file))
                data.append([*features, genre])
            except:
                pass

columns = (
    [f"mfcc_mean_{i}" for i in range(40)] +
    [f"mfcc_var_{i}" for i in range(40)] +
    ["label"]
)

pd.DataFrame(data, columns=columns).to_csv(OUTPUT_FILE, index=False)
print("✅ features.csv updated")
