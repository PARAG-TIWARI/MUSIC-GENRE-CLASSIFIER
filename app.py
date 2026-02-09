import streamlit as st
import librosa
import numpy as np
import joblib
import tempfile
import os

model = joblib.load("model/genre_model.pkl")
scaler = joblib.load("model/scaler.pkl")

def extract_features(file_path):
    signal, sr = librosa.load(file_path, duration=30)
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
    return np.mean(mfcc, axis=1)

st.set_page_config(page_title="Music Genre Classifier", page_icon="🎵")

st.title("🎵 Music Genre Classification")
st.write("Upload a WAV audio file to predict its genre")

uploaded_file = st.file_uploader("Upload Audio", type=["wav"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    st.audio(uploaded_file)

    if st.button("Predict Genre"):
        with st.spinner("Analyzing..."):
            features = extract_features(temp_path)
            features = scaler.transform([features])
            prediction = model.predict(features)[0]

        st.success(f"🎧 Predicted Genre: **{prediction.upper()}**")

    os.remove(temp_path)
