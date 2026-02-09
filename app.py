import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/cnn_genre_model.h5")

model = load_model()

classes = [
    "blues","classical","country","disco","hiphop",
    "jazz","metal","pop","reggae","rock"
]

st.title("🎵 Music Genre Classification (CNN)")
st.write("Upload a spectrogram image")

file = st.file_uploader("Upload Image", type=["png","jpg","jpeg"])

if file:
    img = Image.open(file).convert("RGB")
    st.image(img)

    img = img.resize((128,128))
    img = np.array(img)/255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    genre = classes[np.argmax(pred)]
    conf = np.max(pred)*100

    st.success(f"Genre: {genre.upper()}")
    st.info(f"Confidence: {conf:.2f}%")
