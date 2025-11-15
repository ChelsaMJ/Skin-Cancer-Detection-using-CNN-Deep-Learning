# app.py
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import io

st.set_page_config(page_title="Skin Cancer Detector", layout="centered")

@st.cache_resource
def load_model(path="model.h5"):
    model = tf.keras.models.load_model(path)
    return model

def preprocess_image(image: Image, target_size=(224,224)):
    # convert to RGB, resize, scale to [0,1], expand dims
    image = image.convert("RGB")
    image = image.resize(target_size)
    arr = np.array(image) / 255.0
    arr = np.expand_dims(arr, axis=0).astype(np.float32)
    return arr

# replace with your model's class names in correct order
CLASS_NAMES = [
    "Actinic keratoses and intraepithelial carcinomae (akiec)",
    "Basal cell carcinoma (bcc)",
    "Benign keratosis-like lesions (bkl)",
    "Dermatofibroma (df)",
    "Melanocytic nevi (nv)",
    "Pyogenic granulomas and hemorrhage (vasc)",
    "Melanoma (mel)"
]

st.title("Skin Cancer Detection")
st.write("Upload an image of a skin lesion and the model will predict whether it is benign or malignant.")

uploaded_file = st.file_uploader("Choose an image...", type=["png","jpg","jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", use_column_width=True)
    st.write("")

    st.write("Loading model...")
    try:
        model = load_model("model.keras")  # ensure model.h5 is in repo or path
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

    if st.button("Predict"):
        with st.spinner("Predicting..."):
            x = preprocess_image(image, target_size=(224,224))
            preds = model.predict(x)
            # handle shape: (1, n_classes)
            if preds.ndim == 2:
                probs = preds[0]
            else:
                probs = preds
            top_idx = int(np.argmax(probs))
            top_class = CLASS_NAMES[top_idx] if top_idx < len(CLASS_NAMES) else str(top_idx)
            confidence = float(probs[top_idx])
        st.success(f"Prediction: **{top_class}**  — Confidence: **{confidence:.2%}**")
        st.write("Probabilities:")
        for i, name in enumerate(CLASS_NAMES):
            p = float(probs[i]) if i < len(probs) else 0.0
            st.write(f"- {name}: {p:.3f}")
