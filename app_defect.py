import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("defect_model.h5")

# IMPORTANT: derive class names from training order if possible
# (fallback to manual list if needed)
try:
    # If you saved class_indices earlier, load them; otherwise keep manual list
    classes = ["crazing", "inclusion", "patches", "pitted_surface", "rolled_in_scale", "scratches"]
except:
    classes = ["crazing", "inclusion", "patches", "pitted_surface", "rolled_in_scale", "scratches"]

st.set_page_config(page_title="Defect Detection", layout="centered")
st.title("Defect Detection System")

file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if file:
    img = Image.open(file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img_resized = img.resize((150,150))
    x = np.array(img_resized) / 255.0
    x = np.expand_dims(x, axis=0)

    # Predict
    preds = model.predict(x)[0]
    top_idx = np.argmax(preds)
    confidence = preds[top_idx]

    st.subheader("Prediction")
    st.success(f"{classes[top_idx]}  (confidence: {confidence:.2f})")

    # Show top-3 probabilities
    st.write("Top predictions:")
    top3 = np.argsort(preds)[-3:][::-1]
    for i in top3:
        st.write(f"{classes[i]}: {preds[i]:.2f}")