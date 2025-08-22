import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps

# Load the trained model
model = load_model("mnist_model.h5")

st.title("MNIST Digit Recognition")
st.write("Upload a digit image and the model will predict it.")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Open image
    image = Image.open(uploaded_file).convert("L")  # Convert to grayscale

    # Preprocess: resize → normalize → flatten
    img_resized = ImageOps.fit(image, (28, 28), Image.Resampling.LANCZOS)
    img_array = np.array(img_resized)

    # Invert if background is white
    if np.mean(img_array) > 127:
        img_array = 255 - img_array

    img_array = img_array / 255.0  # normalize
    img_array = img_array.reshape(1, 28*28)  # flatten

    # Show original and processed
    st.image(image, caption="Uploaded Image", width=150)
    st.image(img_array.reshape(28, 28), caption="Processed (what model sees)", width=150)

    # Predict
    prediction = model.predict(img_array)
    pred_label = np.argmax(prediction)

    st.subheader(f"Predicted Digit: {pred_label}")
    st.write(f"Confidence: {100*np.max(prediction):.2f}%")
