"""
Aim:
Build a Streamlit web application that:
1. Loads the trained U-Net model (.h5).
2. Lets the user upload any photo.
3. Converts the photo into a sketch using the model.
4. Displays the input photo and predicted sketch side by side.
5. Credits: Built by Satwik Shreshth.
"""

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

# --------------------------
# Loss functions (required for loading)
# --------------------------
def ssim_loss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

def combined_loss(y_true, y_pred):
    mae = tf.reduce_mean(tf.abs(y_true - y_pred))
    return mae + ssim_loss(y_true, y_pred)

# --------------------------
# Load model
# --------------------------
MODEL_PATH = r"C:\Users\dell\Desktop\Image to Sketch\Latest_image2Sketch_Model.h5"

@st.cache_resource
def load_unet():
    model = load_model(
        MODEL_PATH,
        compile=False,
        custom_objects={"combined_loss": combined_loss, "ssim_loss": ssim_loss}
    )
    return model

unet_model = load_unet()

# --------------------------
# Preprocess uploaded image
# --------------------------
def preprocess_image(img, img_size=128):
    img = img.convert("RGB")
    img = img.resize((img_size, img_size))
    img = np.array(img).astype("float32") / 255.0
    return img

# --------------------------
# Prediction
# --------------------------
def predict_sketch(photo):
    inp = np.expand_dims(photo, axis=0)
    pred = unet_model.predict(inp)[0, :, :, 0]
    return pred

# --------------------------
# Streamlit UI
# --------------------------
st.title("📸 Image to Sketch Generator")
st.subheader("✨ Built by **Satwik Shreshth**")
st.write("Upload a photo and the model will convert it into a sketch.")

uploaded_file = st.file_uploader("Upload a Photo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    # Read image
    img = Image.open(uploaded_file)

    # Preprocess
    processed = preprocess_image(img)

    # Predict sketch
    sketch = predict_sketch(processed)

    # Display
    col1, col2 = st.columns(2)

    with col1:
        st.image(img, caption="Input Photo", use_column_width=True)

    with col2:
        st.image(sketch, caption="Predicted Sketch", clamp=True, use_column_width=True)

    st.success("Sketch generated successfully!")
