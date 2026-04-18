import os
# Force TensorFlow to use CPU only
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# ==========================================
# CONSTANTS & CONFIGURATION
# ==========================================
MODEL_PATH = "unet_brain_tumor.keras"
IMG_SIZE = 256

# Configure the Streamlit page
st.set_page_config(
    page_title="Brain Tumor Segmentation",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ==========================================
# SESSION STATE INITIALIZATION
# ==========================================
# A dynamic key for the file uploader allows us to clear it via the Reset button
if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 1

def reset_app():
    st.session_state["uploader_key"] += 1

# ==========================================
# MODEL LOADING
# ==========================================
@st.cache_resource
def load_model():
    try:
        # compile=False allows loading without needing to define custom loss/metrics
        return tf.keras.models.load_model(MODEL_PATH, compile=False)
    except Exception as e:
        st.error(f"Error loading model '{MODEL_PATH}': {e}")
        return None

model = load_model()

# ==========================================
# UI LAYOUT & LOGIC
# ==========================================
st.title("🧠 Brain Tumor Segmentation")
st.markdown("Upload a brain MRI image and use the U-Net model to predict the segmentation mask.")

# Top Controls: Uploader and Reset Button
col_upload, col_reset = st.columns([4, 1])

with col_upload:
    uploaded_file = st.file_uploader(
        "Load Image", 
        type=["png", "jpg", "jpeg"],
        key=st.session_state["uploader_key"],
        label_visibility="collapsed"
    )

with col_reset:
    if st.button("🗑️ Reset", use_container_width=True):
        reset_app()
        st.rerun()

# Image Display and Prediction
if uploaded_file is not None:
    # Open the image using PIL
    raw_image = Image.open(uploaded_file).convert("RGB")
    
    # Create two columns for Original and Predicted Mask
    img_col, mask_col = st.columns(2)
    
    with img_col:
        st.subheader("Original Image")
        # Display the uploaded image
        st.image(raw_image, use_container_width=True)
        
    with mask_col:
        st.subheader("Predicted Mask")
        
        # Predict Button
        if st.button("✨ Predict", use_container_width=True, type="primary"):
            if model is None:
                st.error("Model is not loaded. Cannot predict.")
            else:
                with st.spinner("Generating prediction..."):
                    # 1. Preprocess exactly as in training
                    img_array = np.array(raw_image)
                    img_tensor = tf.convert_to_tensor(img_array)
                    img_tensor = tf.image.resize(img_tensor, [IMG_SIZE, IMG_SIZE])
                    img_tensor = tf.cast(img_tensor, tf.float32) / 255.0
                    
                    # 2. Expand dimensions to create a batch of 1
                    img_expanded = tf.expand_dims(img_tensor, axis=0)
                    
                    # 3. Predict
                    pred_mask = model.predict(img_expanded, verbose=0)[0]
                    
                    # 4. Binarize (threshold > 0.5) and scale to 255
                    pred_mask_bin = (pred_mask > 0.5).astype(np.uint8) * 255
                    
                    # 5. Format for UI Display (Drop the channel dimension)
                    mask_2d = np.squeeze(pred_mask_bin)
                    mask_pil = Image.fromarray(mask_2d, mode='L')
                    
                    # Display the final mask
                    st.image(mask_pil, use_container_width=True)
                    st.success("Prediction complete!")