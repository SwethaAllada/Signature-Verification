import streamlit as st
import torch
from PIL import Image
import numpy as np
from ultralytics import YOLO  # Import YOLO class directly

# Load the YOLO model
model = YOLO('sign.pt')  # Adjust the path if necessary

# Streamlit interface
st.title("Signature Verification")
st.write("Upload a signature image to verify its authenticity.")

# Image uploader
uploaded_file = st.file_uploader("Choose a signature image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Signature", use_column_width=True)
    
    # Convert image to a format compatible with YOLO
    image_np = np.array(image)
    
    # Run inference
    results = model(image_np)
    
    # Display the results with bounding boxes
    results_img = Image.fromarray(results[0].plot())
    st.image(results_img, caption="Verification Result", use_column_width=True)
    
    # Display confidence scores and labels
    st.write("Detected signature match confidence:")
    for pred in results[0].boxes.data:  # Access the bounding box data
        label = model.names[int(pred[-1])]  # Label name
        confidence = pred[4].item()  # Confidence score
        st.write(f"{label}: {confidence:.2f}")
