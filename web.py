import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
import base64

# -------------------------------
# Load YOLO model
# -------------------------------
model = YOLO(r"D:\NTI\project deployment\rock_paper\best (3).pt")  # path to your trained model

# -------------------------------
# Streamlit UI
# -------------------------------
def set_background(image_file):
  
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)
set_background(r"D:\NTI\project deployment\rock_paper\piqsels.com-id-zwfob.jpg")

st.set_page_config(page_title="YOLO Real-Time Detection", layout="centered")

st.title("🖐 Rock Paper Scissors - Real Time YOLO")
st.markdown("Press **Play** to start camera and run YOLO detection")

play = st.button("▶ Play")
stop = st.button("⏹ Stop")

frame_placeholder = st.empty()

# -------------------------------
# Webcam + YOLO Loop
# -------------------------------
if play:
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        st.error("❌ Cannot access camera")
    else:
        while cap.isOpened():
            if stop:
                break

            ret, frame = cap.read()
            if not ret:
                st.warning("⚠ Failed to read frame")
                break

            # YOLO inference
            results = model(frame, conf=0.5, iou=0.5)

            # Draw detections
            annotated_frame = results[0].plot()

            # Convert BGR → RGB
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

            # Display frame
            frame_placeholder.image(
                annotated_frame,
                channels="RGB",
                use_container_width=True
            )

        cap.release()