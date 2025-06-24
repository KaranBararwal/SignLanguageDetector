import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import tensorflow as tf
from tensorflow.keras.models import load_model
import av

# Load model
model = tf.keras.models.load_model("asl_model.keras")

import os
print("Exists:", os.path.exists("asl_model.keras"))


labels_map = [chr(i) for i in range(65, 91) if i != 74]  # A-Z excluding 'J'

def predict_sign(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28))
    norm = resized / 255.0
    reshaped = norm.reshape(1, 28, 28, 1)
    prediction = model.predict(reshaped)
    return labels_map[np.argmax(prediction)]

class SignDetector(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        x1, y1, x2, y2 = 100, 100, 300, 300
        roi = img[y1:y2, x1:x2]
        try:
            label = predict_sign(roi)
            cv2.putText(img, f"Prediction: {label}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        except:
            pass
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

st.title("🧠 Sign Language Live Detector")
webrtc_streamer(key="live", video_processor_factory=SignDetector)