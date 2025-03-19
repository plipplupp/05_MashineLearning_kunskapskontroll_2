import streamlit as st
import cv2
import numpy as np
import joblib
from streamlit_webrtc import webrtc_streamer

# Ladda modellen och scaler
model = joblib.load('../../mnist_final_voting_classifier_v2.joblib')
scaler = joblib.load('../scaler.joblib')

def preprocess_image(frame):
    if frame is not None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(contour)
            roi = gray[y:y+h, x:x+w]
            resized_roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
            moments = cv2.moments(resized_roi)
            if moments["m00"] != 0:
                center_x = int(moments["m10"] / moments["m00"])
                center_y = int(moments["m01"] / moments["m00"])
                shift_x = 14 - center_x
                shift_y = 14 - center_y
                shift_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
                centered_roi = cv2.warpAffine(resized_roi, shift_matrix, (28, 28), borderValue=0)
                normalized_roi = centered_roi / 255.0
                processed_roi = normalized_roi.ravel().reshape(1, -1)
                processed_roi = scaler.transform(processed_roi)
                return processed_roi
    return None

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    processed_img = preprocess_image(img)
    if processed_img is not None:
        prediction = model.predict(processed_img)[0]
        st.write(f"Prediktion: {prediction}")
    return frame

st.title("Videoigenkänning av siffror")
webrtc_streamer(key="example", video_frame_callback=video_frame_callback)