import streamlit as st
import numpy as np
import joblib 
import cv2
import joblib
import time
from PIL import Image


# Ladda modellen och scaler
model = joblib.load('mnist_random_forest_final_compress5.joblib')
scaler = joblib.load('scaler.joblib')

def preprocess_image(image_data, show_images=True):
    if image_data is not None:
        image = np.array(image_data)
        if show_images:
            st.image(image, caption="Unprocessed Image", use_container_width=True)

        # Konvertera till gråskala
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        if show_images:
            st.image(image, caption="Grayscale Image", use_container_width=True)

        # Medianoskärpa och adaptiv tröskelvärdesättning (första steget)
        image = cv2.medianBlur(image, 5)
        image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        if show_images:
            st.image(image, caption="Thresholded Image 1", use_container_width=True)

        # Konturdetektering och ROI-beskärning
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(contour)
            margin = 30  # Justera marginalerna efter behov
            x -= margin
            y -= margin
            w += 2 * margin
            h += 2 * margin
            #Se till att rektangeln inte hamnar utanför bilden
            x = max(0,x)
            y = max(0,y)
            w = min(image.shape[1] - x, w)
            h = min(image.shape[0] - y, h)
            image = image[y:y+h, x:x+w]  # Beskär bilden

        if show_images:
            st.image(image, caption="Cropped Image", use_container_width=True)

        # Anpassa storlek till 28x28
        image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)
        if show_images:
            st.image(image, caption="Resized Image", use_container_width=True)

        # Beräkna center of mass och centrera siffran
        moments = cv2.moments(image)
        if moments["m00"] != 0:
            center_x = int(moments["m10"] / moments["m00"])
            center_y = int(moments["m01"] / moments["m00"])
            shift_x = 14 - center_x
            shift_y = 14 - center_y
            shift_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            image = cv2.warpAffine(image, shift_matrix, (28, 28), borderValue=0)
        if show_images:
            st.image(image, caption="Centered Image", use_container_width=True)

        # Normalisera pixelvärden till [0, 1]
        image = image / 255.0
        if show_images:
            st.image(image, caption="Normalized Image", use_container_width=True)

        # Platta ut och standardisera med StandardScaler
        image = image.ravel().reshape(1, -1)
        image = scaler.transform(image)

        return image
    return None

st.title("Use the camera to capture a handwritten digit for recognition.")

picture = st.camera_input("Capture a photo")

if picture:
    start_time = time.time()
    image = Image.open(picture)
    processed_image = preprocess_image(image, show_images=False)  # Visa inte bilder under prediktion
    if processed_image is not None:
        prediction = model.predict(processed_image)[0]
        end_time = time.time()
        elapsed_time = end_time - start_time
        st.markdown(f"### **Prediction: {prediction}**")
        st.write(f"Prediction time: {elapsed_time:.4f} seconds")

        st.write("Here's how your image was processed:")
        preprocess_image(image, show_images=True)  # Visa bilder efter prediktion
    else:
        st.write("Please capture an image first!")