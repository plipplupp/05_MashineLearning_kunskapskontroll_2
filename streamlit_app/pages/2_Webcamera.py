import streamlit as st
import numpy as np
import joblib 
import cv2
from PIL import Image

# Ladda modellen och scaler
model = joblib.load('../../mnist_final_voting_classifier_v2.joblib')
scaler = joblib.load('../scaler.joblib')

def preprocess_image(image_data):
    if image_data is not None:
        image = np.array(image_data)
        st.image(image, caption="Oprocessad bild", use_container_width=True)

        # Konvertera till gråskala
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        st.image(image, caption="Gråskala bild", use_container_width=True)

        # Medianoskärpa och adaptiv tröskelvärdesättning (första steget)
        image = cv2.medianBlur(image, 5)
        image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        st.image(image, caption="Tröskelbild 1", use_container_width=True)

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

        st.image(image, caption="Beskuren bild", use_container_width=True)

        # Anpassa storlek till 28x28
        image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)
        st.image(image, caption="Resized bild", use_container_width=True)

        # Beräkna center of mass och centrera siffran
        moments = cv2.moments(image)
        if moments["m00"] != 0:
            center_x = int(moments["m10"] / moments["m00"])
            center_y = int(moments["m01"] / moments["m00"])
            shift_x = 14 - center_x
            shift_y = 14 - center_y
            shift_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            image = cv2.warpAffine(image, shift_matrix, (28, 28), borderValue=0)
            st.image(image, caption="Centrerad bild", use_container_width=True)

        # Normalisera pixelvärden till [0, 1]
        image = image / 255.0
        st.image(image, caption="Normaliserad bild", use_container_width=True)

        # Platta ut och standardisera med StandardScaler
        image = image.ravel().reshape(1, -1)
        image = scaler.transform(image)

        return image
    return None

st.title("Webbkameraigenkänning")

picture = st.camera_input("Ta en bild")

if picture:
    image = Image.open(picture)
    processed_image = preprocess_image(image)
    if processed_image is not None:
        prediction = model.predict(processed_image)[0]
        st.write(f"Prediktion: {prediction}")