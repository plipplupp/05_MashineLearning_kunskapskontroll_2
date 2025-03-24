import streamlit as st
import numpy as np
import cv2
import time
import joblib
from streamlit_drawable_canvas import st_canvas

#CSS
# def local_css(file_name):
#     with open(file_name) as f:
#         st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# local_css("pages/style.css")

# Ladda modellen och scaler
model = joblib.load('mnist_random_forest_final_compress5.joblib')
scaler = joblib.load('scaler.joblib')

def preprocess_image(image_data, show_images=False):
    start_time_preprocess = time.time()  # Starta tidmätningen
    if image_data is not None:
        image = image_data
        if show_images:
            st.image(image, caption="Oprocessad bild", use_container_width=True)
        print(f"Första bilden: {image.shape}")

        # Konvertera till gråskala
        image = cv2.cvtColor(image_data.astype(np.uint8), cv2.COLOR_RGBA2GRAY)
        if show_images:
            st.image(image, caption="Gråskala bild", use_container_width=True)
        print(f"Form efter gråskal: {image.shape}")

        # Tröskelvärde för att binärisera bilden
        _, image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
        if show_images:
            st.image(image, caption="Tröskelbild", use_container_width=True)
        print(f"Form efter tröskelvärde: {image.shape}")

        # Anpassa storlek till 28x28
        image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)
        if show_images:
            st.image(image, caption="Resized bild", use_container_width=True)
        print(f"Form efter resize: {image.shape}")

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
                st.image(image, caption="Centrerad bild", use_container_width=True)
            print(f"Form efter centrering: {image.shape}")

        # Normalisera pixelvärden till [0, 1]
        image = image / 255.0
        if show_images:
            st.image(image, caption="Normaliserad bild", use_container_width=True)
        print(f"Form efter normalisering: {image.shape}")
        print(f"Minsta värde efter normalisering: {image.min()}")
        print(f"Största värde efter normalisering: {image.max()}")
        print(f"Datatyp efter normalisering: {image.dtype}")

        # Platta ut och standardisera med StandardScaler
        image = image.ravel().reshape(1, -1)
        print(f"Form före standardisering: {image.shape}")

        image = scaler.transform(image)
        print(f"Form efter standardisering: {image.shape}")

        end_time_preprocess = time.time()  # Stoppa tidmätningen
        elapsed_time_preprocess = end_time_preprocess - start_time_preprocess  # Beräkna förfluten tid

        return image, elapsed_time_preprocess
    return None, None

st.title("Draw a digit")

# Skapa en canvas
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=10,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if st.button("Predict"):
    start_time = time.time()
    image, elapsed_time_preprocess = preprocess_image(canvas_result.image_data, show_images=False)  # Visa inte bilder under prediktion
    if image is not None:
        prediction = model.predict(image)[0]
        end_time = time.time()
        elapsed_time = end_time - start_time
        st.markdown(f"### **Prediction: {prediction}**")
        st.write(f"Preprocess time: {elapsed_time_preprocess:.4f} seconds") 
        st.write(f"Prediction time: {elapsed_time:.4f} seconds")

        st.write("Here's how your image was processed:")
        preprocess_image(canvas_result.image_data, show_images=True)  # Visa bilder efter prediktion
    else:
        st.write("Rita en siffra först!")