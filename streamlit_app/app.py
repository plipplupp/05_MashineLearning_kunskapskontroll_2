import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Digit Recognition",
    page_icon="",
)

st.sidebar.success("Select a page above.")

st.title("Handwritten Digit Recognition with Machine Learning")

# Bild på startsidan
try:
    image = Image.open("cyborg.jpg")
    st.image(image, caption="Cyborg Machine Learning", use_container_width=True)
except FileNotFoundError:
    st.error("Bilden 'cyborg.jpg' kunde inte hittas.")

st.write("## **About the App**")

st.write("### This app uses a machine learning model trained on the MNIST dataset to recognize handwritten digits.")
st.write("### You can choose to draw a digit with the mouse or use the webcam.")