import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Digit Recognition",
    page_icon="",
)

st.sidebar.success("Select a page above.")

st.title("Handwritten Digit Recognition with Machine Learning")

# Bild på startsidan
image = Image.open("Cyborg_Machine_Learning.jpg")
st.image(image, caption="Cyborg Machine Learning", use_container_width=True)