# -*- coding: utf-8 -*-
import streamlit as st
from streamlit_option_menu import option_menu
import streamlit as st
import proj

# increase font size of title
st.markdown("<h1 style='text-align: center; color: black;'>OcuLens Project</h1>", unsafe_allow_html=True)

# change background colour of website to white


# File Upload
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

# Display Image and Predict
if uploaded_file is not None:
    st.write("Classifying...")

    # Save the uploaded image temporarily
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.read())


    prediction = proj.predicted_class("temp.jpg")

    st.write (f"Actual: {proj.label}")
    st.write(f"Prediction: {prediction}")
