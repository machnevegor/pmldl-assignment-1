import os

import requests
import streamlit as st

ENDPOINT = os.getenv("ENDPOINT")

st.title("Blonde Test")
st.write("Attach your photo and find out if you're blonde or not!")

uploaded_file = st.camera_input("Smile :D")

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image")

    if st.button("Predict"):
        response = requests.post(ENDPOINT, files={"file": uploaded_file.getvalue()})

        if response.status_code == 200:
            prediction = response.json()["prediction"]
            st.write(f"Prediction: {prediction}")
        else:
            st.write(f"Error: {response.status_code}, {response.text}")
