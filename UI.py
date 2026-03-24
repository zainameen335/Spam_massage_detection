import streamlit as st
import requests

st.title("Spam Message Detection")

user_input = st.text_area("Enter your message:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message!")
    else:
        response = requests.post("http://127.0.0.1:8000/predict", json={"message": user_input})
        prediction = response.json()["prediction"]
        st.success(f"Prediction: {prediction}")