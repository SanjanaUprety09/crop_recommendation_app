import streamlit as st
import pickle
import numpy as np

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

st.title("Crop Recommendation System")

N = st.number_input("Nitrogen (N)", min_value=0)
P = st.number_input("Phosphorus (P)", min_value=0)
K = st.number_input("Potassium (K)", min_value=0)
temperature = st.number_input("Temperature (°C)")
humidity = st.number_input("Humidity (%)")
ph = st.number_input("pH Value")
rainfall = st.number_input("Rainfall (mm)")

if st.button("Predict Crop"):
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

    prediction = model.predict(input_data)[0]  # this will be an encoded label like 0, 1, 2...

    predicted_crop = le.inverse_transform([prediction])[0]

    st.success(f" Recommended Crop: **{predicted_crop}**")
