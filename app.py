import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model and encoders
rf = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")

st.title("🩸 Anemia Prediction App")

# Choose input method
option = st.radio("Choose Input Method:", ("Upload CSV", "Manual Entry"))

if option == "Upload CSV":
    uploaded_file = st.file_uploader("Upload patient CSV", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        # Drop PatientID if present
        if 'PatientID' in data.columns:
            features = data.drop(columns=['PatientID'])
        else:
            features = data
        scaled = scaler.transform(features)
        preds = rf.predict(scaled)
        data['Prediction'] = le.inverse_transform(preds)
        st.write(data)

else:
    st.subheader("Manual Patient Data Entry")

    age = st.number_input("Age", min_value=1, max_value=120)
    gender = st.selectbox("Gender", ["Female", "Male"])  # better UX
    rbc = st.number_input("RBC (10¹²/L)")
    hgb = st.number_input("HGB (g/dL)")
    hct = st.number_input("HCT (%)")
    mcv = st.number_input("MCV (fL)")
    mch = st.number_input("MCH (pg)")
    mchc = st.number_input("MCHC (g/dL)")
    rdw = st.number_input("RDW-CV (%)")

    # Convert gender to numeric (match training)
    gender_val = 0 if gender == "Female" else 1

    # Submit button
    if st.button("🔍 Predict Anemia"):
        input_data = np.array([[
            age, gender_val, rbc, hgb, hct, mcv, mch, mchc, rdw
        ]])

        scaled = scaler.transform(input_data)
        pred = rf.predict(scaled)
        result = le.inverse_transform(pred)[0]

        st.success(f"🩸 Prediction: **{result}**")
