import streamlit as st
import joblib
import numpy as np
import os

st.set_page_config(page_title="Disease Predictor", layout="centered")
st.title("🧬 Multi-Disease Prediction App")
disease = st.selectbox("Select Disease to Predict", ["Heart Disease", "Diabetes", "Breast Cancer"])

# === HEART DISEASE ===
if disease == "Heart Disease":
    st.subheader("💓 Heart Disease Prediction")
    try:
        model = joblib.load("models/heart_model.pkl")
        scaler = joblib.load("models/scaler.pkl")
    except:
        st.error("Heart model/scaler not found.")
        st.stop()

    age = st.slider("Age", 20, 100, 50)
    sex = st.selectbox("Sex (1 = male, 0 = female)", [1, 0])
    cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
    chol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = true, 0 = false)", [0, 1])
    restecg = st.selectbox("Resting ECG results (0–2)", [0, 1, 2])
    thalach = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
    exang = st.selectbox("Exercise-induced Angina (1 = yes, 0 = no)", [0, 1])
    oldpeak = st.number_input("ST depression induced by exercise", 0.0, 6.0, 1.0)
    slope = st.selectbox("Slope of ST segment (0–2)", [0, 1, 2])
    ca = st.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia (0 = normal, 1 = fixed defect, 2 = reversible defect)", [0, 1, 2])

    if st.button("Predict"):
        data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                          thalach, exang, oldpeak, slope, ca, thal]])
        scaled = scaler.transform(data)
        result = model.predict(scaled)[0]
        st.success("✅ No Heart Disease" if result == 0 else "⚠️ Heart Disease Detected")

# === DIABETES ===
elif disease == "Diabetes":
    st.subheader("🩸 Diabetes Prediction")
    try:
        model = joblib.load("models/diabetes_model.pkl")
        scaler = joblib.load("models/diabetes_scaler.pkl")
    except:
        st.error("Diabetes model/scaler not found.")
        st.stop()

    pregnancies = st.slider("Pregnancies", 0, 20, 1)
    glucose = st.slider("Glucose", 0, 200, 100)
    bp = st.slider("Blood Pressure", 0, 140, 70)
    skin = st.slider("Skin Thickness", 0, 100, 20)
    insulin = st.slider("Insulin", 0, 900, 80)
    bmi = st.slider("BMI", 0.0, 70.0, 25.0)
    dpf = st.slider("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
    age = st.slider("Age", 10, 100, 30)

    if st.button("Predict"):
        data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
        scaled = scaler.transform(data)
        result = model.predict(scaled)[0]
        st.success("✅ No Diabetes" if result == 0 else "⚠️ Diabetes Detected")

# === BREAST CANCER ===
elif disease == "Breast Cancer":
    st.subheader("🧪 Breast Cancer Prediction")
    try:
        model = joblib.load("models/breast_cancer_model.pkl")
        scaler = joblib.load("models/breast_cancer_scaler.pkl")
    except:
        st.error("Breast cancer model/scaler not found.")
        st.stop()

    # List of all 30 input features from breast_cancer.csv
    features = [
        "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
        "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean", "fractal_dimension_mean",
        "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
        "compactness_se", "concavity_se", "concave points_se", "symmetry_se", "fractal_dimension_se",
        "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
        "compactness_worst", "concavity_worst", "concave points_worst", "symmetry_worst", "fractal_dimension_worst"
    ]

    user_input = []
    for feature in features:
        val = st.number_input(f"{feature.replace('_', ' ').capitalize()}:", min_value=0.0, step=0.1)
        user_input.append(val)

    if st.button("Predict"):
         data = np.array([user_input])
        scaled = scaler.transform(data)
        result = model.predict(scaled)[0]
        st.success("✅ Tumor is Benign" if result == 0 else "⚠️ Tumor is Malignant")
# to run this use this command in the terminal (streamlit run app.py)

        data = np.array([user_input])
        scaled = scaler.transform(data)
        result = model.predict(scaled)[0]
        st.success("✅ Tumor is Benign" if result == 0 else "⚠️ Tumor is Malignant")
