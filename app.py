import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Load pre-trained models
try:
    pca = joblib.load('pca_model.joblib')
    model = joblib.load('best_model.joblib')
except Exception as e:
    st.error(f"Error loading models: {e}")

st.title("Breast Cancer Prediction App")
st.write("Enter the clinical measurements below to predict diagnosis.")

# List of 30 expected feature names
feature_names = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
    'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
    'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

# Organize inputs
inputs = {}
st.sidebar.header("Input Features")
for name in feature_names:
    inputs[name] = st.sidebar.number_input(f"{name}", value=0.0)

if st.button("Predict"):
    # Format inputs into DataFrame
    input_df = pd.DataFrame([inputs])

    # Project inputs to PCA space
    try:
        input_pca = pca.transform(input_df)
        # Predict
        prediction = model.predict(input_pca)[0]
        probability = model.decision_function(input_pca)[0] if hasattr(model, "decision_function") else None

        st.subheader("Prediction Result:")
        if prediction == 1:
            st.error("Result: Malignant (M)")
        else:
            st.success("Result: Benign (B)")

        if probability is not None:
            st.write(f"Decision Score: {probability:.4f}")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
