import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

# Load the model
model = joblib.load('RF.pkl')

# Define feature options
Grade_options = {    
    1: 'Well (1)',    
    2: 'Moderate (2)',    
    3: 'Poor (3)'}

# Define feature names
feature_names = [    
    "Size", "DOI", "TT", "TB", "BASO",    
    "NLR", "Grade", "PNI", "LVI"
]

# Streamlit user interface
st.title("OLNM Predictor")

# Size: numerical input
Size = st.number_input("Tumor Size (mm):", min_value=1, max_value=40, value=10)

# DOI: numerical input
DOI = st.number_input("DOI (mm):", min_value=0.1, max_value=10.2, value=6.8)

# TT: numerical input
TT = st.number_input("Tumor Tickness (mm):", min_value=0.01, max_value=20.0, value=10.0)

# TB: numerical input
TB = st.number_input("Tumor Budding:", min_value=0, max_value=36, value=16)

# BASO%: numerical input
BASO = st.number_input("BASO:", min_value=0, max_value=1.5, value=0.8)

# NLR: numerical input
NLR = st.number_input("Neutrophil-to-Lymphocyte Ratio:", min_value=0.00, max_value=6.00, value=3.20)

# Grade: categorical selection
Grade = st.selectbox("Tumor Grade:", options=list(Grade_options.keys()), format_func=lambda x: Grade_options[x])

# PNI: categorical selection
PNI = st.selectbox("PNI:", options=[0, 1], format_func=lambda x: 'NO (0)' if x == 0 else 'Yes (1)')

# LVI: categorical selection
LVI = st.selectbox("LVI:", options=[0, 1], format_func=lambda x: 'NO (0)' if x == 0 else 'Yes (1)')

# Process inputs and make predictions
feature_values = [Size, DOI, TT, TB, BASO, NLR, Grade, PNI, LVI]
features = np.array([feature_values])

if st.button("Predict"):
    # Predict class and probabilities
    predicted_class = model.predict(features)[0]  # Ensure this is an integer value (e.g., 0 or 1)
    predicted_proba = model.predict_proba(features)[0]  # Array of probabilities

    # Display prediction results
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Prediction Probabilities:** {predicted_proba}")

    # Calculate the probability of the predicted class
    probability = predicted_proba[predicted_class] * 100

    if predicted_class == 1:
        advice = (
            f"According to our model, you have a high risk of occult lymph node metastasis. "
            f"The model predicts that your probability of having occult lymph node metastasis is {probability:.1f}%. "
            "While this is just an estimate, it suggests that you may be at significant risk. "
            "I recommend that you receive neck dissection, this procedure is intended to remove any potentially affected lymph nodes in the neck region, "
            "which may help prevent the spread of cancer and improve your overall prognosis."
        )
    else:
        advice = (
            f"According to our model, you have a low risk of occult lymph node metastasis. "
            f"The model predicts that your probability of not having occult lymph node metastasis is {probability:.1f}%. "
            "However, maintain a healthy lifestyle is still very important. "
            "I recommend regular check-ups to monitor your health, "
            "and to seek medical advice promptly if you experience any symptoms."
        )

    # Display advice
    st.write(advice)

# Calculate SHAP values and display force plot    
    
    explainer_shap = shap.TreeExplainer(model)    
    
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))

    if predicted_class == 1:
        shap.force_plot(explainer_shap.expected_value[1], shap_values[:,:,1], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)    
    else:        
        shap.force_plot(explainer_shap.expected_value[0], shap_values[:,:,0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)    
        
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)    
       
    st.image("shap_force_plot.png", caption='SHAP Force Plot Explanation')    