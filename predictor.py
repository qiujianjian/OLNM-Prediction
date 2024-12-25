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

# Size: numerical input
Size = st.number_input("Tumor size (Size, mm):", min_value=1, max_value=40, value=26)

# DOI: numerical input
DOI = st.number_input("Depth of invasion (DOI, mm):", min_value=0.1, max_value=10.2, value=8.5)

# TT: numerical input
TT = st.number_input("Tumor Thickness (TT, mm):", min_value=0.01, max_value=20.0, value=10.6)

# TB: numerical input
TB = st.number_input("Tumor Budding (TB):", min_value=0, max_value=36, value=18)

# BASO: numerical input
BASO = st.number_input("Basophil percentage(BASO, %):", min_value=0.0, max_value=1.5, value=0.3)

# NLR: numerical input
NLR = st.number_input("Neutrophil-to-Lymphocyte Ratio (NLR):", min_value=0.00, max_value=6.00, value=2.24)

# Grade: categorical selection
Grade = st.selectbox("Tumor Grade (Grade):", options=list(Grade_options.keys()), format_func=lambda x: Grade_options[x])

# PNI: categorical selection
PNI = st.selectbox("Perineural invasion (PNI):", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')

# LVI: categorical selection
LVI = st.selectbox("Lymphovascular invasion (LVI):", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')

# Process inputs and make predictions
feature_values = [Size, DOI, TT, TB, BASO, NLR, Grade, PNI, LVI]
features = np.array([feature_values])

if st.button("Predict"):

    # Predict class and probabilities    
    predicted_class = model.predict(features)[0]    
    predicted_proba = model.predict_proba(features)[0]
    
    # Calculate the probability of the predicted class
    probability = predicted_proba[predicted_class] * 100

    if predicted_class == 1:
        advice = (f"""
                  <div style="text-align: center; font-style: italic; font-weight: bold; font-size: 21px; font-family: 'Times New Roman', Times, serif;">
                  Based on feature values, predicted probability of with OLNM is {probability:.1f}%
                  </div>
                 """
        )
    else:
        advice = (
            f"""
                  <div style="text-align: center; font-style: italic; font-weight: bold; font-size: 21px; font-family: 'Times New Roman', Times, serif;">
                  Based on feature values, predicted probability of without OLNM is {probability:.1f}%
                  </div>
                 """
        )

    # Display advice
    
    st.markdown(advice, unsafe_allow_html=True)

# Calculate SHAP values and display force plot 

explainer_shap = shap.TreeExplainer(model)

shap_values = explainer_shap.shap_values(pd.DataFrame([feature_values], columns=feature_names))

def custom_formatter(x):
    return f"{x:.3f}"
 
   if predicted_class == 1:
        shap.force_plot(
        explainer_shap.expected_value[1], 
        shap_values[:,:,1], 
        pd.DataFrame([feature_values], columns=feature_names),
        matplotlib=True, 
        show=True,
        link='identity',
        feature_names=[f"{name} ({custom_formatter(value)})" for name, value in zip(feature_names, feature_values)]
    )
  else:
    shap.force_plot(
        explainer_shap.expected_value[0], 
        shap_values[:,:,0], 
        pd.DataFrame([feature_values], columns=feature_names),
        matplotlib=True, 
        show=True,
        link='identity',
        feature_names=[f"{name} ({custom_formatter(value)})" for name, value in zip(feature_names, feature_values)]
    )
    
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)        
    st.image("shap_force_plot.png", caption='')
