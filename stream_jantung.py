# -*- coding: utf-8 -*-
"""stream-jantung.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1sg0ir2YOm9_hnmtdPdhCg1TGzAvQgIGM
"""

import pickle
import numpy as np

!pip install matplotlib-venn

pip install streamlit

pip install -r requirements.txt
streamlit run stream_jantung.py

import streamlit as st

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

#load save model
model = pickle.load(open('penyakitjantung_model.sav','rb'))

#judul web
st.title('Prediksi Penyakit Jantung')

st.write('Please enter the following details for heart disease prediction:')

age = st.slider('Age', min_value=20, max_value=80, step=1)
sex = st.radio('Sex', ['Male', 'Female'])
cp = st.selectbox('Chest Pain Type', ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'])
trestbps = st.slider('Resting Blood Pressure (mm Hg)', min_value=90, max_value=200, step=1)
chol = st.slider('Serum Cholesterol (mg/dl)', min_value=100, max_value=600, step=1)
fbs = st.radio('Fasting Blood Sugar > 120 mg/dl', ['True', 'False'])
restecg = st.selectbox('Resting Electrocardiographic Results', ['Normal', 'ST-T wave abnormality', 'Probable or definite left ventricular hypertrophy'])
thalach = st.slider('Maximum Heart Rate Achieved (bpm)', min_value=70, max_value=220, step=1)
exang = st.radio('Exercise Induced Angina', ['Yes', 'No'])
oldpeak = st.slider('ST Depression Induced by Exercise Relative to Rest', min_value=0.0, max_value=6.2, step=0.1)
slope = st.selectbox('Slope of the Peak Exercise ST Segment', ['Upsloping', 'Flat', 'Downsloping'])
ca = st.slider('Number of Major Vessels Colored by Fluoroscopy', min_value=0, max_value=4, step=1)
thal = st.selectbox('Thalassemia', ['Normal', 'Fixed Defect', 'Reversible Defect'])

sex = 1 if sex == 'Male' else 0
fbs = 1 if fbs == 'True' else 0
exang = 1 if exang == 'Yes' else 0

cp_map = {'Typical Angina': 0, 'Atypical Angina': 1, 'Non-anginal Pain': 2, 'Asymptomatic': 3}
restecg_map = {'Normal': 0, 'ST-T wave abnormality': 1, 'Probable or definite left ventricular hypertrophy': 2}
slope_map = {'Upsloping': 0, 'Flat': 1, 'Downsloping': 2}
thal_map = {'Normal': 0, 'Fixed Defect': 1, 'Reversible Defect': 2}

new_data = np.array([[age, sex, cp_map[cp], trestbps, chol, fbs, restecg_map[restecg], thalach, exang, oldpeak, slope_map[slope], ca, thal_map[thal]]])

if st.button('Predict'):
    prediction_prob, prediction_binary = predict_heart_disease(new_data)
    st.write(f"Predicted Probability: {prediction_prob}")
    st.write("Result:", "Presence" if prediction_binary == 1 else "Absence")
