import pickle
import numpy as np
import streamlit as st

# Load saved model
model = pickle.load(open('penyakitjantung_model.sav','rb'))

# Function to preprocess input data and make predictions
def predict_heart_disease(new_data):
    # Define your preprocessing steps here (e.g., scaling)
    # new_data_scaled = preprocess_data(new_data)
    
    # Make predictions
    predictions = model.predict(new_data)
    binary_predictions = (predictions > 0.5).astype(int)
    return predictions[0][0], binary_predictions[0][0]

# Streamlit app
def main():
    st.title('Prediksi Penyakit Jantung')

    st.write('Please enter the following details for heart disease prediction:')

    # Your UI components here
    
    if st.button('Predict'):
        prediction_prob, prediction_binary = predict_heart_disease(new_data)
        st.write(f"Predicted Probability: {prediction_prob}")
        st.write("Result:", "Presence" if prediction_binary == 1 else "Absence")

if __name__ == '__main__':
    main()
