# -*- coding: utf-8 -*-
"""
Created on Thu Sep 5 13:52:40 2024

@author: ELBOSTAN
"""

import numpy as np
import pickle
import streamlit as st

# Loading the trained model
loaded_model = pickle.load(open('C:/Users/ELBOSTAN/Desktop/AI/Projects/My project/heart_disease_model.sav', 'rb'))

# Creating a function for cancer prediction
def Heart_Disease_Prediction(input_data):
    
    data = [58,0,0,100,248,0,0,156,1,0,2,0,3] # Example data with 13 features

    #converting the data into numppy array to be ready for prediction
    data_as_array = np.asarray(data)

    #reshaping the data to fit the model
    data_reshaped = data_as_array.reshape(1,-1)

    #prediction
    prediction = loaded_model.predict(data_reshaped)
    print(prediction)

    if (prediction[0]== 0):
        return'The Person does not have a Heart Disease'
    else:
         return'The Person has Heart Disease'


# Main function for testing the code
import streamlit as st

def main():
    # giving a title
    st.title("Heart Disease Prediction")
    
    # User inputs
    age = st.text_input("Age")
    sex = st.text_input("Your Gender")
    cp = st.text_input("Chest pain type")
    trestbps = st.text_input("Resting blood pressure")
    chol = st.text_input("Serum cholesterol in mg/dl")
    fbs = st.text_input("Fasting blood sugar")
    restecg = st.text_input("Resting electrocardiographic results")
    thalach = st.text_input("Maximum heart rate achieved")
    exang = st.text_input("Exercise-induced angina")
    oldpeak = st.text_input("ST depression induced by exercise relative to rest")
    slope = st.text_input("Slope of the peak exercise ST segment")
    ca = st.text_input("Number of major vessels (0-3) colored by fluoroscopy")
    thal = st.text_input("Thal: 0 = normal; 1 = fixed defect; 2 = reversible defect")

    # Code for prediction
    diagnosis = ''
    
    # Creating a button for prediction
    if st.button('Heart Status'):
           
            # Make prediction (assuming the function diabetes_prediction exists)
            diagnosis = Heart_Disease_Prediction([age, sex, cp, trestbps, chol, fbs, restecg,
                                             thalach, exang, oldpeak, slope, ca, thal])
        
    # Display diagnosis
    st.success(diagnosis)

if __name__ == '__main__':
    main()

    
    
    
    
    
    
    
    
    
    
    
    
    
    