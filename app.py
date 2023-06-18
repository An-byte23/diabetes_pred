import numpy as np
import pickle
import streamlit as st
# loading the saved model 
loaded_model=pickle.load(open(r'C:\Users\Admin\Desktop\MDPS\trained_model.sav','rb'))
standard_model=pickle.load(open(r'C:\Users\Admin\Desktop\MDPS\standard.sav','rb'))
#  creating a function for prediction
def diabetes_prediction(input_data):

    input_data_as_numpy_array=np.asarray(input_data)
    input_data_reshape=input_data_as_numpy_array.reshape(1,-1)
    std_data=standard_model.transform(input_data_reshape)
    pred=loaded_model.predict(std_data)
    if(pred[0]==0):
        return "The patient is Non-Diabetic"
    else:
        return "The patient is Diabetic"

def main():

    # giving title
    st.title("Diabetes Prediction WebApp")
    # getting input data from the user
    pregnancies=st.text_input('Number of Pregnancies')
    glucose=st.text_input('Blood Glucose Level')
    bp=st.text_input('Blood Pressure Value')
    skin_thick=st.text_input('Skin Thickness Value')
    insulin=st.text_input('Insulin Level')
    bmi=st.text_input('BMI (Body Mass Index) Value')
    dpf=st.text_input('Diabetes Pedigree Function Value')
    age=st.text_input('Age of the patient')

    #code for prediction
    diagnosis = ''
    # creating button for prediction
    if st.button("Get Diabetes Test Result"):
        diagnosis=diabetes_prediction([pregnancies,glucose,bp,skin_thick,insulin,bmi,dpf,age])

    st.success(diagnosis)

if __name__ == '__main__':
    main()
    
    