import streamlit as st 
import pandas as pd
import numpy as np 
import tensorflow as tf
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

# load the trained model 
model = tf.keras.models.load_model("model.h5")

#load other pick files 
with open("label_encoder_gender.pkl","rb") as file:
    label_encoder_gender = pickle.load(file)
with open("oneHot_Encoder.pkl","rb") as file:
    oneHot_Encoder = pickle.load(file)
with open("standard_scalar.pkl","rb") as file:
    standard_scalar = pickle.load(file)
    
#streamlit app
st.title("Customer Churn Prediction")


# user input 
geography = st.selectbox("Geography",oneHot_Encoder.categories_[0])
gender = st.selectbox("Gender", label_encoder_gender.classes_)
age = st.slider("Age",18,92)
balence = st.number_input("Balance")
credit_score = st.number_input("Credit Score")
estimated_salary = st.number_input("Estimated Salary")
tenure = st.slider("Tenure",0,10)
num_of_products = st.slider("Number Of Products", 1,4)
has_cr_card = st.selectbox("Has Credit Card",[0,1])
is_active_member = st.selectbox("Is Active Member",[0,1])
print("Load all the details")
# Prepare the input data
input_data =pd.DataFrame({
    "CreditScore":[credit_score],
    "Gender":[label_encoder_gender.transform([gender])[0]],
    "Age":[age],
    "Tenure":[tenure],
    "Balance":[balence],               
    "NumOfProducts":[num_of_products],
    "HasCrCard":[has_cr_card],
    "IsActiveMember":[is_active_member],
    "EstimatedSalary":[estimated_salary]
})
print("assign all the values")

print("before geo_encoding")

geo_encoded = oneHot_Encoder.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=oneHot_Encoder.get_feature_names_out(["Geography"]))
print("after geo_encoding")

#combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)

#scale the input data
input_data_scaled = standard_scalar.transform(input_data)

#prediction churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]
st.write(f"Customer Probabilities : {prediction_proba}")
if prediction_proba >0.5:
    st.write("Customer is likely to Churn")
else:
    st.write("Customer is not likely to churn")
