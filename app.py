import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score

# Load the pre-trained model
model = joblib.load('c2_Classifier_LoyalCustomers')

# App title and description
st.title("Marketing Campaign Lead Scoring")
st.write("""
This application allows you to upload customer data, preprocess it, and generate predictions for loyalty classification using a logistic regression model.
""")

# File upload section
st.sidebar.header("Upload Your Dataset")
uploaded_file = st.sidebar.file_uploader("Upload an Excel file", type=["xlsx"])

if uploaded_file:
    # Load dataset
    dataset = pd.read_excel(uploaded_file)
    st.write("Dataset Preview")
    st.write(dataset.head())

    # Preprocess the data
    st.write("Preprocessing data...")
    dataset = dataset.drop(['ID'], axis=1)

    # Fill missing values
    dataset['DemAffl'] = dataset['DemAffl'].fillna(dataset['DemAffl'].mode()[0])
    dataset['DemAge'] = dataset['DemAge'].fillna(dataset['DemAge'].mode()[0])
    dataset['DemClusterGroup'] = dataset['DemClusterGroup'].fillna(dataset['DemClusterGroup'].mode()[0])
    dataset['DemGender'] = dataset['DemGender'].fillna(dataset['DemGender'].mode()[0])
    dataset['DemReg'] = dataset['DemReg'].fillna(dataset['DemReg'].mode()[0])
    dataset['DemTVReg'] = dataset['DemTVReg'].fillna(dataset['DemTVReg'].mode()[0])
    dataset['LoyalTime'] = dataset['LoyalTime'].fillna(dataset['LoyalTime'].mean())

    # Encode categorical variables
    label_encoders = {}
    for column in ['DemClusterGroup', 'DemGender', 'DemReg', 'DemTVReg', 'LoyalClass']:
        le = LabelEncoder()
        dataset[column] = le.fit_transform(dataset[column].astype(str))
        label_encoders[column] = le

    st.write("Processed Dataset")
    st.write(dataset.head())

    # Model Prediction
    X = dataset.iloc[:, :-1]  # Select all columns except the target
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)

    # Show predictions
    st.write("Predictions")
    prediction_df = pd.DataFrame({
        "Predicted Outcome": y_pred,
        "Probability of Class 0": y_pred_proba[:, 0],
        "Probability of Class 1": y_pred_proba[:, 1]
    })
    st.write(prediction_df)

    # Model Evaluation (optional)
    st.write("### Model Performance Metrics")
    if 'Actual Outcome' in dataset.columns:
        y_true = dataset['Actual Outcome']
        accuracy = accuracy_score(y_true, y_pred)
        confusion = confusion_matrix(y_true, y_pred)
        
        st.write("Confusion Matrix")
        st.write(confusion)
        st.write(f"Accuracy: {accuracy:.2f}")

else:
    st.write("Please upload a dataset to start.")
