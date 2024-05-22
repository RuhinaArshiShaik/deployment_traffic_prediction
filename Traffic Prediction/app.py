import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

# Load your machine learning model
model_traffic = pickle.load(open('model.pkl', 'rb'))

st.title('Traffic Analysis')

# Input fields for traffic analysis
st.header('TRAFFIC ANALYSIS')

coded_day_traffic = st.text_input('Enter the Coded Day (if applicable):')
zone_traffic = st.text_input('Enter the Zone (if applicable):')
weather_traffic = st.text_input('Enter the Weather (e.g., Sunny, Rainy, etc.):')
temperature_traffic = st.text_input('Enter the Temperature (in Celsius):')

# Prediction button for traffic analysis
if st.button('Predict Traffic'):
    data_traffic = [[coded_day_traffic, zone_traffic, weather_traffic, temperature_traffic]]
    # Convert inputs to appropriate data types
    # data=pd.read_csv('Dataset.csv')
    # X=data.iloc[:,2:6].values
    # Y=data.iloc[:,6:7].values
    # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)
    # X_test=np.append(X_test,data_traffic,axis=0)
    # print(X_test)
    # sc_X=StandardScaler()
    # X_train = sc_X.fit_transform(X_train)
    # X_test = sc_X.fit_transform(X_test)
    # data_traffic=[X_test[-1,:]]


    # Debugging: Print input data
    st.write(f"Input Data: {data_traffic}")

    result_traffic = model_traffic.predict(data_traffic)

    # Debugging: Print raw output from the model
    st.write(f"Raw Model Output: {result_traffic}")

    # Display result for traffic analysis
    traffic_level = result_traffic[0]

    # Debugging: Print traffic level
    st.write(f"Traffic Level: {traffic_level}")

    # Map predicted traffic levels to specified conditions using greater or lesser than
    if result_traffic[0] > 2.5:
        st.warning("Traffic is detected. Expect delays.")
    else:
        st.success("No significant traffic detected. Have a safe journey!")

