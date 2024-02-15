import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib

# Load your trained model and scaler object
model = load_model('lstm_model.h5')
scaler = joblib.load('scaler.pkl')


# Function to make predictions
def make_prediction(input_data, n_futures):
    prediction = model.predict(input_data)
    # Reverse transformation
    prediction = prediction.reshape(-1, 1)
    dummy_features = np.zeros((prediction.shape[0], 2))  # Adjust the 2 based on number of features - 1
    prediction_combined = np.hstack([prediction, dummy_features])
    prediction_original = scaler.inverse_transform(prediction_combined)[:, 0]
    return prediction_original

# Streamlit app
st.title('Power Consumption Forecast')

# User input for the forecast period
options = {'Next Day': 24, 'Next Two Days': 48, 'Next Three Days': 72}  # Assuming 24 observations per day
selected_option = st.selectbox('Select forecast period:', options.keys())
n_futures = options[selected_option]

# Assuming you have some way to input or use the last available observations
# For example, you could load the most recent observations from a file or use placeholders
df = pd.read_csv("flattened_power_Humidity_temp.csv")
df.drop(columns=['date','time'], inplace = True)
scaler = StandardScaler()
df_trans = scaler.fit_transform(df)
# Extract the last N observations where N is the number of timesteps
# Assuming your LSTM expects 24 timesteps and your data has 3 features
last_observations = np.array(df_trans[-n_futures:].reshape(1, 24, 3))

if st.button('Forecast'):
    predictions = make_prediction(last_observations, n_futures)
    future_dates = pd.date_range(pd.Timestamp.now(), periods=n_futures, freq='H')  # Adjust frequency as needed
    df_forecast = pd.DataFrame({'Date': future_dates, 'Predicted Power': predictions})
    st.write(df_forecast)

# Run this in your terminal: streamlit run app.py
