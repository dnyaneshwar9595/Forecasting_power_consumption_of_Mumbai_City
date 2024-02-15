
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt

df = pd.read_csv("flattened_power_Humidity_temp.csv")

df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])

df.drop(columns=['date','time'], inplace = True)

df = df[['datetime', 'power', 'temperature', 'humidity']]

cols = list(df)[1:4]

df_train = df[cols].astype(float)

scaler = StandardScaler()
df_trans = scaler.fit_transform(df_train)

len(df_trans)

def create_data(df_trans,window_length,n_future):
  x,y=[],[]
  for i in range(len(df_trans) - window_length - n_future):
    x.append(df_trans[i:i+window_length])
    y.append(df_trans[i+window_length:i+window_length + n_future,0])
  return np.array(x), np.array(y)

window_length = 24
n_future = 2
train_x, train_y = create_data(df_trans, window_length, n_future)

# Define the LSTM model
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(window_length, 3)))
model.add(Dense(train_y.shape[1]))  # Output layer with 24 units
model.compile(optimizer='adam', loss='mse')
model.summary()

# Train the LSTM model
history = model.fit(train_x, train_y, epochs=1, validation_split=0.1)

# plt.plot(history.history['loss'], label = 'Training loss' )
# plt.plot(history.history['val_loss'], label = 'Validation loss')
# plt.legend()
# plt.show()

train_data = df['datetime']



predicted_power = model.predict(train_x[-1:])

# Assuming predicted_power is shaped (24, ) or (1, 24) since you're predicting 24 points.
# Reshape or adjust it to have a shape of (-1, 1) if necessary
predicted_power = predicted_power.reshape(-1, 1)

# Create a dummy array with zeros to match the shape required for inverse_transform
dummy_features = np.zeros((predicted_power.shape[0], df_train.shape[1] - 1))

# Combine the predicted values with dummy features
predicted_combined = np.hstack([predicted_power, dummy_features])

# Apply inverse_transform
predicted_original = scaler.inverse_transform(predicted_combined)

# Extract the power consumption predictions from the transformed array
predicted_power_original = predicted_original[:, 0]  # Assuming power consumption is the first feature

n_futures = train_y.shape[1]
period = pd.date_range(list(train_data)[-1],periods=n_futures, freq = '15T')

# Ensure predicted_power_original is a numpy array or a list
# If it's not already, you can convert it like this:
# predicted_power_original = predicted_power_original.flatten() or .ravel() if it's multi-dimensional

# Create a DataFrame with the period as the index and the predicted values as a column
df_forecast = pd.DataFrame(data=predicted_power_original, index=period, columns=['Predicted Power'])

df_forecast.shape

from tensorflow.keras.models import load_model

# Assume your model is named `model`
model.save('lstm_model.h5')  # Specify the path and filename

import joblib

# Assume your scaler is named `scaler`
joblib.dump(scaler, 'scaler.pkl')  # Specify the path and filename

