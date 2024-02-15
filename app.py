from sklearn.preprocessing import StandardScaler
import pandas as pd 


df = pd.read_csv("flattened_power_Humidity_temp.csv")
df.drop(columns=['date','time'], inplace = True)
scaler = StandardScaler()
df_trans = scaler.fit_transform(df)
# Extract the last N observations where N is the number of timesteps
# Assuming your LSTM expects 24 timesteps and your data has 3 features
last_observations = df_trans[-24:].reshape(1, 24, 3)
