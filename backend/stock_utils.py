import numpy as np
import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_data(ticker, start="2015-01-01", end="2025-01-01"):
    df = yf.download(ticker, start=start, end=end)
    df.reset_index(inplace=True)
    df = df.dropna(subset=["Close"])
    return df

def prepare_sequences(data, sequence_length):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[['Close']])
    
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i])
        y.append(scaled_data[i])
    
    return np.array(X), np.array(y), scaler

def predict_future_days(model, last_sequence, n_days, scaler):
    future_predictions = []
    current_input = last_sequence

    for _ in range(n_days):
        pred = model.predict(current_input[np.newaxis, :, :])[0][0]
        future_predictions.append(pred)
        current_input = np.append(current_input[1:], [[pred]], axis=0)

    return scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten()
