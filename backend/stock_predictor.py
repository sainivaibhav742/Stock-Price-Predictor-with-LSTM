import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.models import load_model

# CONFIG
STOCK_SYMBOL = "AAPL"
DATA_PATH = f"{STOCK_SYMBOL}.csv"
MODEL_PATH = "lstm_stock_model.h5"
LOOK_BACK = 60
FUTURE_DAYS = 5

# STEP 1: Load or Download Data
if not os.path.exists(DATA_PATH):
    print("⬇️ Downloading stock data...")
    df = yf.download(STOCK_SYMBOL, start="2015-01-01", end="2025-01-01")
    df.reset_index(inplace=True)
    df.to_csv(DATA_PATH, index=False)
else:
    print("📄 Using existing stock data...")
    df = pd.read_csv(DATA_PATH, parse_dates=['Date'])

# STEP 2: Preprocessing
df = df.dropna(subset=['Date', 'Close'])
df = df.sort_values(by='Date')

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['Close']])

X, y = [], []
for i in range(LOOK_BACK, len(scaled_data)):
    X.append(scaled_data[i - LOOK_BACK:i])
    y.append(scaled_data[i])
X, y = np.array(X), np.array(y)

# STEP 3: Model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
    LSTM(50),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')

print("🚀 Training LSTM model...")
model.fit(X, y, epochs=10, batch_size=32)

# STEP 4: Save model
model.save(MODEL_PATH)
print(f"✅ Model saved as {MODEL_PATH}")

# STEP 5: Predict Next N Days
def predict_future_days(model, last_sequence, n_days, scaler):
    future_predictions = []
    current_input = last_sequence
    for _ in range(n_days):
        pred = model.predict(current_input[np.newaxis, :, :])[0][0]
        future_predictions.append(pred)
        current_input = np.append(current_input[1:], [[pred]], axis=0)
    return scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

last_sequence = scaled_data[-LOOK_BACK:]
future = predict_future_days(model, last_sequence, FUTURE_DAYS, scaler)

# STEP 6: Plot
plt.figure(figsize=(10, 6))
plt.plot(df['Date'][-200:], df['Close'][-200:], label="Real Price")
future_dates = pd.date_range(start=df['Date'].iloc[-1], periods=FUTURE_DAYS+1)[1:]
plt.plot(future_dates, future, label="Predicted", marker='o')
plt.title("Stock Price Prediction 📈")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.tight_layout()
plt.show()

print(f"📈 Predicted next {FUTURE_DAYS} days close prices:")
for date, price in zip(future_dates, future.flatten()):
    print(f"{date.date()}: ${price:.2f}")
