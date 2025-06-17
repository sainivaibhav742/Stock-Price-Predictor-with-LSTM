# 📈 Stock Price Predictor with LSTM

This project predicts future stock prices using a trained LSTM (Long Short-Term Memory) model. It includes:

- 🔮 Frontend: React + Tailwind CSS
- ⚙️ Backend: Flask + TensorFlow (LSTM model)
- 📊 Data: Fetched from Yahoo Finance (via `yfinance`)

---

## 🚀 Features

- Predict stock prices for the next **N days**
- Interactive UI for stock ticker input
- Line plot of predicted values
- Error handling + live feedback
- Model saved as `.h5` file for reuse

---

## 🧠 Technologies Used

- **Frontend**: React, Tailwind CSS
- **Backend**: Flask, TensorFlow, Keras
- **Model**: LSTM
- **Data**: `yfinance`, `pandas`, `scikit-learn`

---

## 🖥️ Running Locally

### 🔧 Backend (Flask)

```bash
cd backend/
pip install -r requirements.txt
python app.py
