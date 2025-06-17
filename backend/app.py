from flask import Flask, request, jsonify, send_from_directory
from tensorflow.keras.models import load_model
import numpy as np
import os
from stock_utils import load_data, prepare_sequences, predict_future_days

app = Flask(__name__, static_folder="stock-predictor-ui/build", static_url_path="")

# Load model once
model = load_model("lstm_stock_model.h5")

# Home route serves React
@app.route("/")
def serve_react_index():
    return send_from_directory(app.static_folder, "index.html")

# Serve all other static files (JS, CSS, images)
@app.route("/<path:path>")
def serve_static(path):
    return send_from_directory(app.static_folder, path)

# Prediction API
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    ticker = data.get("ticker")
    n_days = data.get("n_days", 5)
    sequence_len = 60

    try:
        df = load_data(ticker)
        X, y, scaler = prepare_sequences(df, sequence_len)
        last_seq = X[-1]
        predictions = predict_future_days(model, last_seq, n_days, scaler)
        return jsonify({"predictions": [round(p, 2) for p in predictions]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
