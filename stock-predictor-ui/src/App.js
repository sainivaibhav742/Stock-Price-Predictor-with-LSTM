import { useState } from 'react';
import './App.css';

function App() {
  const [ticker, setTicker] = useState('AAPL');
  const [days, setDays] = useState(5);
  const [result, setResult] = useState([]);
  const [loading, setLoading] = useState(false);

  const handlePredict = async () => {
    setLoading(true);
    try {
      const res = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ticker, days }),
      });

      const data = await res.json();
      if (data.predictions) {
        setResult(data.predictions);
      } else {
        alert(data.error || 'Prediction failed');
      }
    } catch (error) {
      alert('Error connecting to backend');
    }
    setLoading(false);
  };

  return (
    <div className="App">
      <h1>📈 Stock Price Predictor</h1>
      <input value={ticker} onChange={e => setTicker(e.target.value)} placeholder="Stock Ticker (e.g., AAPL)" />
      <input type="number" value={days} onChange={e => setDays(e.target.value)} min={1} max={30} />
      <button onClick={handlePredict}>{loading ? "Predicting..." : "Predict"}</button>
      {result.length > 0 && (
        <div>
          <h3>Predicted Prices</h3>
          <ul>
            {result.map((price, i) => <li key={i}>Day {i + 1}: ${price}</li>)}
          </ul>
        </div>
      )}
    </div>
  );
}

export default App;
