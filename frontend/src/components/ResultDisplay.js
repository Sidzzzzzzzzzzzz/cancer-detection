import React from "react";
import "./ResultDisplay.css";

function ResultDisplay({ result }) {
  if (!result) return null;

  return (
    <div className="result-card">
      <h2>Prediction Result</h2>

      <div className="result-item">
        <span className="label">Prediction:</span>
        <span className={`value ${result.prediction.toLowerCase()}`}>
          {result.prediction}
        </span>
      </div>

      <div className="result-item">
        <span className="label">Positive %:</span>
        <span className="value">{result.positive_pct}%</span>
      </div>

      {result.shape && (
        <div className="result-item">
          <span className="label">Input Shape:</span>
          <span className="value">[{result.shape.join(", ")}]</span>
        </div>
      )}

      {result.summary && (
        <div className="summary-box">
          <h3>Summary:</h3>
          <p>{result.summary}</p>
        </div>
      )}
    </div>
  );
}

export default ResultDisplay;

