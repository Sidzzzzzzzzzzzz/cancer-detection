import React, { useState } from "react";
import axios from "axios";

function App() {
  const [cancerType, setCancerType] = useState("breast");
  const [file, setFile] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const BACKEND_BASE = "http://127.0.0.1:5000";

  const onChoose = (e) => {
    setFile(e.target.files[0]);
    setResult(null);
    setError(null);
  };

  const onPredict = async () => {
    if (!file) {
      setError("Please select a file first");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    setAnalyzing(true);
    setError(null);
    setResult(null);

    try {
      const response = await axios.post(
        `${BACKEND_BASE}/predict/${cancerType}`,
        formData,
        { headers: { "Content-Type": "multipart/form-data" } }
      );
      setResult(response.data);
    } catch (err) {
      setError(err.response?.data?.error || "Prediction failed");
    } finally {
      setAnalyzing(false);
    }
  };

  return (
    <div style={{ textAlign: "center", padding: "40px", color: "white", background: "#121212", minHeight: "100vh" }}>
      <h2>Cancer Detection</h2>
      <p>Upload an MRI/image (.npy or .mhd) to get predictions</p>

      <div style={{ marginBottom: "20px" }}>
        <button
          onClick={() => setCancerType("breast")}
          style={{
            margin: "5px",
            padding: "10px 20px",
            background: cancerType === "breast" ? "#007bff" : "#333",
            color: "white",
            border: "none",
            borderRadius: "5px",
            cursor: "pointer"
          }}
        >
          Breast Cancer
        </button>
        <button
          onClick={() => setCancerType("prostate")}
          style={{
            margin: "5px",
            padding: "10px 20px",
            background: cancerType === "prostate" ? "#007bff" : "#333",
            color: "white",
            border: "none",
            borderRadius: "5px",
            cursor: "pointer"
          }}
        >
          Prostate Cancer
        </button>
      </div>

      <input type="file" onChange={onChoose} />

      <div style={{ marginTop: "20px" }}>
        <button
          onClick={onPredict}
          disabled={analyzing}
          style={{
            padding: "10px 20px",
            background: "#28a745",
            color: "white",
            border: "none",
            borderRadius: "5px",
            cursor: "pointer"
          }}
        >
          {analyzing ? "Analyzing..." : "Analyze"}
        </button>
      </div>

      {error && <p style={{ color: "red" }}>{error}</p>}
      {result && (
        <div style={{ marginTop: "20px", textAlign: "left", display: "inline-block", background: "#222", padding: "15px", borderRadius: "8px" }}>
          <h3>Result:</h3>
          <pre style={{ color: "lightgreen" }}>{JSON.stringify(result, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}

export default App;


















