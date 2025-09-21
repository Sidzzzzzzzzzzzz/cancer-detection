import React, { useState } from "react";

function UploadForm() {
  const [file, setFile] = useState(null);
  const [cancerType, setCancerType] = useState("breast");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const handleUpload = async () => {
    if (!file) {
      alert("Please select a file first!");
      return;
    }
    setLoading(true);
    setResult(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch(`http://127.0.0.1:5000/predict/${cancerType}`, {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      setResult(data);
    } catch (err) {
      alert("Error while uploading file.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <div>
        <label>
          <input
            type="radio"
            value="breast"
            checked={cancerType === "breast"}
            onChange={() => setCancerType("breast")}
          />
          Breast Cancer
        </label>
        <label>
          <input
            type="radio"
            value="prostate"
            checked={cancerType === "prostate"}
            onChange={() => setCancerType("prostate")}
          />
          Prostate Cancer
        </label>
      </div>

      <input type="file" onChange={(e) => setFile(e.target.files[0])} />
      <br />
      <button onClick={handleUpload} disabled={loading}>
        {loading ? "Processing..." : "Upload & Predict"}
      </button>

      {result && (
        <div className="result-box">
          <h2>Result:</h2>
          <pre>{JSON.stringify(result, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}

export default UploadForm;

