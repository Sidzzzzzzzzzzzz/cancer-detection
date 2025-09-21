# 🧬 Cancer Detection (Breast & Prostate MRI)

![Python](https://img.shields.io/badge/Python-3.9-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12-red)
![Flask](https://img.shields.io/badge/Flask-API-lightgrey)
![React](https://img.shields.io/badge/React-Frontend-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-green)

This project demonstrates a **full-stack ML deployment workflow**, focusing on **prostate** and **breast cancer detection** from MRI images using **3D U-Net** and other deep learning architectures.

---

## 🚀 Features
- **Deep Learning Models**
  - 3D U-Net for prostate cancer detection
  - CNN-based model for breast cancer detection
- **Backend**
  - Flask REST API for model inference
- **Frontend**
  - React.js interface for uploading MRI scans and viewing predictions
- **End-to-End Flow**
  - Upload → Preprocess → Predict → Display results

---

## 📂 Project Structure
cancer-detection/
│── backend/ # Flask API code
│ ├── app.py # Main API script
│ ├── models/ # Saved ML models
│ └── utils/ # Preprocessing & helper functions
│
│── frontend/ # React.js frontend
│ ├── src/ # React source code
│ └── public/ # Static assets
│
│── datasets/ # MRI datasets (not included in repo)
│── requirements.txt # Python dependencies
│── README.md # Project documentation

---

## ⚙️ Installation

1️⃣ Clone the Repository

git clone https://github.com/your-username/cancer-detection.git

cd cancer-detection


2️⃣ Backend Setup (Flask + PyTorch)

cd backend

python -m venv .venv

source .venv/bin/activate   # (Linux/Mac)

.venv\Scripts\activate      # (Windows)

pip install -r requirements.txt

Run the Flask server:

python app.py


3️⃣ Frontend Setup (React.js)

cd frontend

npm install

npm start

---

🖥️ Usage

- Start the Flask backend.
- Run the React frontend.
- Upload an MRI scan via the UI.
- Get predictions for cancer presence (positive/negative).

---

📦 Requirements

All Python dependencies are listed in requirements.txt.
Key libraries:

- PyTorch
- Flask
- scikit-learn
- numpy, pandas
- opencv-python
- matplotlib
- React.js (frontend)


