from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from inference_breast import predict_breast
from inference_prostate import predict_prostate

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/predict/<cancer_type>", methods=["POST"])
def predict(cancer_type):
    print(f"📥 Request received for {cancer_type}")
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    print(f"✅ File saved: {file_path}")

    try:
        if cancer_type == "breast":
            result = predict_breast(file_path)
        elif cancer_type == "prostate":
            result = predict_prostate(file_path)
        else:
            return jsonify({"error": "Invalid cancer type"}), 400
        print(f"✅ Prediction result: {result}")
        return jsonify(result)
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"🗑️ File removed: {file_path}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)































