from flask import Flask, request, jsonify
import numpy as np
import joblib
import json

app = Flask(__name__)

model = joblib.load("model.joblib")


@app.route("/")
def home():
    return "ML API is running 🚀"


# Health check (SageMaker uses this internally)
@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "ok"}), 200


# Your original API (EC2 / manual testing)
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if "features" not in data:
            return jsonify({"error": "Missing 'features' key"}), 400

        features = data["features"]

        features_array = np.array(features, dtype=float).reshape(1, -1)

        prediction = model.predict(features_array)

        return jsonify({
            "prediction": prediction.tolist()
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


# ⭐ REQUIRED FOR SAGEMAKER ENDPOINT
@app.route("/invocations", methods=["POST"])
def invocations():
    try:
        # SageMaker sends raw body (CSV or JSON)
        data = request.get_data(as_text=True)

        # Try JSON first
        try:
            parsed = json.loads(data)
            features = parsed
        except:
            # fallback CSV format: "3,0,22"
            features = [float(x) for x in data.split(",")]

        features_array = np.array(features, dtype=float).reshape(1, -1)

        prediction = model.predict(features_array)

        return str(prediction[0])

    except Exception as e:
        return str(e), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
