from flask import Flask, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load("model.joblib")


@app.route("/")
def home():
    return "ML API is running 🚀"


@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "ok"}), 200


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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
