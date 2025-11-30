from flask import Flask, request, jsonify
import joblib
import os
import pandas as pd
from .config import config
from .logging_utils import logger
from .monitoring import monitored
from .train_model import train_and_evaluate

app = Flask(__name__)

def load_model():
    if not os.path.exists(config.model_path):
        logger.info("Model file not found, training a new model...")
        train_and_evaluate()
    logger.info(f"Loading model from {config.model_path}")
    return joblib.load(config.model_path)

model = load_model()

@app.route("/health", methods=["GET"])
@monitored("health")
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/predict", methods=["POST"])
@monitored("predict")
def predict():
    payload = request.get_json(force=True)
    if "customers" not in payload or not isinstance(payload["customers"], list):
        return jsonify({"error": "Payload must contain a 'customers' list"}), 400

    df = pd.DataFrame(payload["customers"])
    required = ["country", "tenure", "monthly_charges", "num_streams"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        return jsonify({"error": f"Missing columns: {missing_cols}"}), 400

    preds = model.predict_proba(df[required])[:, 1]
    results = [{"churn_probability": float(p)} for p in preds]
    avg_prob = float(preds.mean())
    return jsonify({"predictions": results, "average_churn_probability": avg_prob}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
