from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load model and encoders
try:
    # Get the absolute path to the model file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "model", "wicket_model_with_batsman.joblib")
    encoders_path = os.path.join(current_dir, "model", "label_encoders.joblib")
    
    logger.info(f"Attempting to load model from: {model_path}")
    logger.info(f"Attempting to load encoders from: {encoders_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    if not os.path.exists(encoders_path):
        raise FileNotFoundError(f"Encoders file not found at: {encoders_path}")
        
    model = joblib.load(model_path)
    label_encoders = joblib.load(encoders_path)
    logger.info("Model and encoders loaded successfully")
except FileNotFoundError as e:
    logger.error(f"File not found error: {str(e)}")
    model = None
    label_encoders = None
except Exception as e:
    logger.error(f"Error loading model/encoders: {str(e)}")
    logger.error(f"Error type: {type(e).__name__}")
    model = None
    label_encoders = None

# Define the prediction route
@app.route("/predict", methods=["POST"])
def predict():
    if model is None or label_encoders is None:
        logger.error("Model or encoders not loaded. Cannot make predictions.")
        return jsonify({"error": "Model not loaded. Please check server logs."}), 500

    try:
        data = request.get_json()
        if not data:
            logger.warning("No data provided in request")
            return jsonify({"error": "No data provided"}), 400

        # Validate required fields
        required_fields = ["batsman", "bowlers", "venue", "pitch", "innings"]
        for field in required_fields:
            if field not in data:
                logger.warning(f"Missing required field: {field}")
                return jsonify({"error": f"Missing required field: {field}"}), 400

        batsman = data.get("batsman").lower()
        bowlers = [b.lower() for b in data.get("bowlers", [])]
        venue = data.get("venue").lower()
        pitch = data.get("pitch").lower()
        try:
            innings = int(data.get("innings"))
            if innings not in [1, 2]:
                logger.warning(f"Invalid innings value: {innings}")
                return jsonify({"error": "innings must be either 1 or 2"}), 400
        except ValueError:
            logger.warning("Invalid innings value (not a number)")
            return jsonify({"error": "innings must be a number"}), 400

        logger.info(f"Making prediction for batsman: {batsman}, venue: {venue}, pitch: {pitch}, innings: {innings}")
        logger.info(f"Bowlers: {bowlers}")

        # Simulate over-wise prediction
        for over in range(1, 21):
            bowler = bowlers[over % len(bowlers)]
            row = {
                "over": over,
                "bowler": bowler,
                "bowler_type": "right arm medium bowler",  # can be improved
                "batsman": batsman,
                "batsman_type": "left handed batter",
                "batting_number": 1,
                "ball condition": "new",
                "batsman_run": np.random.randint(0, 7),
                "match_time": "night",
                "venue": venue,
                "pitch_type": pitch,
                "bounce": 0.7,
                "grip": 0.2,
                "inning": innings
            }

            df = pd.DataFrame([row])
            # Use saved label encoders
            for col in df.select_dtypes(include='object').columns:
                if col in label_encoders:
                    df[col] = label_encoders[col].transform(df[col].astype(str))

            prediction = model.predict(df)[0]
            if prediction == 1:
                logger.info(f"Prediction: {batsman} will be dismissed in over {over} by {bowler}")
                return jsonify({
                    "batsman": batsman,
                    "dismissed_by": bowler,
                    "over": over,
                    "estimated_runs": row["batsman_run"]
                })

        logger.info(f"Prediction: {batsman} will remain not out")
        return jsonify({"batsman": batsman, "status": "Not out"})

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    logger.info("Starting Flask application...")
    app.run(debug=True)
