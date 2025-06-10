from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import pickle
from sklearn.preprocessing import LabelEncoder
import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Player type dictionaries
BOWLER_TYPES = {
    # Right arm fast bowlers
    "jasprit bumrah": "right arm fast bowler",
    "mohammed shami": "right arm fast bowler",
    "pat cummins": "right arm fast bowler",
    "josh hazlewood": "right arm fast bowler",
    "mitchell starc": "left arm fast bowler",
    "trent boult": "left arm fast bowler",
    "shaheen afridi": "left arm fast bowler",
    "bhuvneshwar kumar": "right arm medium bowler",
    "deepak chahar": "right arm medium bowler",
    "hardik pandya": "right arm medium bowler",
    "rashid khan": "right arm leg spinner",
    "yuzvendra chahal": "right arm leg spinner",
    "kuldeep yadav": "left arm chinaman",
    "ravindra jadeja": "left arm orthodox",
    "axar patel": "left arm orthodox",
    "ravichandran ashwin": "right arm off spinner",
    "moeen ali": "right arm off spinner",
    "washington sundar": "right arm off spinner"
}

BATSMAN_TYPES = {
    # Right handed batsmen
    "virat kohli": "right handed batter",
    "rohit sharma": "right handed batter",
    "steve smith": "right handed batter",
    "joe root": "right handed batter",
    "kane williamson": "right handed batter",
    "babar azam": "right handed batter",
    "david warner": "left handed batter",
    "quinton de kock": "left handed batter",
    "ben stokes": "left handed batter",
    "shikhar dhawan": "left handed batter",
    "suryakumar yadav": "right handed batter",
    "kl rahul": "right handed batter",
    "rishabh pant": "left handed batter",
    "hardik pandya": "right handed batter",
    "ravindra jadeja": "left handed batter"
}

def get_bowler_type(bowler_name):
    return BOWLER_TYPES.get(bowler_name.lower(), "right arm medium bowler")

def get_batsman_type(batsman_name):
    return BATSMAN_TYPES.get(batsman_name.lower(), "right handed batter")

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

# Load wicket count model and encoders
try:
    wicket_count_model_path = os.path.join(current_dir, "model", "wicket_count_model.pkl")
    wicket_count_encoders_path = os.path.join(current_dir, "model", "wicket_count_label_encoders.pkl")
    
    logger.info(f"Attempting to load wicket count model from: {wicket_count_model_path}")
    logger.info(f"Attempting to load wicket count encoders from: {wicket_count_encoders_path}")
    
    if not os.path.exists(wicket_count_model_path):
        raise FileNotFoundError(f"Wicket count model file not found at: {wicket_count_model_path}")
    if not os.path.exists(wicket_count_encoders_path):
        raise FileNotFoundError(f"Wicket count encoders file not found at: {wicket_count_encoders_path}")
        
    wicket_count_model = pickle.load(open(wicket_count_model_path, 'rb'))
    wicket_count_encoders = pickle.load(open(wicket_count_encoders_path, 'rb'))
    logger.info("Wicket count model and encoders loaded successfully")
except Exception as e:
    logger.error(f"Error loading wicket count model/encoders: {str(e)}")
    wicket_count_model = None
    wicket_count_encoders = None

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
                "bowler_type": get_bowler_type(bowler),
                "batsman": batsman,
                "batsman_type": get_batsman_type(batsman),
                "batting_number": 1,
                "ball condition": "new" if over <= 5 else "old",
                "batsman_run": np.random.randint(0, 7),
                "match_time": "night",
                "venue": venue,
                "pitch_type": pitch,
                "bounce": 0.7 if over <= 10 else 0.5,
                "grip": 0.2 if over <= 10 else 0.4,
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
                    "estimated_runs": row["batsman_run"],
                    "bowler_type": get_bowler_type(bowler),
                    "batsman_type": get_batsman_type(batsman)
                })

        logger.info(f"Prediction: {batsman} will remain not out")
        return jsonify({
            "batsman": batsman, 
            "status": "Not out",
            "batsman_type": get_batsman_type(batsman)
        })

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/wicket-count")
def wicket_count():
    return render_template("wicket_count.html")

@app.route("/predict-wickets", methods=["POST"])
def predict_wickets():
    if wicket_count_model is None or wicket_count_encoders is None:
        logger.error("Wicket count model or encoders not loaded. Cannot make predictions.")
        return render_template("wicket_count.html", error="Model not loaded. Please check server logs.")

    try:
        # Get form data
        bowler = request.form.get("bowler")
        opposition = request.form.get("opposition")
        venue = request.form.get("venue")
        overs = float(request.form.get("overs"))

        # Create input data
        input_data = pd.DataFrame({
            'bowler': [bowler],
            'opposition': [opposition],
            'venue': [venue],
            'overs': [overs]
        })

        # Transform categorical variables using saved encoders
        for col in input_data.select_dtypes(include=['object']).columns:
            if col in wicket_count_encoders:
                input_data[col] = wicket_count_encoders[col].transform(input_data[col])

        # Make prediction
        prediction = wicket_count_model.predict(input_data)[0]

        return render_template("wicket_count.html", prediction=prediction)

    except Exception as e:
        logger.error(f"Wicket count prediction error: {str(e)}")
        return render_template("wicket_count.html", error=str(e))

if __name__ == "__main__":
    logger.info("Starting Flask application...")
    app.run(debug=True)
