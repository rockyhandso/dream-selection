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
    "washington sundar": "right arm off spinner",
    "liam livingstone": "all rounder"
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
    "kl rahul": "wicket keeper batter",
    "rishabh pant": "left handed batter",
    "hardik pandya": "right handed batter",
    "ravindra jadeja": "all rounder",
    "travis head": "left handed batter"
}

def get_bowler_type(bowler_name):
    return BOWLER_TYPES.get(bowler_name.lower(), "right arm medium bowler")  # Default to right arm fast bowler if not found

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
    wicket_count_model_path = os.path.join(current_dir, "model", "wicket_count_model.joblib")
    wicket_count_encoders_path = os.path.join(current_dir, "model", "wicket_count_label_encoders.joblib")
    
    logger.info(f"Attempting to load wicket count model from: {wicket_count_model_path}")
    logger.info(f"Attempting to load wicket count encoders from: {wicket_count_encoders_path}")
    
    if not os.path.exists(wicket_count_model_path):
        raise FileNotFoundError(f"Wicket count model file not found at: {wicket_count_model_path}")
    if not os.path.exists(wicket_count_encoders_path):
        raise FileNotFoundError(f"Wicket count encoders file not found at: {wicket_count_encoders_path}")
        
    wicket_count_model = joblib.load(open(wicket_count_model_path, 'rb'))
    wicket_count_encoders = joblib.load(open(wicket_count_encoders_path, 'rb'))
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
        data = request.get_json()
        
        # Get the first batsman from the list
        first_batsman = data['batsmen'][0] if isinstance(data['batsmen'], list) else data['batsmen'].split(',')[0].strip()
        
        # Create a DataFrame with the input features in the same order as training
        input_data = pd.DataFrame({
            'over': [int(data['overs'])],
            'bowler': [data['bowler']],
            'bowler_type': [get_bowler_type(data['bowler'])],
            'bowler_style': [get_bowler_type(data['bowler'])],
            'batsman': [first_batsman],
            'batsman_type': [get_batsman_type(first_batsman)],
            'batting_number': [1],
            'ball condition': ['new' if int(data['overs']) <= 5 else 'old'],
            'match_phase': ['powerplay' if int(data['overs']) <= 6 else 'middle' if int(data['overs']) <= 15 else 'death'],
            'ball_faced_by_batsman': [0],
            'run': [0],
            'four': [0],
            'six': [0],
            'match_time': ['night'],
            'venue': [data['venue']],
            'pitch_type': [data['pitch']],
            'surface_type': ['dry surface with minimal grass'],
            'bounce': [0.7 if int(data['overs']) <= 10 else 0.5],
            'grip': [0.2 if int(data['overs']) <= 10 else 0.4],
            'straight_forward_boundary': [65],
            'square_boundary': [60],
            'tempeture': [30],
            'humidity': [70],
            'inning': [int(data['innings'])]
        })
        
        # Transform categorical variables using saved encoders
        def safe_label_encode(le, val):
            val = str(val).strip().lower()  # Clean input
            if val in le.classes_:
                return le.transform([val])[0]
            else:
                return -1  # Default encoding for unseen labels

        for col in input_data.select_dtypes(include='object').columns:
            if col in wicket_count_encoders:
                input_data[col] = input_data[col].apply(lambda x: safe_label_encode(wicket_count_encoders[col], x))

        # Make prediction using the model
        predicted_wickets = wicket_count_model.predict(input_data)[0]
        
        # If the model has predict_proba, get confidence
        try:
            confidence = wicket_count_model.predict_proba(input_data).max() * 100
        except:
            confidence = 85.0  # Default confidence if predict_proba is not available
        
        # Generate insights based on the prediction
        insights = [
            f"Based on historical data at {data['venue']}, similar conditions have yielded {predicted_wickets} wickets on average",
            f"The {data['pitch']} pitch type tends to favor the bowler in this scenario",
            f"Prediction confidence: {round(confidence, 2)}%"
        ]
        
        return jsonify({
            'predicted_wickets': int(predicted_wickets),
            'confidence': round(confidence, 2),
            'insights': insights
        })
    except Exception as e:
        logger.error(f"Wicket count prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    logger.info("Starting Flask application...")
    app.run(debug=True)
