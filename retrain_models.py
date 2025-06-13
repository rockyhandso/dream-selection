import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def create_model_directory():
    if not os.path.exists('model'):
        os.makedirs('model')

def train_wicket_model():
    print("Training wicket model...")
    # Load data
    df = pd.read_csv('bowler wise performance.csv')
    
    # Preprocessing
    df = df.drop(columns=['date'])
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip()

    # Encode categorical columns
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Features and target
    X = df.drop(columns=['dismissal'])
    y = df['dismissal']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)

    # Evaluate
    y_pred = rf_model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Save model and encoders
    joblib.dump(rf_model, 'model/wicket_model_with_batsman.joblib')
    joblib.dump(label_encoders, 'model/label_encoders.joblib')
    print("Wicket model and encoders saved successfully!")

def train_wicket_count_model():
    print("\nTraining wicket count model...")
    # Load data
    df = pd.read_csv('bowler wise performance.csv')
    
    # Preprocessing
    df = df.drop(columns=['date'])
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip()

    # Encode categorical columns
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Features and target (using 'dismissal' as target)
    X = df.drop(columns=['dismissal'])
    y = df['dismissal']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)

    # Evaluate
    y_pred = rf_model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Save model and encoders
    joblib.dump(rf_model, 'model/wicket_count_model.joblib')
    joblib.dump(label_encoders, 'model/wicket_count_label_encoders.joblib')
    print("Wicket count model and encoders saved successfully!")

if __name__ == "__main__":
    create_model_directory()
    train_wicket_model()
    train_wicket_count_model()
    print("\nAll models have been retrained with scikit-learn 1.4.0!") 