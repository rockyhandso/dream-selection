import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Step 1: Load data
file_path = 'bowler wise performance.csv'  # Apni file ka path yahan dalein
df = pd.read_csv(file_path)

# Step 2: Preprocessing
df = df.drop(columns=['date'])  # Date column drop kar rahe hain
df.columns = df.columns.str.strip()  # Column names ke aage peeche spaces hata rahe hain
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].str.strip()  # String columns ke values ke aage peeche spaces hata rahe hain

# Step 3: Encode categorical columns
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Step 4: Features aur target define karna
X = df.drop(columns=['dismissal'])
y = df['dismissal']

# Step 5: Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Random Forest model train karna
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Step 7: Model evaluate karna
y_pred = rf_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 8: Model ko save karna (pickle file)
with open('model/wicket_count_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

# Optional: Label encoders ko bhi save karna
with open('model/wicket_count_label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

print("Model aur label encoders pickle files me save kar diye gaye hain.")
