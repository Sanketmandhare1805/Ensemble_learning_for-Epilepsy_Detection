import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Load dataset
df = pd.read_csv("data/processed/processed_epilepsy.csv")

# Features and labels
X = df.drop(columns=['y'])
y = df['y']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Scale for SVM
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------- SVM ----------------
svm_model = SVC(
    kernel='rbf',
    probability=True
)

print("\nTraining SVM...")
svm_model.fit(X_train_scaled, y_train)

svm_probs = svm_model.predict_proba(X_test_scaled)[:, 1]

# ---------------- RF ----------------
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

print("\nTraining RF...")
rf_model.fit(X_train, y_train)

rf_probs = rf_model.predict_proba(X_test)[:, 1]

# ---------------- XGB ----------------
xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42
)

print("\nTraining XGB...")
xgb_model.fit(X_train, y_train)

xgb_probs = xgb_model.predict_proba(X_test)[:, 1]

# Save predictions
predictions_df = pd.DataFrame({
    'y_true': y_test.values,
    'svm_probs': svm_probs,
    'rf_probs': rf_probs,
    'xgb_probs': xgb_probs
})

predictions_df.to_csv(
    "ensemble/ml_predictions.csv",
    index=False
)

print("\nML prediction probabilities saved successfully.")