import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Load prediction files
ml_df = pd.read_csv("ensemble/ml_predictions.csv")
bilstm_df = pd.read_csv("ensemble/bilstm_predictions.csv")

# True labels
y_true = ml_df['y_true']

# Weighted Ensemble
final_probs = (
    0.35 * bilstm_df['bilstm_probs'] +
    0.25 * ml_df['rf_probs'] +
    0.20 * ml_df['xgb_probs'] +
    0.20 * ml_df['svm_probs']
)

# Convert probabilities to classes
y_pred = (final_probs > 0.5).astype(int)

# Evaluation
accuracy = accuracy_score(y_true, y_pred)

print(f"\nEnsemble Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_true, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))