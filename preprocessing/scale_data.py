import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Load processed dataset
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

# Standardization
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nScaled Training Shape:")
print(X_train_scaled.shape)

print("\nScaled Testing Shape:")
print(X_test_scaled.shape)

# Save scaler
joblib.dump(scaler, "saved_models/scaler.pkl")

print("\nScaler saved successfully.")