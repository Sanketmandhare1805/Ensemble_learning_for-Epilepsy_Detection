import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("data/processed/processed_epilepsy.csv")

# Features and labels
X = df.drop(columns=['y']).values
y = df['y'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape for BiLSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

print("\nBiLSTM Training Shape:")
print(X_train.shape)

print("\nBiLSTM Testing Shape:")
print(X_test.shape)

print("\nSample Input Shape:")
print(X_train[0].shape)