import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout

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

# Build BiLSTM Model
model = Sequential()

model.add(
    Bidirectional(
        LSTM(64),
        input_shape=(178, 1)
    )
)

model.add(Dropout(0.3))

model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\nTraining BiLSTM...")

# Train
model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Prediction probabilities
bilstm_probs = model.predict(X_test).flatten()

# Save probabilities
bilstm_df = pd.DataFrame({
    'y_true': y_test,
    'bilstm_probs': bilstm_probs
})

bilstm_df.to_csv(
    "ensemble/bilstm_predictions.csv",
    index=False
)

print("\nBiLSTM prediction probabilities saved successfully.")