import pandas as pd

# Load dataset
df = pd.read_csv("data/raw/Epileptic Seizure Recognition.csv")

# Remove unnecessary ID column
df.drop(columns=["Unnamed"], inplace=True)

# Convert to binary classification
# Class 1 = seizure
# Classes 2,3,4,5 = non-seizure

df['y'] = df['y'].apply(lambda x: 1 if x == 1 else 0)

print("\nUpdated Class Distribution:")
print(df['y'].value_counts())

print("\nDataset Shape:")
print(df.shape)

# Save processed dataset
df.to_csv("data/processed/processed_epilepsy.csv", index=False)

print("\nProcessed dataset saved successfully.")