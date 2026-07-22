import pandas as pd

# Load dataset
df = pd.read_csv("data/raw/Epileptic Seizure Recognition.csv")

# Basic information
print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nColumn Names:")
print(df.columns)

print("\nClass Distribution:")
print(df['y'].value_counts())