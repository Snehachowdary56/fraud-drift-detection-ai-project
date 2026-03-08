import pandas as pd

# Load dataset
data = pd.read_csv("creditcard1.csv")

# Display first few rows
print("Dataset Preview:")
print(data.head())

# Check dataset shape
print("\nDataset Shape:", data.shape)

# Separate features and target variable
X = data.drop(columns=["Class"])
y = data["Class"]

print("\nFeature Matrix Shape:", X.shape)
print("Target Vector Shape:", y.shape)

# Check class distribution
print("\nFraud vs Non-Fraud Distribution:")
print(y.value_counts())