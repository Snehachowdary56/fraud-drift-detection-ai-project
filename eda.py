# eda.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv("creditcard1.csv")

# Preview first 5 rows
print("Dataset Preview:")
print(data.head())

# Dataset info
print("\nDataset Info:")
print(data.info())

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Check dataset shape
print("\nDataset Shape:", data.shape)

# Summary statistics
print("\nSummary Statistics:")
print(data.describe())

# Class distribution (fraud vs non-fraud)
print("\nClass Distribution:")
print(data['Class'].value_counts())

# Visualize class imbalance
sns.countplot(x='Class', data=data)
plt.title('Fraud vs Non-Fraud Transactions')
plt.show()

# Correlation heatmap
plt.figure(figsize=(12,8))
sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()