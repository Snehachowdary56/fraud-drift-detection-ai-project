import pandas as pd

# Load the CSV file
data = pd.read_csv("data/creditcard1.csv")

# If you want X (features) and y (labels), assuming 'Class' is the label column
X = data.drop(columns=['Class'])
y = data['Class']

print(X.shape, y.shape)