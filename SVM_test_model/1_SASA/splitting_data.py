import pandas as pd
from sklearn.model_selection import train_test_split

# Read your dataset
df = pd.read_csv('sasa_data_ns.csv')

# Assuming 'Time' is the time column and 'SASA' is the SASA values
# Define your features and target variable
X = df[['Time']]
y = df['SASA']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the shapes of the resulting sets

print("X_train and y_train are training features and labels")
print("Training set shape (X, y):", X_train.shape, y_train.shape)
print("\n")
print("X_test and y_test are testing features and labels")
print("Testing set shape (X, y):", X_test.shape, y_test.shape)


