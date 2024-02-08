import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Read your dataset
df = pd.read_csv('sasa_data_ns.csv')

# Assuming 'Time' is the time column and 'SASA' is the SASA values
# Define your features and target variable
X = df[['Time']]
y = (df['SASA'] > 78.5).astype(int)  # Creating a binary label: 1 if SASA > 78.5, 0 otherwise

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("X_train and y_train are training features and labels")
print("Training set shape (X, y):", X_train.shape, y_train.shape)
print("\n")
print("X_test and y_test are testing features and labels")
print("Testing set shape (X, y):", X_test.shape, y_test.shape)
print("\n")

# Create and train the SVM model with verbose set to 2 for detailed output
model = SVC(kernel='linear', verbose=2)  # You can also try 'rbf' (radial basis function) or other kernels
model.fit(X_train, y_train)

# Save the trained model to a file
joblib.dump(model, 'trained_svm_model.pkl')

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Display the results
print(f'Accuracy: {accuracy:.2f}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)