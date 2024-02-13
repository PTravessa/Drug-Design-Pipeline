import sys

print("\nWarning: The model is trained for 145ns it must start at 0.0 timestep in order to prevent misalignment. \n")
sys.stdout.flush()

import pandas as pd #And pyarrow
from sklearn.svm import SVC
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained SVM model
loaded_model = joblib.load('trained_svm_model.pkl')

# Prepare the new data
# Replace 'new_sasa_data_ns.csv' with the actual file containing your new SASA data
df_new_data = pd.read_csv('new_sasa_data_ns.csv')
X_new = df_new_data[['Time']]

# Make predictions on the new data
predictions = loaded_model.predict(X_new)

# Display the predictions
df_new_data['Predicted_Label'] = predictions
print(df_new_data)

# Evaluate the model
y_true = (df_new_data['SASA'] >=78.5).astype(int)

accuracy = accuracy_score(y_true, df_new_data['Predicted_Label'])
precision = precision_score(y_true, df_new_data['Predicted_Label'], zero_division=1)
recall = recall_score(y_true, df_new_data['Predicted_Label'], zero_division=1)
f1 = f1_score(y_true, df_new_data['Predicted_Label'], zero_division=1)

# Confusion Matrix Plot
conf_matrix = confusion_matrix(y_true, df_new_data['Predicted_Label'])
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.savefig('confusion_matrix.png')

# Save metrics to file
metrics_dict = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1-score': f1}
pd.DataFrame.from_dict(metrics_dict, orient='index', columns=['Value']).to_csv('model_evaluation_metrics.csv')

# Display and save visualizations
plt.figure(figsize=(10, 6))
plt.plot(df_new_data['Time'], df_new_data['SASA'], label='Actual SASA')
plt.scatter(df_new_data['Time'], df_new_data['SASA'], c=df_new_data['Predicted_Label'], cmap='viridis', marker='x', label='Predicted Labels')
plt.xlabel('Time (ns)')
plt.ylabel('SASA (nm^2)')
plt.title('Actual SASA vs. Predicted Labels')
plt.legend()
plt.savefig('sasa_prediction_plot.png')

# Display results
print("\n Evaluation Metrics:")
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1-score: {f1:.2f}')

print("\nUse display confusion_matrix.png")

print("Use display sasa_prediction_plot.png")


