# SASA Prediction with SVM

## 1st Try of the SVM Prediction
This model aimed to predict Solvent Accessible Surface Area (SASA) using a Support Vector Machine (SVM) model trained on SASA and time data obtained from molecular simulations. 

## Model Training
- Trained a SVM model using SASA and time as features.
- Data processing steps included loading the dataset, splitting into training and testing sets, and scaling the features.(*As described by the necessary documentation*)
- Explored different SVM kernels and hyperparameters to optimize model performance.

## Model Evaluation
- Evaluated the trained SVM model using performance metrics such as accuracy, precision, recall, and F1-score.
- Tested the model with **the same** SASA data and analyzed the predictions.
- Identified challenges related to using time as a feature and potential implications for model accuracy.(**Wrong**)
- Limited predictive power observed when using time as a feature for SASA prediction.
