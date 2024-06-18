import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

# Custom function for RMSE
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Create a custom RMSE scorer
rmse_scorer = make_scorer(rmse, greater_is_better=False)
cv = 4

# Read and merge data
embbedings = pd.read_csv("peptides_embs.csv")
rg_data = pd.read_csv("peptides_rg.csv")
rg_data = rg_data[["peptide_id", "mean_Average_Rg"]]
merged_df = pd.merge(rg_data, embbedings, left_on="peptide_id", right_on="peptide_id", how="inner")
merged_df.to_csv("MLdata.csv")

# Prepare features and target
X = merged_df.drop(["peptide_id", "mean_Average_Rg"], axis=1)
X = X.astype(float)
y = merged_df["mean_Average_Rg"].astype(float)

# Standardize the features
#scaler = StandardScaler()
#X_scaled = scaler.fit_transform(X)

# Models
models = {
    "LinearRegression": linear_model.LinearRegression(),
    "RidgeCV": linear_model.RidgeCV(),
    "LassoCV": linear_model.LassoCV(),
    "RandomForestRegressor": RandomForestRegressor(),
    "SVR": SVR()
}

# Evaluate each model
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=cv, scoring=rmse_scorer)
    print(f"{name}:")
    print(scores)
    print(f"Mean RMSE: {scores.mean():.4f}")
