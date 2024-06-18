import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

# Custom function for RMSE
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Create a custom RMSE scorer
rmse_scorer = make_scorer(rmse, greater_is_better=False)
cv = 3

# Read and merge data
embeddings = pd.read_csv("peptides_embs.csv")
rg_data = pd.read_csv("peptides_rg.csv")
rg_data = rg_data[["peptide_id", "mean_Average_Rg"]]
merged_df = pd.merge(rg_data, embeddings, left_on="peptide_id", right_on="peptide_id", how="inner")

# Save merged data for inspection
merged_df.to_csv("MLdata.csv", index=False)

# Prepare features and target
X = merged_df.drop(["peptide_id", "mean_Average_Rg"], axis=1).astype(float)
y = merged_df["mean_Average_Rg"].astype(float)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Models
models = {
    "LinearRegression": linear_model.LinearRegression(),
    "RidgeCV": linear_model.RidgeCV(),
    "LassoCV": linear_model.LassoCV(),
    "RandomForestRegressor": RandomForestRegressor(),
    "SVR": SVR()
}

# Open a file to write results
with open("results.txt", "w") as f:
    # Evaluate each model
    results = []
    for name, model in models.items():
        scores = cross_val_score(model, X_scaled, y, cv=cv, scoring=rmse_scorer)
        mean_score = -scores.mean()  # convert back to positive RMSE
        std_score = scores.std()
        results.append((name, mean_score, std_score))
        f.write(f"{name}:\n")
        f.write(f"Scores: {scores}\n")
        f.write(f"Mean RMSE: {mean_score:.4f}\n")
        f.write(f"Standard Deviation: {std_score:.4f}\n")
        f.write("="*50 + "\n")

    # Hyperparameter tuning for RandomForest, SVR, RidgeCV, and LassoCV
    param_grid_rf = {
        'n_estimators': [100, 200],
        'max_features': [None, 'sqrt', 'log2'],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    param_grid_svr = {
        'C': [0.1, 1, 10, 100],
        'epsilon': [0.01, 0.1, 1],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
    }

    param_grid_ridge = {
        'alphas': [(0.1, 1.0, 10.0), (0.01, 0.1, 1.0, 10.0, 100.0)]
    }

    param_grid_lasso = {
        'alphas': [None]
    }

    # Grid search for RandomForestRegressor
    grid_search_rf = GridSearchCV(RandomForestRegressor(), param_grid_rf, cv=cv, scoring=rmse_scorer, n_jobs=-1, error_score='raise')
    try:
        grid_search_rf.fit(X_scaled, y)
        f.write("RandomForestRegressor best parameters: {}\n".format(grid_search_rf.best_params_))
        f.write("RandomForestRegressor best score: {:.4f}\n".format(-grid_search_rf.best_score_))
    except Exception as e:
        f.write(f"Grid search for RandomForestRegressor failed: {e}\n")

    # Grid search for SVR
    grid_search_svr = GridSearchCV(SVR(), param_grid_svr, cv=cv, scoring=rmse_scorer, n_jobs=-1, error_score='raise')
    try:
        grid_search_svr.fit(X_scaled, y)
        f.write("SVR best parameters: {}\n".format(grid_search_svr.best_params_))
        f.write("SVR best score: {:.4f}\n".format(-grid_search_svr.best_score_))
    except Exception as e:
        f.write(f"Grid search for SVR failed: {e}\n")

    # Grid search for RidgeCV
    grid_search_ridge = GridSearchCV(linear_model.RidgeCV(), param_grid_ridge, cv=cv, scoring=rmse_scorer, n_jobs=-1, error_score='raise')
    try:
        grid_search_ridge.fit(X_scaled, y)
        f.write("RidgeCV best parameters: {}\n".format(grid_search_ridge.best_params_))
        f.write("RidgeCV best score: {:.4f}\n".format(-grid_search_ridge.best_score_))
    except Exception as e:
        f.write(f"Grid search for RidgeCV failed: {e}\n")

    # Grid search for LassoCV
    grid_search_lasso = GridSearchCV(linear_model.LassoCV(), param_grid_lasso, cv=cv, scoring=rmse_scorer, n_jobs=-1, error_score='raise')
    try:
        grid_search_lasso.fit(X_scaled, y)
        f.write("LassoCV best parameters: {}\n".format(grid_search_lasso.best_params_))
        f.write("LassoCV best score: {:.4f}\n".format(-grid_search_lasso.best_score_))
    except Exception as e:
        f.write(f"Grid search for LassoCV failed: {e}\n")
