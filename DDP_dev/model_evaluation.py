import pickle
import os
import numpy as np
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

class ModelEvaluation:
    def __init__(self, cv=3):
        self.cv = cv
        self.models = {
            "LinearRegression": linear_model.LinearRegression(),
            "RidgeCV": linear_model.RidgeCV(),
            "LassoCV": linear_model.LassoCV(),
            "RandomForestRegressor": RandomForestRegressor(),
            "SVR": SVR()
        }

    def evaluate_models(self, X_scaled, y):
        results = []
        with open("results.txt", "w") as f:
            for name, model in self.models.items():
                scores = cross_val_score(model, X_scaled, y, cv=self.cv, scoring=rmse_scorer)
                mean_score = -scores.mean()  # convert back to positive RMSE
                std_score = scores.std()
                results.append((name, mean_score, std_score))
                f.write(f"{name}:\n")
                f.write(f"Scores: {scores}\n")
                f.write(f"Mean RMSE: {mean_score:.4f}\n")
                f.write(f"Standard Deviation: {std_score:.4f}\n")
                f.write("="*50 + "\n")
        return results

    def perform_grid_search(self, model, param_grid, X_scaled, y):
        grid_search = GridSearchCV(model, param_grid, cv=self.cv, scoring=rmse_scorer, n_jobs=-1, error_score='raise')
        grid_search.fit(X_scaled, y)
        return grid_search.best_estimator_, -grid_search.best_score_, grid_search.best_params_

    def save_model_and_scaler(self, best_model, best_model_name, scaler):
        model_filename = f"best_model.pkl"
        with open(model_filename, "wb") as model_file:
            pickle.dump(best_model, model_file)
        with open("scaler.pkl", "wb") as scaler_file:
            pickle.dump(scaler, scaler_file)
        return model_filename

    def main_evaluation(X_scaled, y, param_grids):
        # Check if the best model and scaler already exist
        if os.path.exists("best_model.pkl") and os.path.exists("scaler.pkl"):
            print("Best model and scaler already exist. Skipping evaluation.")
            return

        evaluator = ModelEvaluation()
        results = evaluator.evaluate_models(X_scaled, y)

        best_model = None
        best_model_name = None
        best_score = float('inf')
        best_params = None

        with open("results.txt", "a") as f:
            for model_name, param_grid in param_grids.items():
                model = evaluator.models[model_name]
                try:
                    best_estimator, best_rmse, best_params = evaluator.perform_grid_search(model, param_grid, X_scaled, y)
                    f.write(f"{model_name} best parameters: {best_params}\n")
                    f.write(f"{model_name} best score: {best_rmse:.4f}\n")
                    if best_rmse < best_score:
                        best_model = best_estimator
                        best_model_name = model_name
                        best_score = best_rmse
                except Exception as e:
                    f.write(f"Grid search for {model_name} failed: {e}\n")

        if best_model:
            model_filename = evaluator.save_model_and_scaler(best_model, best_model_name, X_scaled)
            with open("results.txt", "a") as f:
                f.write(f"Best model: {best_model_name}\n")
                f.write(f"Best model filename: {model_filename}\n")
            print(f"Best model and scaler saved as {model_filename} and scaler.pkl")
