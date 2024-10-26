import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import cross_val_score,cross_val_predict, GridSearchCV
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

    def save_model_and_scaler(self, best_model, scaler):
        with open("best_model.pkl", "wb") as model_file:
            pickle.dump(best_model, model_file)
        with open("scaler.pkl", "wb") as scaler_file:
            pickle.dump(scaler, scaler_file)
        with open("results.txt", "a") as f:
            f.write(f"\nBest model saved as: best_model.pkl\n")
            f.write("Scaler saved as: scaler.pkl\n")
            
    def plot_predictions(self, model, X_scaled, y, filename='predictions_plot.png'):
        predictions = cross_val_predict(model, X_scaled, y, cv=self.cv)
        plt.figure(figsize=(8, 6))
        plt.scatter(y, predictions, alpha=0.5)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'Cross-Validated Predictions - Model: {filename.split("_")[-1][:-4]}')
        plt.savefig(filename)
        plt.close()
        print(f"Plot saved as '{filename}'.(Data Visualization Purposes)")