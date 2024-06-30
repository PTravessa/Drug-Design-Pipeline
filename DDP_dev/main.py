import os
import pandas as pd
from data_preparation import DataPreparation
from model_evaluation import ModelEvaluation
from embedding_generator import EmbeddingGenerator
from predictor import Predictor

def main():
    # Step 1: Generate embeddings from ProtTrans for training
    embedding_generator = EmbeddingGenerator()
    embedding_generator.get_embeddings_from_prottrans(
        pd.read_csv("peptides_rg.csv").set_index('peptide_id')['peptide_sequence']
    )
    
    # Step 2: Read and merge data
    merged_df = DataPreparation.read_and_merge_data("peptides_embs.csv", "peptides_rg.csv")
    
    # Step 3: Prepare data
    X_scaled, y, scaler = DataPreparation.prepare_data(merged_df)
    
    # Step 4: Evaluate models and save the best model and scaler
    model_evaluation = ModelEvaluation()
    results = model_evaluation.evaluate_models(X_scaled, y)
    print("Model evaluation results saved to results.txt")

    # Step 5: Hyperparameter tuning and model selection
    param_grids = {
        "RandomForestRegressor": {
            'n_estimators': [100, 200],
            'max_features': [None, 'sqrt', 'log2'],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        "SVR": {
            'C': [0.1, 1, 10, 100],
            'epsilon': [0.01, 0.1, 1],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
        },
        "RidgeCV": {
            'alphas': [(0.1, 1.0, 10.0), (0.01, 0.1, 1.0, 10.0, 100.0)]
        },
        "LassoCV": {
            'alphas': [(0.01, 0.1, 1.0, 10.0, 100.0)]
        }
    }

    best_model = None
    best_score = float('inf')
    best_model_name = None

    with open("results.txt", "a") as f:
        for model_name, param_grid in param_grids.items():
            model = model_evaluation.models[model_name]
            try:
                best_estimator, best_rmse, best_params = model_evaluation.perform_grid_search(model, param_grid, X_scaled, y)
                f.write(f"{model_name} best parameters: {best_params}\n")
                f.write(f"{model_name} best score: {best_rmse:.4f}\n")
                if best_rmse < best_score:
                    best_model = best_estimator
                    best_score = best_rmse
                    best_model_name = model_name
            except Exception as e:
                f.write(f"Grid search for {model_name} failed: {e}\n")

    # Plot predictions for the best model
    model_evaluation.plot_predictions(best_model, X_scaled, y, f'predictions_vs_actual_{best_model_name}.png')
    
    # Step 6: Save the best model and scaler
    model_evaluation.save_model_and_scaler(best_model, scaler)
    print("Best model and scaler saved")

    # Step 7: Predict new peptide data
    new_peptide_data_path = "new_peptide.csv"
    if os.path.exists(new_peptide_data_path):
        predictor = Predictor()
        predictions = predictor.predict_new_peptide(new_peptide_data_path)
        print("Predicted Rg values for the new peptide:", predictions)
    else:
        print(f"New peptide data file '{new_peptide_data_path}' does not exist.")

if __name__ == "__main__":
    main()
