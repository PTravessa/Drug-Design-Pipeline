import pandas as pd
from data_preparation import DataPreparation
from model_evaluation import ModelEvaluation
from embedding_generator import EmbeddingGenerator
from predictor import Predictor

def main():
    # Step 1: Generate embeddings from ProtTrans
    embedding_generator = EmbeddingGenerator()
    embedding_generator.get_embeddings_from_prottrans(pd.read_csv("peptides_rg.csv").set_index('peptide_id')['peptide_sequence'])

    # Step 2: Read and merge data
    data_prep = DataPreparation()
    merged_df = data_prep.read_and_merge_data("peptides_embs.csv", "peptides_rg.csv")

    # Step 3: Prepare data
    X_scaled, y, scaler = data_prep.prepare_data(merged_df)


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

    #Evaluate models and save the best model and scaler
    ModelEvaluation.main_evaluation(X_scaled, y, param_grids)

    # Step 7: Predict new peptide data
    predictor = Predictor()
    predictions = predictor.predict_new_peptide("1new_peptide.csv")
    print("Predicted Rg values for the new peptide:", predictions)

if __name__ == "__main__":
    main()
