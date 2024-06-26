import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from transformers import T5Tokenizer, T5EncoderModel
import torch
import re

def main():
    # Step 1: Generate embeddings from ProtTrans
    get_embeddings_from_prottrans(pd.read_csv("peptides_rg.csv").set_index('peptide_id')['peptide_sequence'])
    
    # Step 2: Read and merge data
    merged_df = read_and_merge_data("peptides_embs.csv", "peptides_rg.csv")
    
    # Step 3: Prepare data
    X_scaled, y, scaler = prepare_data(merged_df)
    
    # Step 4: Evaluate models
    results, models = evaluate_models(X_scaled, y)
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
            'alphas': [0.01, 0.1, 1.0, 10.0, 100.0]
        }
    }

    best_model = None
    best_score = float('inf')
    best_params = None

    with open("results.txt", "a") as f:
        for model_name, param_grid in param_grids.items():
            model = models[model_name]
            try:
                best_estimator, best_rmse, best_params = perform_grid_search(model, param_grid, X_scaled, y)
                f.write(f"{model_name} best parameters: {best_params}\n")
                f.write(f"{model_name} best score: {best_rmse:.4f}\n")
                if best_rmse < best_score:
                    best_model = best_estimator
                    best_score = best_rmse
            except Exception as e:
                f.write(f"Grid search for {model_name} failed: {e}\n")
    
    # Step 6: Save the best model and scaler
    save_model_and_scaler(best_model, scaler)
    print("Best model and scaler saved")

    # Step 7: Predict new peptide data
    new_peptide_data_path = "1new_peptide.csv"
    predictions = predict_new_peptide(new_peptide_data_path)
    print("Predicted Rg values for the new peptide:", predictions)

# Custom function for RMSE
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Create a custom RMSE scorer
rmse_scorer = make_scorer(rmse, greater_is_better=False)
cv = 3

def read_and_merge_data(embeddings_file, rg_file):
    embeddings = pd.read_csv(embeddings_file)
    rg_data = pd.read_csv(rg_file)
    rg_data = rg_data[["peptide_id", "mean_Average_Rg"]]
    merged_df = pd.merge(rg_data, embeddings, left_on="peptide_id", right_on="peptide_id", how="inner")
    merged_df.to_csv("MLdata.csv", index=False)
    return merged_df

def prepare_data(merged_df):
    X = merged_df.drop(["peptide_id", "mean_Average_Rg"], axis=1).astype(float)
    y = merged_df["mean_Average_Rg"].astype(float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler

def evaluate_models(X_scaled, y):
    models = {
        "LinearRegression": linear_model.LinearRegression(),
        "RidgeCV": linear_model.RidgeCV(),
        "LassoCV": linear_model.LassoCV(),
        "RandomForestRegressor": RandomForestRegressor(),
        "SVR": SVR()
    }

    results = []
    with open("results.txt", "w") as f:
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
    return results, models

def perform_grid_search(model, param_grid, X_scaled, y):
    grid_search = GridSearchCV(model, param_grid, cv=cv, scoring=rmse_scorer, n_jobs=-1, error_score='raise')
    grid_search.fit(X_scaled, y)
    return grid_search.best_estimator_, -grid_search.best_score_, grid_search.best_params_

def save_model_and_scaler(best_model, scaler):
    with open("best_model.pkl", "wb") as model_file:
        pickle.dump(best_model, model_file)
    with open("scaler.pkl", "wb") as scaler_file:
        pickle.dump(scaler, scaler_file)

def get_embeddings_from_prottrans(peptide_sequences):
    # Set the device to GPU if available, otherwise use CPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load the tokenizer
    tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)

    # Load the model
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc").to(device)

    # Initialize a list to store peptide designations and their embeddings
    peptides_and_embeddings = []

    for peptide_id, peptide_sequence in peptide_sequences.items():
        # Remove newline characters and any unwanted characters, then add spaces between amino acids
        sequence = " ".join(list(re.sub(r"\s+", "", peptide_sequence)))

        # Tokenize the sequence and pad up to the longest sequence in the batch
        ids = tokenizer([sequence], add_special_tokens=True, padding="longest")

        # Convert tokenized sequences to tensors and move to the appropriate device
        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)

        # Generate embeddings
        with torch.no_grad():
            embedding_repr = model(input_ids=input_ids, attention_mask=attention_mask)

        # Calculate per-protein embedding
        emb_per_protein = embedding_repr.last_hidden_state[0,:len(peptide_sequence)].mean(dim=0)  # shape (1 x 1024)
    
        # Convert embedding to a list and prepend the peptide designation
        embedding_list = emb_per_protein.tolist()
        peptides_and_embeddings.append([peptide_id] + embedding_list)

    # Convert the list to a DataFrame
    columns = ['peptide_id'] + [f'emb_{i}' for i in range(1024)]
    df_embs = pd.DataFrame(peptides_and_embeddings, columns=columns)

    # Save the DataFrame to a CSV file
    df_embs.to_csv('peptides_embs.csv', index=False)
    print("Embeddings saved to 'peptides_embs.csv' successfully.")

    return df_embs

def predict_new_peptide(new_peptide_file):
    # Load the scaler and model
    with open("scaler.pkl", "rb") as scaler_file:
        scaler = pickle.load(scaler_file)
    with open("best_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)

    # Load and preprocess the new peptide data
    new_peptide_data = pd.read_csv(new_peptide_file)
    
    # Extract peptide sequences
    peptide_sequences = new_peptide_data.set_index('peptide_id')['peptide_sequence']

    # Get embeddings from ProtTrans
    new_data_embeddings = get_embeddings_from_prottrans(peptide_sequences)

    # Standardize the new data
    new_data_scaled = scaler.transform(new_data_embeddings.drop("peptide_id", axis=1))
    
    # Predict
    predictions = model.predict(new_data_scaled)
    new_peptide_data['predicted_mean_Average_Rg'] = predictions
    new_peptide_data.to_csv("predicted_peptides.csv", index=False)
    print("Predictions saved to 'predicted_peptides.csv' successfully.")
    return predictions

if __name__ == "__main__":
    main()
