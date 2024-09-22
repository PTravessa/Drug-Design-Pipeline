import pandas as pd
import pickle
import numpy as np
from embedding_generator import EmbeddingGenerator

class Predictor:
    def predict_new_peptide(self, new_peptide_file):
        # Load the scaler and model
        with open("scaler.pkl", "rb") as scaler_file:
            scaler = pickle.load(scaler_file)
        with open("best_model.pkl", "rb") as model_file:
            model = pickle.load(model_file)

        # Verify the type of the scaler
        if not hasattr(scaler, 'transform'):
            raise TypeError("Loaded scaler is not a valid scaler object")

        # Load and preprocess the new peptide data
        new_peptide_data = pd.read_csv(new_peptide_file)

        # Extract peptide sequences
        try:
            peptide_sequences = new_peptide_data.set_index('peptide_id')['peptide_sequence']
        except KeyError as e:
            print(f"Error: {e}. Ensure the CSV file has 'peptide_id' and 'peptide_sequence' columns.")
            return

        # Get embeddings from ProtT5
        embedding_generator = EmbeddingGenerator()
        new_data_embeddings = embedding_generator.get_embeddings_from_prottrans(peptide_sequences, 'new_peptides_embs.csv')

        # Convert embeddings to DataFrame if they are in numpy array format
        if isinstance(new_data_embeddings, np.ndarray):
            new_data_embeddings = pd.DataFrame(new_data_embeddings, columns=[f'emb_{i}' for i in range(new_data_embeddings.shape[1])])
            new_data_embeddings.insert(0, 'peptide_id', peptide_sequences.index)

        # Standardize the new data
        new_data_scaled = scaler.transform(new_data_embeddings.drop("peptide_id", axis=1))

        # Predict
        predictions = model.predict(new_data_scaled)
        new_peptide_data['predicted_mean_Average_Rg'] = predictions
        new_peptide_data.to_csv("predicted_peptides.csv", index=False)
        print("Predictions saved to 'predicted_peptides.csv' successfully.")
        return predictions
