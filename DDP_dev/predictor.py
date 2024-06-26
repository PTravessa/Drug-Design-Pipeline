import pandas as pd
import pickle
from embedding_generator import EmbeddingGenerator

class Predictor:
    def predict_new_peptide(self, new_peptide_file):
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
        embedding_generator = EmbeddingGenerator()
        new_data_embeddings = embedding_generator.get_embeddings_from_prottrans(peptide_sequences)

        # Standardize the new data
        new_data_scaled = scaler.transform(new_data_embeddings.drop("peptide_id", axis=1))

        # Predict
        predictions = model.predict(new_data_scaled)
        new_peptide_data['predicted_mean_Average_Rg'] = predictions
        new_peptide_data.to_csv("predicted_peptides.csv", index=False)
        print("Predictions saved to 'predicted_peptides.csv' successfully.")
        return predictions
