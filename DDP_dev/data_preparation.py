import pandas as pd
from sklearn.preprocessing import StandardScaler

class DataPreparation:
    def read_and_merge_data(self, embeddings_file, rg_file):
        embeddings = pd.read_csv(embeddings_file)
        rg_data = pd.read_csv(rg_file)
        rg_data = rg_data[["peptide_id", "mean_Average_Rg"]]
        merged_df = pd.merge(rg_data, embeddings, left_on="peptide_id", right_on="peptide_id", how="inner")
        merged_df.to_csv("MLdata.csv", index=False)
        return merged_df

    def prepare_data(self, merged_df):
        X = merged_df.drop(["peptide_id", "mean_Average_Rg"], axis=1).astype(float)
        y = merged_df["mean_Average_Rg"].astype(float)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled, y, scaler
