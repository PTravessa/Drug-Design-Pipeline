import pandas as pd
import numpy as np

# Load the CSV file
df = pd.read_csv('combined_descriptors.csv')
print("CSV file loaded successfully.")

# Load the embedding array from the text file
embedding_file = f'{df["Peptide Designation"].iloc[0]}_embedding.txt'
embedding = np.loadtxt(embedding_file)
print("Embedding array loaded successfully.")

# Repeat the embedding for each row in the DataFrame
repeated_embedding = np.tile(embedding, (df.shape[0], 1))
print("Embedding repeated successfully.")

# Convert the repeated embedding to a DataFrame
embedding_df = pd.DataFrame(repeated_embedding, columns=[f'emb_{i}' for i in range(embedding.shape[0])], index=df.index)
print("Conversion to DataFrame successful.")

# Concatenate the embedding DataFrame with the original DataFrame
combined_df = pd.concat([df, embedding_df], axis=1)
print("DataFrames concatenated successfully.")

# Write the combined DataFrame to a file
combined_df.to_csv('combined_descriptors_with_embeddings.csv', index=False)

print("\nCombined DataFrame written to file. \N{check mark}")
print(combined_df.head(4))
