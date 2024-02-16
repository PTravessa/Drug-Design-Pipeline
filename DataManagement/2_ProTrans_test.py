from transformers import T5Tokenizer, T5EncoderModel
import torch
import re
import pandas as pd

# Set the device to GPU if available, otherwise use CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load the tokenizer
tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)

# Load the model
model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc").to(device)

# Initialize a list to store peptide designations and their embeddings
peptides_and_embeddings = []

# Read peptide sequences from the file and process each sequence
with open('peptides.txt', 'r') as file:
    for line in file:
        # Remove newline characters and any unwanted characters, then add spaces between amino acids
        sequence = " ".join(list(re.sub(r"\s+", "", line.strip())))
        print(sequence) #Debug purpose
        peptide_designation = re.sub(r"\s+", "", line.strip())  # Cleaned peptide designation

        # Tokenize the sequence and pad up to the longest sequence in the batch
        ids = tokenizer([sequence], add_special_tokens=True, padding="longest")

        # Convert tokenized sequences to tensors and move to the appropriate device
        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)

        # Generate embeddings
        with torch.no_grad():
            embedding_repr = model(input_ids=input_ids, attention_mask=attention_mask)

        # Calculate per-protein embedding
        emb_per_protein = embedding_repr.last_hidden_state.mean(dim=1)  # shape (1 x 1024)

        # Convert embedding to a list and prepend the peptide designation
        embedding_list = emb_per_protein.cpu().numpy().flatten().tolist()
        peptides_and_embeddings.append([peptide_designation] + embedding_list)

# Convert the list to a DataFrame
columns = ['Peptide Designation'] + [f'emb_{i}' for i in range(1024)]
df = pd.DataFrame(peptides_and_embeddings, columns=columns)

# Save the DataFrame to a CSV file
df.to_csv('peptides_embs.csv', index=False)
print("Embeddings saved to 'peptides_embs.csv' successfully. \N{check mark}")
