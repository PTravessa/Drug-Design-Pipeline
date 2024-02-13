from transformers import T5Tokenizer, T5EncoderModel
import torch
import re

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load the tokenizer
tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)

# Load the model
model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc").to(device)

sequence_examples = ["L P L P L P L"]  # Prepared sequence must have spaces or sm function to join them

# tokenize sequences and pad up to the longest sequence in the batch
ids = tokenizer(sequence_examples, add_special_tokens=True, padding="longest")

input_ids = torch.tensor(ids['input_ids']).to(device)
attention_mask = torch.tensor(ids['attention_mask']).to(device)

# generate embeddings
with torch.no_grad():
    embedding_repr = model(input_ids=input_ids, attention_mask=attention_mask)

#per-protein embedding for the truncated alpha-syn protein
emb_per_protein = embedding_repr.last_hidden_state.mean(dim=1)  # shape (1 x 1024)

# Write embeddings to files
def write_embeddings_to_file(embeddings, file_name):
    with open(file_name, 'w') as file:
        for embedding in embeddings:
            embedding_str = ' '.join([str(value) for value in embedding])
            file.write(embedding_str + '\n')

# Write embeddings to files
write_embeddings_to_file(emb_per_protein.numpy(), 'LPLPLPL_embedding.txt')
print("File 'LPLPLPL_embedding.txt' created successfully. \N{check mark} \n")