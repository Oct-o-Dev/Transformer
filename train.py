import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from tokenizer import SimpleTokenizer
from dataset import CodeDataset
from model import Transformer, generate_causal_mask

# ==========================================
# 1. SETUP AND HUGGING FACE DATASET (C++ SWAP)
# ==========================================
print("Downloading multi-language dataset from Hugging Face...")
# Loading a larger multi-language instruction dataset
raw_dataset = load_dataset("TokenBender/code_instructions_122k_alpaca_style", split="train")

english_data = []
cpp_data = []

print("Filtering strictly for C++ algorithmic data...")
for row in raw_dataset:
    prompt = row['instruction']
    code = row['output']
    
    # Force the dataset to only grab C++ related instructions
    if "c++" in prompt.lower() or "cpp" in prompt.lower():
        # Keep sequences manageable for the space-based tokenizer
        if len(prompt.split()) < 40 and len(code.split()) < 50:
            english_data.append(prompt)
            cpp_data.append(code)

# Limit to 2000 to keep the Colab RAM stable during vocabulary building
english_data = english_data[:2000]
cpp_data = cpp_data[:2000]

print(f"Successfully extracted {len(english_data)} C++ logic pairs.")

eng_tokenizer = SimpleTokenizer()
cpp_tokenizer = SimpleTokenizer()
eng_tokenizer.build_vocab(english_data)
cpp_tokenizer.build_vocab(cpp_data)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

# Increase sequence length to handle verbose C++ syntax
seq_len = 60       
batch_size = 16    

# T4 GPU Transformer parameters
d_model = 512
num_heads = 8
num_layers = 4     
d_ff = 2048
dropout = 0.1
lr = 1e-4
epochs = 20

# ==========================================
# 2. INITIALIZATION
# ==========================================
dataset = CodeDataset(english_data, cpp_data, eng_tokenizer, cpp_tokenizer, seq_len)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Instantiate the model and send it to the GPU
model = Transformer(eng_tokenizer.vocab_size, cpp_tokenizer.vocab_size, 
                    seq_len, seq_len, d_model, num_heads, num_layers, d_ff, dropout).to(device)

# Adam optimizer is standard for Transformer architectures
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# CrossEntropyLoss automatically applies softmax. 
# CRITICAL: We tell it to completely ignore the [PAD] token.
pad_idx = cpp_tokenizer.word2idx["[PAD]"]
loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx)

# ==========================================
# 3. THE TRAINING LOOP
# ==========================================
for epoch in range(epochs):
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        # Move all tensors to the GPU
        # Move all tensors to the GPU and explicitly cast inputs to integers (long)
        encoder_input = batch['encoder_input'].to(device, dtype=torch.long)
        decoder_input = batch['decoder_input'].to(device, dtype=torch.long)
        label = batch['label'].to(device, dtype=torch.long)
        
        src_mask = batch['src_mask'].to(device)
        label = batch['label'].to(device)                 # (batch, seq_len-1)
        
        # Generate the causal mask so the decoder cannot look at future tokens
        tgt_mask = generate_causal_mask(decoder_input.size(1)).to(device)
        
        # Forward pass
        output = model(encoder_input, decoder_input, src_mask, tgt_mask) 
        # output shape: (batch, seq_len-1, vocab_size)
        
        # Flatten the output and labels for CrossEntropyLoss
        output_flatten = output.view(-1, cpp_tokenizer.vocab_size)
        label_flatten = label.view(-1).long()
        
        # Calculate loss
        loss = loss_fn(output_flatten, label_flatten)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    print(f"Epoch {epoch+1:02d}/{epochs} | Loss: {total_loss/len(dataloader):.4f}")