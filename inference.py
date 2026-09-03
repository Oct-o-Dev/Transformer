import torch
from tokenizer import SimpleTokenizer
from model import Transformer, generate_causal_mask
from datasets import load_dataset

# 1. Rebuild the exact same C++ tokenizers
print("Loading C++ dataset for inference tokenizers...")
raw_dataset = load_dataset("TokenBender/code_instructions_122k_alpaca_style", split="train")

english_data = []
cpp_data = []

for row in raw_dataset:
    prompt = row['instruction']
    code = row['output']
    if "c++" in prompt.lower() or "cpp" in prompt.lower():
        if len(prompt.split()) < 40 and len(code.split()) < 50:
            english_data.append(prompt)
            cpp_data.append(code)

english_data = english_data[:2000]
cpp_data = cpp_data[:2000]

eng_tokenizer = SimpleTokenizer()
cpp_tokenizer = SimpleTokenizer()
eng_tokenizer.build_vocab(english_data)
cpp_tokenizer.build_vocab(cpp_data)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Initialize the model with the updated seq_len
seq_len = 60 # This must exactly match train.py
model = Transformer(eng_tokenizer.vocab_size, cpp_tokenizer.vocab_size, 
                    seq_len, seq_len, d_model=512, num_heads=8, num_layers=4, d_ff=2048, dropout=0.1).to(device)

# Load the weights
model.load_state_dict(torch.load("mini_copilot_weights.pth", map_location=device, weights_only=True))
model.eval()

# 3. The Autoregressive Generation Loop
def generate_code(prompt, max_len=50):
    # Encode the English prompt
    src_tokens = eng_tokenizer.encode(prompt).unsqueeze(0).to(device)
    src_mask = (src_tokens != eng_tokenizer.word2idx["[PAD]"]).unsqueeze(0).unsqueeze(0).int().to(device)

    # Pass through the Encoder once
    with torch.no_grad():
        encoder_output = model.encode(src_tokens, src_mask)

    # Start the Decoder with just the [SOS] token
    decoder_input = torch.tensor([[cpp_tokenizer.word2idx["[SOS]"]]], dtype=torch.long).to(device)

    for _ in range(max_len):
        tgt_mask = generate_causal_mask(decoder_input.size(1)).to(device)
        
        with torch.no_grad():
            # Pass the growing sequence through the Decoder
            decoder_output = model.decode(encoder_output, src_mask, decoder_input, tgt_mask)
            
            # Pass the output through the projection layer to get vocabulary logits
            logits = model.projection(decoder_output)
        
        # Get the token with the highest probability at the last position
        next_token = logits[:, -1, :].argmax(dim=-1).item()
        
        # Break if the model predicts [EOS]
        if next_token == cpp_tokenizer.word2idx["[EOS]"]:
            break
            
        # Append the predicted token to the sequence and repeat
        decoder_input = torch.cat([decoder_input, torch.tensor([[next_token]], dtype=torch.long).to(device)], dim=1)

    # Decode the final sequence of integers back into text
    generated_code = cpp_tokenizer.decode(decoder_input.squeeze().tolist())
    # Remove the [SOS] token from the final output string
    return generated_code.replace("[SOS] ", "")

# ==========================================
# TEST THE MINI-COPILOT
# ==========================================
if __name__ == "__main__":
    test_prompt = "Write a function to find the maximum sum of a contiguous subarray"
    print(f"Prompt: {test_prompt}\n")
    
    generated = generate_code(test_prompt)
    print(f"Generated Output:\n{generated}")