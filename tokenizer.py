import torch

class SimpleTokenizer:
    def __init__(self):
        # 1. Define the mandatory special tokens
        self.special_tokens = {"[PAD]": 0, "[UNK]": 1, "[SOS]": 2, "[EOS]": 3}
        
        # 2. Initialize the dictionaries
        self.word2idx = self.special_tokens.copy()
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)

    def build_vocab(self, text_dataset):
        """Learns the vocabulary from a list of sentences/code snippets."""
        for sentence in text_dataset:
            # A very simple word-level split (you can customize this for C++ later)
            tokens = str(sentence).strip().split()
            for token in tokens:
                if token not in self.word2idx:
                    self.word2idx[token] = self.vocab_size
                    self.idx2word[self.vocab_size] = token
                    self.vocab_size += 1

    def encode(self, text, add_special_tokens=True):
        """Converts text into a list of integers."""
        tokens = str(text).strip().split()
        
        # Convert words to IDs, using [UNK] if the word isn't in vocab
        token_ids = [self.word2idx.get(token, self.word2idx["[UNK]"]) for token in tokens]
        
        if add_special_tokens:
            token_ids = [self.word2idx["[SOS]"]] + token_ids + [self.word2idx["[EOS]"]]
            
        return torch.tensor(token_ids, dtype=torch.long)

    def decode(self, token_ids):
        """Converts integers back into human-readable text."""
        # Convert PyTorch tensor to Python list if needed
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
            
        words = [self.idx2word.get(idx, "[UNK]") for idx in token_ids]
        return " ".join(words)

# ==========================================
# Testing it Locally First
# ==========================================
if __name__ == "__main__":
    # Dummy Dataset for C++ logic
    english_data = [
        "Write a function to return the sum of two integers",
        "Print hello world to the console"
    ]
    
    cpp_data = [
        "int sum ( int a , int b ) { return a + b ; }",
        "std :: cout << \" hello world \" << std :: endl ;"
    ]

    # Initialize two separate tokenizers
    eng_tokenizer = SimpleTokenizer()
    cpp_tokenizer = SimpleTokenizer()

    # Build vocabularies
    eng_tokenizer.build_vocab(english_data)
    cpp_tokenizer.build_vocab(cpp_data)

    print(f"English Vocab Size: {eng_tokenizer.vocab_size}")
    print(f"C++ Vocab Size: {cpp_tokenizer.vocab_size}\n")

    # Test Encoding
    sample_text = "Print hello world to the console"
    encoded_tensor = eng_tokenizer.encode(sample_text)
    print(f"Raw Text: {sample_text}")
    print(f"Encoded Tensor: {encoded_tensor}")
    
    # Test Decoding
    decoded_text = eng_tokenizer.decode(encoded_tensor)
    print(f"Decoded Text: {decoded_text}")