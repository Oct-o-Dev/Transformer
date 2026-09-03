import torch
from torch.utils.data import Dataset, DataLoader
from tokenizer import SimpleTokenizer

class CodeDataset(Dataset):
    def __init__(self, english_data, cpp_data, eng_tokenizer, cpp_tokenizer, seq_len):
        self.english_data = english_data
        self.cpp_data = cpp_data
        self.eng_tokenizer = eng_tokenizer
        self.cpp_tokenizer = cpp_tokenizer
        self.seq_len = seq_len
        self.pad_token = eng_tokenizer.word2idx["[PAD]"]

    def __len__(self):
        return len(self.english_data)

    def __getitem__(self, idx):
        src_text = self.english_data[idx]
        tgt_text = self.cpp_data[idx]

        # Encode strings to tensors (SOS and EOS are automatically added by our tokenizer)
        src_tokens = self.eng_tokenizer.encode(src_text)
        tgt_tokens = self.cpp_tokenizer.encode(tgt_text)

        # Calculate how many padding tokens we need to reach seq_len
        src_padding_len = self.seq_len - len(src_tokens)
        tgt_padding_len = self.seq_len - len(tgt_tokens)

        # Truncate sequences if they are somehow longer than seq_len
        if src_padding_len < 0:
            src_tokens = src_tokens[:self.seq_len]
            src_padding_len = 0
        if tgt_padding_len < 0:
            tgt_tokens = tgt_tokens[:self.seq_len]
            tgt_padding_len = 0

        # Pad the tensors with 0s
        src_padded = torch.cat([src_tokens, torch.tensor([self.pad_token] * src_padding_len)])
        tgt_padded = torch.cat([tgt_tokens, torch.tensor([self.pad_token] * tgt_padding_len)])

        # Create a mask so the model ignores the [PAD] tokens during attention
        src_mask = (src_padded != self.pad_token).unsqueeze(0).unsqueeze(0).int()

        return {
            "encoder_input": src_padded,           # Shape: (seq_len)
            "decoder_input": tgt_padded[:-1],      # Shape: (seq_len - 1) -> Everything except the last token
            "label": tgt_padded[1:],               # Shape: (seq_len - 1) -> Everything except the first token
            "src_mask": src_mask,
            "src_text": src_text,
            "tgt_text": tgt_text
        }

# ==========================================
# TEST IT LOCALLY 
# ==========================================
if __name__ == "__main__":
    # Dummy Dataset featuring algorithmic logic
    english_data = [
        "Initialize a vector of integers and sort it in ascending order",
        "Implement a binary search function for a sorted array"
    ]
    cpp_data = [
        "std :: vector < int > v ; std :: sort ( v . begin ( ) , v . end ( ) ) ;",
        "int binarySearch ( int arr [ ] , int l , int r , int x ) { }"
    ]

    # Initialize Tokenizers
    eng_tokenizer = SimpleTokenizer()
    cpp_tokenizer = SimpleTokenizer()
    eng_tokenizer.build_vocab(english_data)
    cpp_tokenizer.build_vocab(cpp_data)

    # Initialize Dataset and DataLoader
    seq_len = 25
    dataset = CodeDataset(english_data, cpp_data, eng_tokenizer, cpp_tokenizer, seq_len)
    
    # DataLoader groups our data into batches
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Fetch one batch to verify
    batch = next(iter(dataloader))
    
    print(f"Encoder Input Shape (batch, seq_len): {batch['encoder_input'].shape}")
    print(f"Decoder Input Shape (batch, seq_len-1): {batch['decoder_input'].shape}")
    print(f"Label Shape (batch, seq_len-1): {batch['label'].shape}")