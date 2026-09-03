import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        # PyTorch's built-in lookup table that maps integers to vectors
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # The paper specifies multiplying the embeddings by the square root of d_model 
        # to scale the weights before adding the positional encodings.
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Create a matrix of shape (seq_len, d_model) full of zeros
        pe = torch.zeros(seq_len, d_model)
        
        # Create a vector for the position (0, 1, 2, ..., seq_len - 1)
        # shape: (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        
        # Create the denominator tensor: 10000^(2i/d_model)
        # We use exponentiation with log for numerical stability
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply Sine to even indices (0, 2, 4...)
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply Cosine to odd indices (1, 3, 5...)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add a batch dimension to the pe matrix: (1, seq_len, d_model)
        pe = pe.unsqueeze(0)
        
        # register_buffer tells PyTorch to save this tensor with the model's state, 
        # but it is NOT a learned parameter that gets updated during backprop.
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x is the embedded input: (batch_size, seq_len, d_model)
        # We add the positional encodings to the embeddings. 
        # We slice self.pe so it matches the actual sequence length of x
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        # Ensure the embedding dimension is perfectly divisible by the number of heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections for Query, Key, and Value
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        
        # Final output projection
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # 1. Linear Projection & Shape Manipulation
        # Original shape: (batch, seq_len, d_model)
        # Target shape for attention: (batch, num_heads, seq_len, d_k)
        Q = self.w_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 2. Scaled Dot-Product Attention
        # Multiply Q by K-transposed. Transpose the last two dimensions (seq_len, d_k) -> (d_k, seq_len)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply the mask (critical for the Decoder so it cannot "see" future tokens)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
            
        # Convert scores to probabilities
        attention_weights = torch.softmax(attention_scores, dim=-1)
        
        # Multiply by V
        attention_output = torch.matmul(attention_weights, V)
        
        # 3. Concatenate all the heads back together
        # Transpose back to (batch, seq_len, num_heads, d_k), then flatten the last two dims
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # 4. Final linear projection
        output = self.w_o(attention_output)
        
        return output    




class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        # Expands from 512 to 2048 dimensions
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        # Compresses from 2048 back to 512 dimensions
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class ResidualConnection(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, sublayer):
        # Implements: LayerNorm(x + Dropout(Sublayer(x)))
        return self.norm(x + self.dropout(sublayer(x)))

class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attention_block = MultiHeadAttention(d_model, num_heads)
        self.feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        # Create a list of two identical residual connections
        self.residual_connections = nn.ModuleList([ResidualConnection(d_model, dropout) for _ in range(2)])

    def forward(self, x, src_mask=None):
        # 1. Multi-Head Attention wrapped in Add & Norm
        # We use a lambda function to pass the sublayer execution into the residual wrapper
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(q=x, k=x, v=x, mask=src_mask))
        
        # 2. Feed-Forward Network wrapped in Add & Norm
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x

class Encoder(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float, num_layers: int):
        super().__init__()
        # Stack N identical EncoderBlocks (The paper uses N=6)
        self.layers = nn.ModuleList([EncoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, src_mask=None):
        for layer in self.layers:
            x = layer(x, src_mask)
        # Apply final layer normalization to the encoder stack
        return self.norm(x)


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float):
        super().__init__()
        # The Decoder uses two separate attention blocks
        self.self_attention_block = MultiHeadAttention(d_model, num_heads)
        self.cross_attention_block = MultiHeadAttention(d_model, num_heads)
        self.feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        
        # Three sub-layers mean we need three residual connections
        self.residual_connections = nn.ModuleList([ResidualConnection(d_model, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # 1. Masked Self-Attention (C++ tokens look at themselves)
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(q=x, k=x, v=x, mask=tgt_mask))
        
        # 2. Cross-Attention (Query = C++, Key & Value = English Encoder Output)
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(q=x, k=encoder_output, v=encoder_output, mask=src_mask))
        
        # 3. Feed-Forward Network
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x

class Decoder(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float, num_layers: int):
        super().__init__()
        # Stack N identical DecoderBlocks (The paper uses N=6)
        self.layers = nn.ModuleList([DecoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        # Apply final layer normalization to the decoder stack
        return self.norm(x)


class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # We return the raw logits. PyTorch's CrossEntropyLoss handles the softmax internally.
        return self.proj(x)

class Transformer(nn.Module):
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, 
                 d_model: int = 512, num_heads: int = 8, num_layers: int = 6, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        
        self.src_embed = InputEmbeddings(d_model, src_vocab_size)
        self.tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)
        
        self.src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
        self.tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
        
        self.encoder = Encoder(d_model, num_heads, d_ff, dropout, num_layers)
        self.decoder = Decoder(d_model, num_heads, d_ff, dropout, num_layers)
        
        self.projection = ProjectionLayer(d_model, tgt_vocab_size)

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        encoder_output = self.encode(src, src_mask)
        decoder_output = self.decode(encoder_output, src_mask, tgt, tgt_mask)
        return self.projection(decoder_output)

def generate_causal_mask(size):
    # Returns a lower-triangular matrix (1s for past/current tokens, 0s for future tokens)
    # Our self-attention block masks out elements where mask == 0
    return torch.tril(torch.ones(1, size, size)).type(torch.int)

# ==========================================
# TEST IT LOCALLY ON YOUR ASUS TUF F15
# ==========================================
if __name__ == "__main__":
    # Vocab sizes based on your local tokenizer test
    src_vocab_size = 18  # English
    tgt_vocab_size = 24  # C++
    
    src_seq_len = 8
    tgt_seq_len = 10
    
    # Initialize the entire model architecture
    model = Transformer(src_vocab_size, tgt_vocab_size, src_seq_len, tgt_seq_len)
    
    # Simulate raw tokenized integer inputs (simulating the output of your SimpleTokenizer)
    src_input = torch.randint(0, src_vocab_size, (1, src_seq_len))
    tgt_input = torch.randint(0, tgt_vocab_size, (1, tgt_seq_len))
    
    # Generate the look-ahead mask for the target sequence
    tgt_mask = generate_causal_mask(tgt_seq_len)
    
    # Execute a full forward pass
    output = model(src_input, tgt_input, src_mask=None, tgt_mask=tgt_mask)
    
    print(f"Source Input shape (batch, src_seq_len): {src_input.shape}")
    print(f"Target Input shape (batch, tgt_seq_len): {tgt_input.shape}")
    print(f"Final Model Output shape (batch, tgt_seq_len, tgt_vocab_size): {output.shape}")