# Mini-Copilot: Transformer from Scratch (English to C++)

An end-to-end PyTorch implementation of a sequence-to-sequence Transformer that translates natural-language programming descriptions into C++ algorithmic routines.

The project follows *[Attention Is All You Need](https://arxiv.org/abs/1706.03762)* (Vaswani et al., 2017) and implements the architecture from foundational PyTorch tensor operations—without using `nn.Transformer` or Hugging Face model classes.

## Features

- Scaled dot-product attention and multi-head attention
- Sinusoidal positional encoding
- Encoder and masked decoder stacks
- Layer normalization, residual connections, and dropout
- Padding and causal attention masks
- Teacher-forced training with padding-aware cross-entropy loss
- Autoregressive greedy decoding
- Separate word-level vocabularies for English and C++

## Architecture Overview

```text
English tokens → Embedding + Positional Encoding → Encoder
													 ↓
C++ tokens    → Embedding + Positional Encoding → Masked Decoder
													 ↓
									  Linear Projection → Logits
```

The decoder uses masked self-attention to prevent access to future target tokens and cross-attention to consume the encoder output.

## Mathematical Formulations

### Positional Encoding

Sinusoidal positional information is added to token embeddings:

\[
PE_{(pos, 2i)} = \sin\left(pos / 10000^{2i/d_{model}}\right)
\]

\[
PE_{(pos, 2i+1)} = \cos\left(pos / 10000^{2i/d_{model}}\right)
\]

### Scaled Dot-Product Attention

\[
Attention(Q,K,V) = softmax\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V
\]

`M` represents padding or causal masking.

### Position-Wise Feed-Forward Network

\[
FFN(x) = ReLU(xW_1 + b_1)W_2 + b_2
\]

## Repository Structure

```text
├── model.py        # Attention, positional encoding, encoder, decoder, Transformer
├── tokenizer.py    # Word-level source and target vocabularies
├── dataset.py      # Dataset, padding, and attention-mask construction
├── train.py        # Optimization loop, loss masking, and checkpoint export
├── inference.py    # Autoregressive English-to-C++ generation
└── readme.md       # Project documentation
```

## Components

| File | Component | Description |
|---|---|---|
| `model.py` | `MultiHeadAttention` | Splits projections into attention heads, computes attention, and concatenates the results. |
| `model.py` | `PositionalEncoding` | Caches non-learnable sinusoidal encodings in a PyTorch buffer. |
| `model.py` | `ResidualConnection` | Applies normalization, dropout, and residual sub-layer connections. |
| `model.py` | `generate_causal_mask` | Builds a lower-triangular mask that blocks future decoder positions. |
| `tokenizer.py` | `SimpleTokenizer` | Maintains source and target vocabularies with reserved special tokens. |
| `dataset.py` | `CodeDataset` | Pads sequences, creates masks, and prepares autoregressive targets. |

## Data Pipeline

Training uses the standard one-token target shift:

```text
Raw:    [SOS], std, ::, vector, ;, [EOS]
Input:  [SOS], std, ::, vector, ;
Label:  std, ::, vector, ;, [EOS]
```

The decoder predicts the next token at each position. Padding positions are ignored with `ignore_index=pad_idx`.

## Getting Started

### Requirements

- Python 3.10+
- PyTorch 2.0+ (CUDA is recommended)
- Hugging Face `datasets`

Install dependencies:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install datasets
```

### Train

```bash
python train.py
```

The trained checkpoint is saved as `mini_copilot_weights.pth`.

### Generate C++

```bash
python inference.py
```

Inference encodes the English prompt once, starts decoding with `[SOS]`, and repeatedly appends the highest-scoring token until `[EOS]` or the maximum sequence length is reached.

## Default Hyperparameters

| Parameter | Value |
|---|---:|
| Model dimension (`d_model`) | 512 |
| Attention heads | 8 |
| Encoder/decoder layers | 4 |
| Feed-forward dimension | 2048 |
| Dropout | 0.1 |
| Batch size | 16 |
| Sequence length | 60 |
| Adam learning rate | `1e-4` |

## Engineering Notes

1. Embedding inputs and cross-entropy targets must use `torch.long` tensors.
2. Encoder attention uses padding masks; decoder attention combines padding and causal masks.
3. Whitespace tokenization is useful for experimentation but cannot reliably represent compact C++ syntax such as `std::vector<int>` or `arr[i]++`.

## Roadmap

- [ ] Add a BPE tokenizer trained on C++ source files.
- [ ] Implement beam search with length penalties.
- [ ] Add key-value caching for faster autoregressive inference.
- [ ] Scale training to the full CodeSearchNet C++ corpus.

## Reference

Vaswani et al., [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762), 2017.
