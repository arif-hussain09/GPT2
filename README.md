# GPT-2 From Scratch — What I Learned

This repository contains my from-scratch implementation of **GPT-2 (124M)** in PyTorch while following Andrej Karpathy's **"Let's reproduce GPT-2 (124M)"** walkthrough.

The goal of this project was not just to implement a Transformer, but to understand what happens underneath an LLM — from QKV projections and attention to GPU memory, CUDA kernels, optimization, batching, distributed training, and evaluation.

> Reference: [Andrej Karpathy — Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU)

---

## 1. GPT-2 Architecture

The model follows the decoder-only Transformer architecture used by GPT-2.

### Configuration

- Vocabulary size: `50,304` in the training implementation
- Context length: `1024`
- Embedding dimension: `768`
- Transformer layers: `12`
- Attention heads: `12`
- Head dimension: `768 / 12 = 64`
- MLP hidden dimension: `4 × 768 = 3072`
- Approximately 124M parameters

The main flow is:

```text
Token IDs
    ↓
Token Embeddings + Position Embeddings
    ↓
12 × Transformer Blocks
    ├── LayerNorm
    ├── Causal Self-Attention
    ├── Residual Connection
    ├── LayerNorm
    ├── MLP + GELU
    └── Residual Connection
    ↓
Final LayerNorm
    ↓
LM Head
    ↓
Logits