---
day: 33
generated_at: '2026-10-08T15:56:26.671071+00:00'
phase: Phase 4 — The Transformer Revolution
recap_summary: Explained the architecture and mechanics of transformers in NLP, focusing
  on how self-attention enables each token to access context from the full sequence
  and showing a minimal implementation in NumPy.
status: pending_review
title: 'Day 33: Building a Mini Transformer From Scratch (Code Walkthrough)'
topic_title: Building a Mini Transformer From Scratch (Code Walkthrough)
---

Transformers are the architecture that reshaped natural language processing (NLP) in 2017. Before transformers, top-performing models used either recurrent neural networks (RNNs) or convolutional neural networks (CNNs). Transformers delivered dramatic improvements in tasks like machine translation, text summarization, and question answering. Their main strength is the **attention mechanism**, which allows the model to focus on important words anywhere in a sentence—not just nearby ones. All large language models (LLMs) you see today—GPT, BERT, Llama—are built on transformers or close relatives.

Before transformers, RNNs and CNNs faced substantial limits. RNNs process input step by step and struggle to connect words far apart. CNNs can spot local patterns, but to "see" across long texts, they need many layers. Both types struggle to use modern GPUs efficiently, since their computations can't be fully parallelized. Parts of the input must be processed before moving on.

Transformers, in contrast, process the entire input sequence at once. Their key invention—**self-attention**—lets every word "look at" every other word in the input. This enables rich, context-aware representations and allows fast, parallel computation.

Let's outline the main components of a transformer:

- **Input Embeddings:** The model translates words or tokens into vectors—lists of numbers—which capture basic meaning and structure.
- **Self-Attention:** Each embedding can gather information from every other embedding in the same sequence, according to their relationships. This collects context from the whole sequence for each token.
- **Feedforward Network:** Each embedding, now context-aware, passes through a simple neural network that further refines it.
- **Output:** This process can be repeated by stacking layers. The final output is a sequence of vectors, ready for predictions like text classification or next-word generation.

Focus on **self-attention**, the core mechanism. Take the sentence:  
**“The cat sat on the mat.”**  
Self-attention enables the token for "mat" to access information from "the", "cat", "sat", and "on", building a richer meaning in context.

Here's the core of self-attention, step by step:

1. For each input **token** (the model's internal representation of a word), create three different versions: **Query**, **Key**, and **Value**. Each version is a vector, made by multiplying the token embedding by small matrices.
2. For every pair of tokens, compute a "score" by taking the Query from one token and the Key from another. This is done by a dot product: multiply and sum their elements.
3. Scale and normalize these scores using the **softmax** function, so they add up to 1 for each Query. This gives you the attention weights—how much to "pay attention" to each other token.
4. Each token’s new embedding is a weighted sum of all the Value vectors, using these attention weights.

For example, if you have tokens A, B, and C, the self-attention for A computes:
- How much should A focus on itself? (A's Query dot A's Key)
- How much on B? (A's Query dot B's Key)
- How much on C? (A's Query dot C's Key)  
A's new representation is a weighted average of A, B, and C's Value vectors. Repeat for each token.

A **transformer layer** contains two main parts:
- A **Multi-Head Self-Attention** block (often runs several attention operations in parallel; for simplicity, we’ll use a single one here)
- A **Feedforward block** (a small neural network for each position)

Transformers stack several of these layers in sequence, creating deep and flexible representations.

Below is a runnable, minimal transformer built step by step with NumPy. This version is stripped to essentials: a single self-attention head, basic feedforward network, no extra features.

```python
import numpy as np

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)  # For numerical stability
    e_x = np.exp(x)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

class MiniTransformer:
    def __init__(self, d_model, seq_len, vocab_size):
        self.d_model = d_model
        self.seq_len = seq_len
        self.vocab_size = vocab_size

        # Random token embeddings (vocab_size x d_model)
        self.token_embeddings = np.random.randn(vocab_size, d_model) / np.sqrt(d_model)

        # Self-attention projection matrices (each d_model x d_model)
        self.W_Q = np.random.randn(d_model, d_model) / np.sqrt(d_model)
        self.W_K = np.random.randn(d_model, d_model) / np.sqrt(d_model)
        self.W_V = np.random.randn(d_model, d_model) / np.sqrt(d_model)

        # Feedforward: (d_model x d_ff) then (d_ff x d_model)
        d_ff = d_model * 2  # Hidden size in feedforward
        self.W1 = np.random.randn(d_model, d_ff) / np.sqrt(d_model)
        self.b1 = np.zeros((d_ff,))
        self.W2 = np.random.randn(d_ff, d_model) / np.sqrt(d_ff)
        self.b2 = np.zeros((d_model,))

    def forward(self, token_ids):
        # token_ids: (seq_len,) sequence of integers (token indices)
        x = self.token_embeddings[token_ids]  # (seq_len, d_model)

        # -- Self-Attention --
        Q = x @ self.W_Q      # (seq_len, d_model)
        K = x @ self.W_K      # (seq_len, d_model)
        V = x @ self.W_V      # (seq_len, d_model)

        # Attention scores (seq_len, seq_len)
        attn_scores = Q @ K.T / np.sqrt(self.d_model)
        attn_weights = softmax(attn_scores, axis=1) # (seq_len, seq_len)

        # Weighted sum: (seq_len, seq_len) @ (seq_len, d_model) -> (seq_len, d_model)
        attn_output = attn_weights @ V

        # -- Feedforward block --
        ff_hidden = np.maximum(0, attn_output @ self.W1 + self.b1)  # ReLU activation
        ff_output = ff_hidden @ self.W2 + self.b2

        return ff_output  # (seq_len, d_model)

# Example usage:
np.random.seed(42)
vocab_size = 10
d_model = 8  # Embedding size
seq_len = 5

# A sample "sentence": a list of integer token IDs
token_ids = np.array([2, 5, 3, 7, 1])

model = MiniTransformer(d_model=d_model, seq_len=seq_len, vocab_size=vocab_size)
output_vectors = model.forward(token_ids)
print(output_vectors.shape)  # (5, 8)
print(output_vectors)
```

What does this code do?
- Takes a fake sentence: a list of token IDs.
- Looks up their embeddings.
- Applies self-attention to let every token mix information from every other.
- Passes the updated vectors through a small neural network layer for refinement.

This is not a production system. There’s no training, no multi-head attention, no normalization, residual connections, or positional encoding—just the bare minimum mechanics. Its purpose is to make the moving parts of a transformer visible and concrete.

Change the input tokens, sequence length, or embedding size. Pass new sequences to `model.forward()`. Observe how the output vectors update.

By walking line-by-line through this simple transformer, you now see how input tokens turn into rich, context-aware representations. Self-attention and feedforward layers—each a simple mix of matrix multiplication and softmax—form the heart of every transformer. Modern large language models stack these pieces higher and wider, but the foundations remain the same.

---

## Key Takeaways

- Transformers use self-attention to allow tokens to access information from any other token in the sequence.
- Previous models like RNNs and CNNs were limited by sequential processing and locality.
- Key components of a transformer include input embeddings, self-attention, and feedforward networks.
- Self-attention computes attention scores between all token pairs to mix contextual information.
- A minimalist transformer can be built with just matrix multiplications and softmax operations.

## Try It Yourself

Modify the provided mini-transformer code to use an input sequence of 6 token IDs instead of 5. Print out the attention matrix (`attn_weights` in the code) for the new sequence. In 1-2 sentences, describe how the model redistributes attention across positions and what patterns you notice in the weights.

## Further Resources

- 🎥 [Coding a Transformer from scratch on PyTorch, with full explanation, training and inference](https://www.youtube.com/watch?v=ISNdQcPhsts)
- 🎥 [How to Code a Transformer FROM SCRATCH!](https://www.youtube.com/watch?v=XefFj4rLHgU)
- 📘 [Mini Transformer from Scratch (decoder‑only GPT‑style in PyTorch)](https://github.com/M-Wasil/Mini-Transformer-From-Scratch)
- 📘 [Mini Transformer: End‑to‑End Walkthrough (NumPy implementation)](https://github.com/Arpit-sakhreliya/mini-transformer-from-scratch)
- 📄 [Building a Miniature Transformer From Scratch: The 'Lego' Approach to Deep Learning](https://pythonanddatascience.com/blog/building-a-miniature-transformer-from-scratch-the-lego-approach-to-deep-learning)

---

**Coming up on Day 34:** What Is Pretraining? Transfer Learning for NLP