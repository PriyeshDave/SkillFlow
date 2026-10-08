---
day: 31
generated_at: '2026-09-30T15:27:21.841348+00:00'
phase: Phase 4 — The Transformer Revolution
recap_summary: Explained the structure and function of a transformer decoder, including
  the roles of masked self-attention, positional encoding, feed-forward networks,
  layer normalization, and how these components enable autoregressive sequence generation.
status: pending_review
title: 'Day 31: The Transformer Decoder Architecture'
topic_title: The Transformer Decoder Architecture
---

## What Is a Transformer Decoder?

A transformer decoder is a type of neural network for generating sequences, such as sentences, one piece at a time. You can think of it as an intelligent “writer” that produces each next word by considering what it’s already written. The decoder architecture powers models like GPT (Generative Pretrained Transformer), enabling them to generate fluent, context-aware language.

Older neural models, like recurrent neural networks (RNNs) and long short-term memory networks (LSTMs), process sequences step by step. Transformers changed this: the decoder looks at all previous words at once, deciding which ones matter most. This approach allows it to “remember” and use information from far back in a sequence, leading to more coherent generation.

## The Building Blocks: Attention and Positional Encoding

Transformer decoders rely on two major ideas: **attention mechanisms** and **positional encoding**.

**Attention mechanisms**: Imagine writing a story and being able to instantly reread any word you’ve written so far, paying more attention to some parts than others when considering what to write next. That’s what attention does. Specifically, “self-attention” means the decoder examines all the tokens it has already generated, weighting how important each is for the current decision.

**Positional encoding**: Transformers can’t tell the order of tokens by themselves—they process inputs in parallel. Positional encoding fixes this. It attaches a unique numerical signal to each word’s representation, marking "first word," "second word," and so on. This encoding can be a fixed pattern (like a wave) or learned by the model.

## Step-by-Step: Inside the Decoder Block

A transformer decoder block has a distinct structure. Here’s how data flows through it:

1. **Masked Self-Attention Layer**  
   This is the heart of the decoder. At each step, a token can “attend to” (look at) all tokens up to its own position, but never the future. Without this mask, the model could cheat by seeing parts of the sequence not yet generated. Masking ensures the model generates text one piece at a time, just like humans do.

2. **Cross-Attention Layer (optional)**  
   When using an encoder-decoder setup (common in translation), the cross-attention layer allows the decoder to attend to the encoder’s output. For example, in translation, the encoder processes the French source sentence, and the decoder uses this information to generate the English output. This layer is omitted in models (like GPT) that generate text from scratch.

3. **Feed-Forward Network (FFN)**  
   Next, each token’s position receives the same mini neural network (a multi-layer perceptron). This models local, nonlinear transformations after context has been mixed by attention.

4. **Layer Normalization**  
   Each attention or FFN step is followed by layer normalization. This standardizes activations, so training is more stable and the network doesn’t diverge.

In a full model, these decoder blocks are stacked. Each layer refines the model’s understanding based on context from earlier layers.

## How a Transformer Decoder Generates Sequences

The transformer decoder is used for **autoregressive** sequence generation: it generates output one token at a time, always conditioning on what it has produced so far.

- Start with a prompt (a few words).
- The decoder applies masked self-attention to the prompt, then predicts the next token.
- The new token is appended to the sequence.
- The decoder receives the updated sequence and again predicts the next token.
- This cycle repeats until the output is complete or a special end-of-sequence token is predicted.

Because of masking, the decoder never “peeks” at later tokens in the sequence. It must generate in order, just as natural language is written.

## Decoder vs. Encoder: Key Differences

Transformers often have both encoders and decoders, but each is tuned for different tasks.

- **Masking**: The decoder’s self-attention is masked to prevent future tokens from being seen. The encoder’s self-attention is unmasked and can see the entire sequence.
- **Cross-attention**: Only the decoder (in encoder-decoder models) includes a cross-attention layer, allowing it to consult the encoder’s output. Models like GPT skip this.
- **Role**: The encoder processes a full sequence all at once to extract meaning. The decoder assembles an output sequence token by token, each time consulting its own prior outputs (and, when present, the encoder’s outputs).

Think of the encoder as a “reader” and the decoder as a “writer.” For models like GPT, we focus just on the decoder side.

## Minimal PyTorch Example: Transformer Decoder Block

The following PyTorch code provides a basic, runnable example. It embeds tokens, adds positional information, applies masked self-attention, passes data through a feed-forward network, and normalizes outputs. This covers the minimal logic of a decoder block.

```python
import torch
import torch.nn as nn
import math

# Settings
vocab_size = 100   # Example vocabulary size
d_model = 32       # Embedding dimension
seq_len = 5

# Simulated input (batch size 1, sequence of 5 tokens)
tokens = torch.randint(0, vocab_size, (1, seq_len))

# Embedding + positional encoding
token_embedding = nn.Embedding(vocab_size, d_model)
positional_encoding = torch.zeros(1, seq_len, d_model)
for pos in range(seq_len):
    for i in range(0, d_model, 2):
        positional_encoding[0, pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
        if i + 1 < d_model:
            positional_encoding[0, pos, i+1] = math.cos(pos / (10000 ** ((2 * (i+1))/d_model)))

x = token_embedding(tokens) + positional_encoding  # (1, seq_len, d_model)

# Mask for autoregressive attention: prevent looking ahead
attn_mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)

# Single decoder block: attention, FFN, layer norm
attention = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
ffn = nn.Sequential(
    nn.Linear(d_model, d_model * 4),
    nn.ReLU(),
    nn.Linear(d_model * 4, d_model)
)
layernorm1 = nn.LayerNorm(d_model)
layernorm2 = nn.LayerNorm(d_model)

# Forward pass
attn_output, _ = attention(x, x, x, attn_mask=attn_mask)
x = layernorm1(x + attn_output)  # Residual connection
ffn_output = ffn(x)
output = layernorm2(x + ffn_output)  # Residual connection

print("Output shape:", output.shape)  # (1, seq_len, d_model)
```

This example:

- Embeds tokens and adds position signals.
- Uses masked self-attention, so each token can only attend to previous tokens.
- Applies a feed-forward network to each position.
- Normalizes after both attention and feed-forward steps.
- Mimics the computation in a single decoder block, omitting deep details.

The crucial part is the attention mask—it enforces “no peeking ahead,” so the decoder can generate text one step at a time, naturally building up the output sequence.

---

## Key Takeaways

- Transformer decoders generate sequences one token at a time using masked self-attention.
- Masked attention prevents access to future tokens, enforcing sequential generation.
- Positional encoding allows transformers to recognize token order in sequences.
- Decoder blocks include attention layers, feed-forward networks, and layer normalization.
- Cross-attention is used only when attending to encoder outputs, as in translation models.

## Try It Yourself

Draw or write code to create the attention mask matrices at each step as a five-token sequence is generated by a transformer decoder. For each generation step (from the first to fifth token), show which positions the current token attends to and indicate the masking effect.

## Further Resources

- 🎥 [Transformer Decoder Architecture | Deep Learning | CampusX](https://www.youtube.com/watch?v=DI2_hrAulYo)
- 📄 [The Transformer Decoder | Deep Learning Notes](https://xiaonanzang.github.io/deep-learning-notes/transformer-decoder.html)
- 📄 [Understanding Transformers I – The Decoder](https://danielchen0.github.io/2025/05/31/Understanding-Transformers-I-The-Decoder.html)
- 📄 [A Visual Guide to a Decoder‑only Transformer (Hugging Face)](https://huggingface.co/blog/velmen/a-visual-introduction-to-decoder-only-transformer)
- 📄 [Implementing the Transformer Decoder from Scratch in TensorFlow and Keras](https://machinelearningmastery.com/implementing-the-transformer-decoder-from-scratch-in-tensorflow-and-keras/)

---

**Coming up on Day 32:** Layer Normalization and Residual Connections