---
day: 30
generated_at: '2026-09-29T15:11:43.674849+00:00'
phase: Phase 4 — The Transformer Revolution
recap_summary: Explained the limitations of RNN/LSTM sequence models and introduced
  the Transformer architecture, highlighting self-attention, parallelism, and the
  structure of a Transformer encoder with practical PyTorch examples.
status: pending_review
title: 'Day 30: The Transformer Encoder Architecture'
topic_title: The Transformer Encoder Architecture
---

Why Sequence Models Struggled, and Why Transformers Matter

Early neural networks gave computers a way to process sequences of text. The first big successes used Recurrent Neural Networks (RNNs), later improved into Long Short-Term Memory (LSTM) networks. An RNN processes a sentence one word at a time, passing information forward at each step—like a baton in a relay race. This step-by-step process seems to match how language works, since words arrive in order.

But this design has major drawbacks:

- RNNs handle words one after another. Training can't be fully parallelized, so they're slow, especially for long sentences.
- Information from early words often "fades" as you move through the sequence—a problem called vanishing gradients. Even LSTMs, which were invented to help, still struggle to remember words that are far apart.
- RNNs naturally focus more on nearby words, because information from distant words has to pass through many intermediate steps. But in real sentences, the important links can be between words at opposite ends.

Because of these issues, RNNs have trouble with long texts, struggle to link distant words, and are slow to train.

Transformers changed the game. In 2017, researchers (Vaswani et al.) introduced an architecture that stops processing words sequentially. Instead, Transformers consider the entire sentence at once, letting each word connect with every other directly. This avoided the bottlenecks of RNNs, and quickly outperformed older models on natural language tasks. Transformers opened the path to large language models like BERT and GPT.

From Sequences to Sets: The Core Idea Behind the Transformer

Traditional models treat text as a chain, moving one link at a time. The Transformer acts differently: it looks at the whole chain at once and finds which links matter most for each word.

In practice, this means it processes all words in parallel. Every word gets updated by looking at all other words in the sentence. This parallelism makes Transformers much faster—every word gets handled together, fully using modern hardware.

It's also more effective. Each word can directly access the information from any other word, regardless of position in the sentence. There’s no special bias for recent words, and no drift in remembering the important links.

A Tour of the Transformer Encoder: The Big Picture

A Transformer encoder is the part of the architecture that reads in text and builds a smart, context-aware representation for each word. It consists of a stack of nearly identical layers—usually six or more, and sometimes many more in large models.

Each encoder layer is built from several parts:

- **Input Embeddings:** Convert words (or sometimes subword tokens) into dense vectors of numbers. Each vector represents a word.
- **Positional Encoding:** Since Transformers don't have any built-in way to know word order, we add numbers to help them tell which word comes first, second, etc.
- **Self-Attention:** The core innovation. Each word "looks at" every other word to decide what’s most important for understanding itself.
- **Feed-Forward Layers:** Standard neural network layers applied separately to each word’s vector.
- **Normalization and Residual Connections:** Methods to keep training stable and help information flow smoothly through deep stacks of layers.

For now, we’re focusing on the encoder—the section that reads sentences and produces context-based word vectors. We won't cover the decoder (used for generating text) or the full sequence-to-sequence Transformer just yet.

Self-Attention: Seeing Everything at Once

Self-attention is what makes Transformers powerful. Imagine reading a sentence and, for each word, thinking: "Which other words do I need to pay attention to, in order to understand this one?"

Take the sentence:
> "The animal didn't cross the street because it was too tired."

What does "it" refer to? Self-attention lets the model compare "it" with every other word, and, through training, learn that "animal" is the most relevant link.

Analogy: Picture a group meeting where everyone wears wireless headsets. At any instant, you (the model) can listen to the whole conversation and focus on the portions that matter for your next response. You’re not limited to just what the previous speaker said.

Each word computes an attention score for every word in the sentence—including itself. Then, it forms its new vector as a weighted average of all others, where each weight is how much it "attends" to that word. This step lets the model capture long-range relationships and context in a single shot.

What Flows Through the Encoder: Step by Step

Let’s walk through how data moves through a single Transformer encoder layer:

1. **Convert words to tokens:** Each word (or subword) in the sentence is mapped to a unique index from the vocabulary.
2. **Create embeddings:** Each token index becomes an embedding—a dense vector capturing word features.
3. **Add positional encoding:** Add numbers to each embedding to capture its position in the sentence. This gives the model a sense of word order.
4. **Apply self-attention:** For each word, create a new vector by combining information from all the other words. The amount contributed by each word is set by the attention weights.
5. **Apply feed-forward layer:** Each word’s vector is processed by the same little neural network. This is applied separately for each position.
6. **Add & Normalize:** The model combines the original vector with the new one (a residual connection), then normalizes to keep things stable during training.
7. **Stack more layers:** The output from one layer is passed as input to the next. With each layer, the representations become richer and more context-aware.

After all layers, you have a set of vectors—one per input word. These vectors now contain both the meaning of the word and its context in the sentence. They can be used for any downstream NLP task, or passed to further model components.

Transformers in Action: Minimal Encoder Example in PyTorch

Here’s a runnable PyTorch example. This builds a minimal Transformer encoder and runs it on random input shaped like word embeddings. There's no real data—you can focus on how the architecture fits together.

```python
import torch
import torch.nn as nn

# Dummy input: a "sentence" of 5 tokens, each mapped to a 16-dimensional embedding vector
batch_size = 1
seq_len = 5
emb_dim = 16
dummy_embeddings = torch.rand(seq_len, batch_size, emb_dim)  # shape: (seq_len, batch_size, emb_dim)

# Step 1: Positional encoding (classic sine/cosine approach)
class PositionalEncoding(nn.Module):
    def __init__(self, emb_dim, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, emb_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2) * (-torch.log(torch.tensor(10000.0)) / emb_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(1)  # (max_len, 1, emb_dim)

    def forward(self, x):
        return x + self.pe[:x.size(0)]

pos_encoder = PositionalEncoding(emb_dim)
emb_plus_pos = pos_encoder(dummy_embeddings)  # add position info to embeddings

# Step 2: Define a Transformer encoder layer (using PyTorch's built-in class)
encoder_layer = nn.TransformerEncoderLayer(
    d_model=emb_dim,
    nhead=4,                  # 4 self-attention heads
    dim_feedforward=32,       # hidden size in feed-forward layers
    dropout=0.0,              # no dropout for this demo
    activation='relu'
)

# Step 3: Stack encoder layers
transformer_encoder = nn.TransformerEncoder(
    encoder_layer,
    num_layers=2              # two stacked layers for demo
)

# Step 4: Pass inputs through encoder
output = transformer_encoder(emb_plus_pos)  # output shape: (seq_len, batch_size, emb_dim)

print("Input shape:", dummy_embeddings.shape)
print("Output shape:", output.shape)
print("Output data:", output.detach())
```

- **dummy_embeddings**: Stand-in for real token embeddings—each word as a vector.
- **PositionalEncoding**: Adds information about word order to each embedding.
- **encoder_layer** and **transformer_encoder**: Construct the main Transformer building blocks—self-attention plus feed-forward networks.
- **output**: For each token, you now have a "contextualized" vector. It captures both the identity of the word and how it fits into the whole sentence.

You can increase the number of tokens, layers, or attention heads to see how the architecture and data shapes change. This code grounds the Transformer theory in something concrete and runnable.

---

## Key Takeaways

- RNNs process text sequentially and struggle with long-range dependencies and slow training.
- Transformers process all words in parallel and use self-attention to link distant words directly.
- Positional encoding gives Transformers awareness of word order.
- A Transformer encoder builds rich contextual representations by stacking layers of self-attention and feed-forward networks.
- PyTorch makes it straightforward to implement and experiment with Transformer encoder components.

## Try It Yourself

Try modifying the dummy_embeddings in the provided PyTorch code to simulate a new sentence (e.g., change its shape or values). Run the code before and after your changes, and compare the output tensors to observe how different input embeddings produce different encoded representations.

## Further Resources

- 🎥 [Transformer Architecture Explained (Transformer Architecture Explanation from the paper: Attention is all you need)](https://www.youtube.com/watch?v=6vThlsJ_ASE)
- 📄 [Attention Is All You Need (original paper)](https://arxiv.org/abs/1706.03762)
- 📘 [TransformerEncoderLayer — PyTorch documentation](https://docs.pytorch.org/docs/main/generated/torch.nn.modules.transformer.TransformerEncoderLayer.html)
- 📄 [Attention Is All You Need: The Paper That Revolutionized AI](https://www.wasilzafar.com/pages/series/ai-data-science/attention-is-all-you-need-transformer-explained.html)

---

**Coming up on Day 31:** The Transformer Decoder Architecture