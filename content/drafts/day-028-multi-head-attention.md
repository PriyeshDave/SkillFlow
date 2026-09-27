---
day: 28
generated_at: '2026-09-25T14:15:26.525584+00:00'
phase: Phase 4 — The Transformer Revolution
recap_summary: Explained multi-head attention as used in Transformer models, detailing
  how multiple attention heads enable the capture of diverse and overlapping patterns
  within input sequences through independent, parallel computations, culminating in
  a combined representation.
status: pending_review
title: 'Day 28: Multi-Head Attention'
topic_title: Multi-Head Attention
---

What is Multi-Head Attention?
---

Multi-head attention is a fundamental component of Transformer models. Transformers are now standard for tasks like translation and text summarization. Multi-head attention works by letting the model use several “spotlights,” called attention heads, to focus on different parts of the input at the same time. Each head can highlight different relationships or patterns in the sequence. This setup allows the model to represent more complex information than with a single attention mechanism.

Recap: Attention Basics
---

In deep learning, **attention** is a mechanism that lets a model decide which parts of an input sequence are most relevant when generating each part of the output. When you answer a question about a paragraph, you don’t reread every word equally; you focus on the sentences with the most useful clues. Attention lets a model do something similar—it gives each output position a map showing how much to “look at” each input position, based on learned scores.

For example, in a translation system, when generating a word in French, the model can focus on the related English word in the input.

Why Multiple Attention Heads?
---

A single attention layer is like reading with just one highlighter. You can track one kind of relationship at a time. But language and sequential data have many overlapping patterns, such as grammar, references, and idioms.

**Multiple attention heads** let each head “specialize.” One head might pick out short-range patterns, like matching verbs to subjects (“the cat chased the mouse”). Another might track punctuation or sentence structure (“If..., then...” relationships). All heads process the same input but notice different connections.

Think of a team of detectives. Each detective examines the same evidence but looks for a different clue—fingerprints, timelines, motives. Together, they can solve more than any individual alone.

How Multi-Head Attention Works: Step by Step
---

Let’s break down the process:

**1. Inputs**

You start with a batch of input sequences. Each sequence is a list of word or token embeddings—vectors that represent the words.

**2. Linear Projections: Queries, Keys, and Values**

Each input embedding is passed through three trainable linear transformations (simple matrix multiplications):

- **Query (Q):** Represents the question or “what we’re looking for.”
- **Key (K):** Represents what each input token “offers.”
- **Value (V):** The information available at each token.

Analogy: In a database lookup, Query is your search term, Key is a searchable field, and Value is the full record if the Key matches.

**3. Split into Multiple Heads**

Instead of using big Q, K, and V matrices all at once, slice them into *h* smaller versions. Each slice is one head. So if you have 8 heads, you split the embedding into 8 strips—one per head.

**4. Each Head Computes Attention Independently**

Each head does its own attention calculation:

- It measures how well every Query matches every Key (often by a scaled dot product—multiply and sum the vectors, then scale).
- The matching scores become weights, showing how much attention to pay to each position.
- The head uses these weights to combine (blend) the Value vectors, producing a context vector for each head.

**5. Concatenate and Project**

Once every head has its output, stack the result vectors side by side to re-form the original embedding size. Pass this combined output through one more linear transformation. This mixes the information from all heads.

**Key Terms Recap:**

- **Head:** One independent attention mechanism, focusing on some aspect of the input.
- **Query, Key, Value:** Projections of the input—each plays a specific role in the attention calculation.
- **Projection:** A trainable linear transformation (matrix multiply) that creates Q, K, or V from the input vectors.

Visualizing Multi-Head Attention
---

Picture a sequence: [A, B, C, D, E, F].

Imagine several stacked rows, one for each head. Head 1 draws arrows from E to D and F—maybe capturing local context. Head 2 links B to A, picking out a name or title. Head 3 draws a wide arc from C to F, perhaps tracking a pronoun linking back to its noun.

Each head draws different patterns. At the end, their perspectives are joined and distilled into a new embedding for each position in the sequence.

Code Example: Multi-Head Attention in PyTorch
---

Below is a minimal PyTorch implementation of multi-head attention for small, fixed-size inputs. This code is kept simple to focus on the main idea.

```python
import torch
import torch.nn.functional as F

# Parameters
batch_size = 2
seq_len = 4         # Sequence length
embed_dim = 8       # Embedding size of input tokens
num_heads = 2       # Number of attention heads
head_dim = embed_dim // num_heads

# Example data: batch of sequences, each token is a vector
x = torch.randn(batch_size, seq_len, embed_dim)

# Trainable projection matrices for queries, keys, values
W_q = torch.randn(embed_dim, embed_dim)
W_k = torch.randn(embed_dim, embed_dim)
W_v = torch.randn(embed_dim, embed_dim)

# Output projection after concatenation
W_o = torch.randn(embed_dim, embed_dim)

def split_heads(tensor, num_heads):
    # Reshape last dimension for num_heads, transpose to (batch, heads, seq, head_dim)
    batch, seq, embed = tensor.size()
    tensor = tensor.view(batch, seq, num_heads, head_dim)
    return tensor.transpose(1, 2)

def combine_heads(tensor):
    # Reverse: (batch, heads, seq, head_dim) -> (batch, seq, embed_dim)
    batch, num_heads, seq, head_dim = tensor.size()
    tensor = tensor.transpose(1, 2).contiguous()
    return tensor.view(batch, seq, num_heads * head_dim)

# 1. Project input vectors for queries, keys, and values
Q = x @ W_q        # (batch, seq, embed_dim)
K = x @ W_k
V = x @ W_v

# 2. Split into heads
Q = split_heads(Q, num_heads)   # (batch, num_heads, seq, head_dim)
K = split_heads(K, num_heads)
V = split_heads(V, num_heads)

# 3. Scaled dot-product attention for each head
scores = Q @ K.transpose(-2, -1) / (head_dim ** 0.5)  # (batch, num_heads, seq, seq)
weights = F.softmax(scores, dim=-1)
heads = weights @ V                                   # (batch, num_heads, seq, head_dim)

# 4. Concatenate heads and final projection
concat = combine_heads(heads)                         # (batch, seq, embed_dim)
output = concat @ W_o                                 # (batch, seq, embed_dim)

print("Output shape:", output.shape)
# Output: torch.Size([2, 4, 8])
```

This example demonstrates:

- Projecting each token’s vector to query, key, and value vectors.
- Dividing them into heads, so each head sees a slice of the embedding.
- Each head running attention in parallel, getting its own view.
- Concatenating all heads and applying an output projection.

When Does Multi-Head Attention Help?
---

Multi-head attention makes models better at capturing the complexity and subtlety in sequences. In language, for example, relationships are not just local—words can refer to ideas many tokens away, and multiple patterns overlap.

With several heads, models can:

- Track different kinds of relationships simultaneously.
- Build separate attention maps for different linguistic or logical functions.
- Combine information across short and long ranges in one pass.

This ability is the core reason why Transformers, relying on multi-head attention, became the dominant choice for machine translation, summarization, and general language modeling. It also works well for images, audio, and any task where sequences have several layered relationships.

---

## Key Takeaways

- Multi-head attention allows the model to focus on different parts of the input sequence simultaneously.
- Each attention head specializes in discovering unique relationships or patterns within the data.
- Queries, Keys, and Values are created via trainable linear projections of the input embeddings.
- Outputs from all heads are concatenated and linearly projected to form the final result.
- This mechanism enables Transformers to handle complex dependencies in language and other sequential data.

## Try It Yourself

Modify the provided PyTorch code so that each attention head can only attend to a specific subset of sequence positions—for example, restrict the first head to attend only to the first half of the sequence, and the second head to the second half. Run the code and observe how the outputs differ from the original implementation. Write a few sentences describing what changed in the final attention output and what this suggests about the role of attention heads.

## Further Resources

- 🎥 [Multi-Head Attention Explained Visually | Simple Transformer Guide](https://www.youtube.com/watch?v=42L1q1Z4Ojc)
- 🎥 [Day 7 | Transformer Architecture Series | Multi-Head Self-Attention From First Principles](https://www.youtube.com/watch?v=rKjaI6apojQ)
- 📄 [How Transformers Think: Self-Attention and Multi-Head Attention from First Principles](https://rishabh-mondal.github.io/blogs/self-attention-multi-head-attention.html)
- 📄 [Self-Attention from Scratch: The Core of Every LLM](https://theaisingularity.org/self-attention-from-scratch/)
- 📄 [How to Implement Multi-Head Attention from Scratch in TensorFlow and Keras](https://machinelearningmastery.com/how-to-implement-multi-head-attention-from-scratch-in-tensorflow-and-keras/)

---

**Coming up on Day 29:** Positional Encoding