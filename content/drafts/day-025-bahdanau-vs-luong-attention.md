---
day: 25
generated_at: '2026-09-22T13:44:23.960274+00:00'
phase: Phase 3 — Sequence Models & Deep Learning for NLP
recap_summary: Explained the concept of attention in sequence models, detailing how
  Bahdanau (additive) and Luong (multiplicative) attention mechanisms work to let
  decoders focus on relevant encoder outputs, with step-by-step breakdowns and code
  examples for both approaches.
status: pending_review
title: 'Day 25: Bahdanau vs. Luong Attention'
topic_title: Bahdanau vs. Luong Attention
---

### What Is Attention in Sequence Models?

Attention is a technique that allows neural networks to focus on the most relevant parts of their input when generating each output, instead of giving every input element equal weight. 

Think of translating a sentence from French to English. When you produce each English word, your focus shifts to the French words most closely related to what you’re writing—like focusing on verbs when translating a verb. You don’t need to keep the entire sentence in mind at every step.

In natural language processing (NLP), attention enables a model to mimic this selective focus. Rather than squeezing a whole sentence into a single summary vector, attention lets the model “look back” and weigh parts of the input differently for each output word.

### The Problem: Seq2Seq Bottlenecks

Basic sequence-to-sequence (seq2seq) models use two parts:

- An **encoder** turns an input sequence (like a sentence) into a single summary vector.
- A **decoder** converts that vector into an output sequence (such as a translation).

This approach works for short sequences but stumbles on longer ones. The model tries to cram all meaning into one vector, losing details. For long sentences, important words fade. The model may lose track of “who did what to whom.”

Attention allows the decoder to refer directly to any part of the encoder’s output at each output step. This sidesteps the compression bottleneck.

### Bahdanau Attention: The First "Soft" Attention

Bahdanau attention, also called **additive attention**, was the first widely-used attention mechanism in NLP. Its key idea: for each word the decoder produces, the model decides how much to focus on each word from the input.

Suppose the encoder processes a sentence, producing a **hidden state** for each word:  
`h₁, h₂, ..., hₙ` (each h is a vector summarizing a word’s context).

The decoder also maintains a **hidden state** at each output step, labeled `sₜ`.

Bahdanau attention asks: for the current output word, which encoder states are most relevant? It answers this by assigning an **alignment score** to each encoder state, reflecting how well it matches the decoder’s current state.

Unlike a simple similarity (like dot product), Bahdanau attention uses a small neural network (a single feedforward layer with tanh activation) to compute each score. 

These scores, once normalized (adding up to 1 using softmax), become weights. The model computes a **weighted average** of all encoder hidden states using these weights—a new **context vector**. This vector provides the decoder with focused information from the encoder, tailored to each output step.

### Bahdanau Attention Step-by-Step

At each output step, Bahdanau attention goes through:

1. **Score Calculation:**  
   For every encoder hidden state `hᵢ`, combine it with the decoder state `sₜ` to get a score:
   ```
   scoreᵢ = vᵀ tanh(W₁ hᵢ + W₂ sₜ)
   ```
   Here, `W₁`, `W₂`, and `v` are trainable weights. `tanh` is a nonlinear function that limits outputs to a range between -1 and 1.

2. **Softmax Normalization:**  
   Convert all the raw scores into attention weights (probabilities), ensuring they sum to 1:
   ```
   αᵢ = softmax(scoreᵢ)
   ```

3. **Context Vector Computation:**  
   Use attention weights to take the weighted average of encoder hidden states:
   ```
   context = Σ (αᵢ * hᵢ)
   ```
   The context vector is then fed to the decoder to help generate the next output word.

### Luong Attention: A Simpler, Faster Variant

Luong attention, also called **multiplicative attention**, simplifies and speeds up the scoring step. It makes two main changes:

- **When attention is applied:**  
  Bahdanau applies attention before the decoder updates its hidden state; Luong applies it after. This changes the flow of information but not the core idea.

- **How scores are computed:**  
  Luong offers simpler score functions. Instead of a feedforward network, it uses one of:
  - **Dot product:** Directly measures similarity between encoder and decoder hidden states.
  - **General:** Like dot, but introducing a learned weight matrix.

  In notation:
  ```
  scoreᵢ = sₜᵀ W hᵢ    # general
  scoreᵢ = sₜᵀ hᵢ      # dot
  ```

Like Bahdanau, the softmax normalization and context vector calculation remain the same.

### Comparing Bahdanau and Luong Attention

| Feature         | Bahdanau (Additive)              | Luong (Multiplicative)         |
|-----------------|----------------------------------|-------------------------------|
| Scoring         | Neural net with tanh & weights   | Dot/general (matrix multiply)  |
| Computation     | Before decoder update            | After decoder update           |
| Efficiency      | More parameters, slower          | Fewer parameters, faster       |
| Common Usage    | Early NLP sequence tasks         | Later, scalable models         |

**Choosing an approach:**  
Bahdanau’s method is more flexible for complex tasks and small datasets. Luong’s version is faster and better for large batches or long inputs.

### Code Example: Bahdanau and Luong Attention in PyTorch

Below is a minimal example showing Bahdanau and Luong attention side by side. You can see how scores and context vectors are computed.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Dummy data: batch size 1, 5 input steps, hidden size 4
encoder_outputs = torch.randn(1, 5, 4)    # (batch, seq_len, hidden)
decoder_hidden = torch.randn(1, 4)        # (batch, hidden)

### Bahdanau Attention (additive) ###
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.W1 = nn.Linear(hidden_size, hidden_size)
        self.W2 = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1)

    def forward(self, encoder_outputs, decoder_hidden):
        seq_len = encoder_outputs.size(1)
        # Expand decoder hidden to shape (batch, seq_len, hidden)
        dec = decoder_hidden.unsqueeze(1).repeat(1, seq_len, 1)
        score = self.v(
            torch.tanh(self.W1(encoder_outputs) + self.W2(dec))
        ).squeeze(-1)  # (batch, seq_len)
        attn_weights = F.softmax(score, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return attn_weights, context

bahdanau = BahdanauAttention(hidden_size=4)
bahdanau_weights, bahdanau_context = bahdanau(encoder_outputs, decoder_hidden)
print("Bahdanau attention weights:", bahdanau_weights)
print("Bahdanau context vector:", bahdanau_context)

### Luong Attention (dot) ###
class LuongAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, encoder_outputs, decoder_hidden):
        # encoder_outputs: (batch, seq_len, hidden)
        # decoder_hidden: (batch, hidden)
        attn_weights = torch.bmm(
            encoder_outputs, decoder_hidden.unsqueeze(-1)
        ).squeeze(-1)  # (batch, seq_len)
        attn_weights = F.softmax(attn_weights, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return attn_weights, context

luong = LuongAttention()
luong_weights, luong_context = luong(encoder_outputs, decoder_hidden)
print("Luong attention weights:", luong_weights)
print("Luong context vector:", luong_context)
```

Notice the difference:

- Bahdanau attention uses a small neural network for scoring, which gives it extra flexibility at a computational cost.
- Luong attention (dot version) compares encoder and decoder states directly with a dot product—no extra parameters, so it's faster but less expressive.

Try adjusting the dummy data and see how attention weights change as the decoder state moves. This testbed makes the core ideas behind sequence model attention visible and tangible.

---

## Key Takeaways

- Attention lets sequence models focus on relevant input parts for each output step.
- Bahdanau attention computes alignment scores using a small neural network, adding expressiveness.
- Luong attention uses simpler dot products for scoring, making it faster and more efficient.
- Both attention mechanisms produce context vectors as weighted averages of encoder outputs.
- Choosing between Bahdanau and Luong depends on dataset size, task complexity, and efficiency needs.

## Try It Yourself

Using the following numpy arrays for encoder hidden states and a decoder hidden state, manually calculate: (1) the Bahdanau alignment scores and attention weights, (2) the Luong dot-product attention scores and weights, and (3) the resulting context vector in each case. Write your computations as explicit arithmetic or code, and compare how the two methods distribute attention.

## Further Resources

- 🎥 [Attention Mechanism (Bahdanau Attention & Luong Attention) | Deep Learning](https://www.youtube.com/watch?v=uaDIK9mYa1U)
- 📄 [Differences Between Luong Attention and Bahdanau Attention | Baeldung](https://www.baeldung.com/cs/attention-luong-vs-bahdanau)
- 📄 [Bahdanau vs Luong Attention: Which One Should You Actually Use? (Spoiler: Luong)](https://www.sotaaz.com/post/attention-mechanism-implementation-en)
- 📄 [Effective Approaches to Attention-based Neural Machine Translation (Luong et al., 2015)](https://arxiv.org/abs/1508.04025)

---

**Coming up on Day 26:** "Attention Is All You Need": The Paper That Changed Everything