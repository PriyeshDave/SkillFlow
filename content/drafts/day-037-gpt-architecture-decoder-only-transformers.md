---
day: 37
generated_at: '2026-10-14T14:07:49.002616+00:00'
phase: Phase 5 — Pretraining & Language Models
recap_summary: Explained the three main transformer architectures—encoder-only, decoder-only,
  and encoder-decoder—with a focus on how decoder-only models like GPT perform autoregressive
  text generation using masked (causal) self-attention.
status: pending_review
title: 'Day 37: GPT Architecture: Decoder-Only Transformers'
topic_title: 'GPT Architecture: Decoder-Only Transformers'
---

**Previously, on Day 36:** Explained the concept and process of fine-tuning BERT on downstream NLP tasks, specifically illustrating how BERT can be adapted for text classification using labeled data and transfer learning principles.

---

Transformer models have changed how computers understand and generate language. At the heart of transformers are three main structures: encoder-only, decoder-only, and encoder-decoder. Understanding each helps you see how modern models like GPT actually work.

## Transformer Architectures: Encoder, Decoder, and Encoder-Decoder

A **transformer** is a neural network designed to handle sequences—such as sentences or paragraphs. It builds up meaning using two key building blocks: *attention* layers and *feedforward* layers.

**Encoder-only transformers** process a whole input sequence at once. They build a detailed internal summary of the input. For example, with the sentence “Apples are red,” the encoder reads the entire thing together and produces a transformed version of each word. BERT is a famous encoder-only model. Encoders work well for tasks like classifying text or picking out details, because they can see all parts of the input at once, both left and right of each token.

**Decoder-only transformers** focus on generating text. The key difference: when producing the next word (or *token*), each position sees only itself and the tokens before it—not anything after. This setup is called *autoregressive* generation. For example, given the start of a sentence, the model predicts the next word, then uses that word to predict the next one, moving left to right.

**Encoder-decoder transformers** combine both ideas. The encoder summarizes an input sequence. The decoder uses this summary to generate an output sequence, one token at a time. This structure is essential for tasks where the output depends directly on the input, like translation or summarization. T5 and most translation models use an encoder-decoder setup.

## The Decoder Block: Building Autoregressive Models

GPT uses stacks of *decoder blocks* to generate text. Each decoder block performs three steps:

1. **Masked Self-Attention:** Every token “looks back” at itself and the tokens before it—never tokens ahead. This gathers existing context without letting future information slip in.
2. **Feedforward Network:** The model refines each token’s representation using a small neural network. Each token is transformed independently here.
3. **Residual Connections and Layer Normalization:** These pieces help training by stabilizing the network and making learning easier.

Here’s the flow:

- Input tokens are converted to vectors called *embeddings*.
- Each token passes through masked self-attention. The mask makes sure no token sees the future.
- The result is smoothed via normalization.
- The vector is refined by the feedforward network, then normalized again, with a shortcut (residual connection) back to the input.
- After several decoder blocks, the topmost output is used to predict the next token.

Stacking multiple decoder blocks lets the model detect more complex patterns as it generates each new token.

## Why Decoder-Only? GPT vs BERT and T5

GPT is built from only the decoder parts—specifically, the masked self-attention pieces. This design choice matches its purpose: generating text one token at a time.

In BERT (encoder-only), the model sees the entire sentence from the start and can use information from both earlier and later tokens for each prediction. That’s ideal for understanding complete texts or labeling each word based on the whole context. Sequence order for output is less important here.

But for generating text, seeing the future would let the model “cheat”—it could just copy future words from the prompt. GPT’s decoder-only design ensures the model learns to predict the next word based only on the past and present, not the future.

Encoder-decoder models like T5 are better for tasks that require closely relating input and output, such as translation. For pure language generation, decoder-only models are faster and more efficient.

## Causal Self-Attention: Enforcing Left-to-Right Generation

*Causal self-attention* makes sure the model only attends to information up to the current token. For every position, the model can see itself and everything to the left—never anything to the right.

Example: Input sequence “I enjoy sushi”

- When predicting “enjoy”, the model accesses “I” and “enjoy” but not “sushi”.
- For “sushi”, the model sees “I”, “enjoy”, and “sushi”.

This restriction is enforced with an **attention mask**. It’s a matrix that marks out future positions as “off limits.” Those positions are set so low that, after the attention math, they don’t contribute anything. So each token truly only sees the tokens before it and itself.

By contrast, standard self-attention (used in encoder models like BERT) allows every token to see every other token, in any position.

## From Input to Output: How GPT-Style Models Generate Text

Here is how a decoder-only model, like GPT, produces text one step at a time:

1. **Start with a prompt:** For example: “The weather is”
2. **Embed tokens:** Convert each word to an internal vector (embedding).
3. **Run through decoder blocks:** For each token, masked self-attention combines information from itself and the left context. The feedforward layers further refine these vectors.
4. **Predict next token:** The model assigns probabilities to every word in its vocabulary. It picks the most likely (or samples from the top choices).
5. **Repeat:** Add the new token to the input and repeat the process to get the following word.

This token-by-token approach is called *autoregressive* generation. The model always “looks back” at what it’s already written and never ahead.

## Code Example: Causal Masking in PyTorch

Below is a working example of how to build a causal mask for self-attention with PyTorch’s `nn.MultiheadAttention`:

```python
import torch
import torch.nn as nn

# Sequence length of 4, embedding size of 8
seq_len = 4
d_model = 8

# Random example data: shapes (sequence length, batch size, embedding dim)
x = torch.randn(seq_len, 1, d_model)

mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=2)

# Build a causal mask:
# It's "True" where attention should be *blocked* (future positions)
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()

# Forward pass with the mask; masked positions will be ignored in attention
output, attn_weights = mha(x, x, x, attn_mask=mask)

print("Masked attention weights:\n", attn_weights)
```

In this code:
- `x` is your sequence as embeddings.
- The mask is built with `torch.triu`: everything above the diagonal is set as `True`, blocking attention to future tokens.
- When running the model, attention weights for each token ignore everything to the right, following the causal (left-to-right) pattern.

This mechanism ensures every generated token depends only on information "up to now" and never on what comes after, just as required for GPT-style, autoregressive language modeling.

---

## Key Takeaways

- Encoder-only transformers process entire input sequences at once and are ideal for classification and understanding tasks.
- Decoder-only transformers generate text by predicting each token based only on previous tokens, enforcing causality.
- Encoder-decoder architectures combine both approaches for tasks like translation where input directly influences output.
- Causal masking prevents tokens from attending to future positions, ensuring left-to-right generation in GPT-style models.

## Try It Yourself

Modify the provided PyTorch code example by setting the attention mask to None, allowing each token to attend to every other token. Run the code and observe the resulting attention weights compared to the original causal mask setup. Write a brief note about how removing the causal mask changes the model's behavior and what this teaches about the need for causal self-attention.

## Further Resources

- 🎥 [How GPT Actually Works: Transformer Decoder Explained Visually](https://www.youtube.com/watch?v=KE9fqU4EG4o)
- 📄 [A Walk With Embeddings: A Visual Guide to a Decoder‑only Transformer](https://huggingface.co/blog/velmen/a-visual-introduction-to-decoder-only-transformer)
- 📘 [Section 3.2: Transformer Init, Causal Mask & Forward Pass (LLM Book)](https://llmbook.icsgen-ai.org/part-1-llm-building-blocks/module-03-transformer-architecture/section-3.2.html)
- 📄 [Mastering Decoder‑Only Transformers: Architecture, Training, Inference, and Modern LLM Design](https://www.itechguides.com/mastering-decoder-only-transformers-architecture-training-inference-and-modern-llm-design/)

---

**Coming up on Day 38:** GPT-1 to GPT-3: The Evolution of Scale