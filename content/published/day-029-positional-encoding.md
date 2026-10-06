---
day: 29
generated_at: '2026-09-28T17:07:38.831906+00:00'
phase: Phase 4 — The Transformer Revolution
recap_summary: Explained how positional encoding gives Transformers a sense of word
  order, described the difference between learnable and fixed (sinusoidal) encodings,
  and showed how sinusoidal positional encoding works and is combined with word embeddings.
status: published
title: 'Day 29: Positional Encoding'
topic_title: Positional Encoding
---

**Previously, on Day 28:** Explained multi-head attention as used in Transformer models, detailing how multiple attention heads enable the capture of diverse and overlapping patterns within input sequences through independent, parallel computations, culminating in a combined representation.

---

### What Positional Encoding Solves in Transformers

Transformers are powerful models for handling language, vision, and other sequence tasks. But they lack a built-in sense of order. Recurrent Neural Networks (RNNs) process sequences step by step, so word order is clear—“cat” before “sat” before “mat.” In contrast, Transformers process all tokens at once. Without extra help, “cat sat mat” and “mat cat sat” look the same to a Transformer. Each input word, or **token**, is treated independently unless we tell the model where it appears in the sequence.

To help Transformers understand that “The cat sat” is different from “Sat the cat,” we must add **order** to their inputs. This is the main job of **positional encoding**.

### The Principle of Positional Encoding

**Positional encoding** adds a unique marker to each word’s input vector. Instead of just giving the model the meaning of each word, we also attach information about its position in the sentence.

For example, with three words, we use `[word1 + pos1, word2 + pos2, word3 + pos3]`. Here, `word1` is the vector for the first word, and `pos1` is a vector encoding “I am in position 1.” Now, the model knows both what the word is and where it appears.

Even if "cat" and "mat" are similar words, this helps the model tell if they are at the start, middle, or end of the sequence.

### Common Approaches to Positional Encoding

Positional encodings are built in two common ways:

1. **Learnable Positional Embeddings:**  
   Each position in the sequence gets a trainable vector, similar to how each word gets its own embedding. During training, the network learns what pattern of positions works best for the task. For sentence positions 1, 2, 3, and so on, you look up a vector for each.

   *Intuition:* The model invents its own “language of order,” tuned to the specific task and dataset. But it only learns positions seen in training and may not handle longer or shifted sequences as gracefully.

2. **Fixed (Sinusoidal) Positional Encodings:**  
   The position vectors are calculated using sine and cosine waves at different speeds. Each position has a mathematical formula, so the model can handle any sequence length, even longer than it saw in training.

   *Intuition:* Each position is labeled with a unique “wave pattern.” Some waves are fast, picking out local order (neighboring words); others are slow, marking broad distances.

**Tradeoff:**  
Learned encodings adapt closely to the training data but struggle to generalize beyond it. Fixed encodings are more universal and work for arbitrary sequence lengths.

### Sinusoidal Positional Encoding: How It Works

A **sinusoidal positional encoding** creates a unique vector for each position by combining sine and cosine waves at different frequencies. You pick a vector length, such as 8 or 512. For every dimension `i` in the vector:

- For even indices:  
  `sin(position / (10000^(2i/dim)))`
- For odd indices:  
  `cos(position / (10000^(2i/dim)))`

- `position` is the word’s place in the sequence (starting from 0).
- `i` is the position within the vector.
- `dim` is the total size of the vector.

Low-numbered vector dimensions change quickly with each word (tracking short distances), while high-numbered dimensions change more slowly (tracking longer-range order). When plotted, these create a series of waves—some rapid, some slow—unique to every token position.

More than just telling the model “this is position 4,” these waves help it easily spot how far apart any two positions are. The numerical difference between two position vectors encodes their distance in the sequence.

### Using Positional Encoding in Practice

Practically, adding positional encodings is direct. You calculate a matrix of positional vectors (one for each token, matching the word embedding size), then **add** it to your word embeddings. This addition is elementwise: for each token, every value in its meaning vector gets the corresponding number from the position vector.

No new input fields. No special changes at inference. Just this formula:

`final_embedding = word_embedding + position_signal`

Nearly every Transformer architecture adds positional encoding immediately after the word embedding, right before the first attention layer.

Here is minimal, runnable code showing the sinusoidal encoding and combination with word embeddings:

```python
import numpy as np

def get_sinusoidal_positional_encoding(seq_len, d_model):
    PE = np.zeros((seq_len, d_model))
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            div_term = np.exp(i * -np.log(10000.0) / d_model)
            PE[pos, i] = np.sin(pos * div_term)
            if i + 1 < d_model:
                PE[pos, i + 1] = np.cos(pos * div_term)
    return PE

# Dummy inputs
seq_len, d_model = 5, 8
dummy_word_embeddings = np.random.randn(seq_len, d_model)
positional_encodings = get_sinusoidal_positional_encoding(seq_len, d_model)

# Combine: elementwise addition
inputs_with_positions = dummy_word_embeddings + positional_encodings

print("Word Embeddings:\n", dummy_word_embeddings)
print("Positional Encodings:\n", positional_encodings)
print("Combined Inputs to Transformer:\n", inputs_with_positions)
```

This is enough: build the position signals, add them in. Now, each token “carries” both its meaning and its spot.

### Intuitive Analogy: Musical Notes in Sequence

Think of music. A note is just a sound—G, D, or E. But a melody isn’t just which notes you play; it’s the order you play them in. If you don’t know which note comes first, the music is lost.

Positional encoding is like adding a timestamp or label to each note in a melody. It tells the performer the order and spacing between notes, even if the sheet is mixed up.

For Transformers, positional encoding gives every word a place in the “melody” of the sentence. The model gets both what each word means and where it fits. This is essential for any task where order matters—which is almost all of language.

---

## Key Takeaways

- Transformers need positional encoding to distinguish word order in a sequence.
- Learnable position embeddings adapt to data but may not generalize to longer sequences.
- Sinusoidal encodings use mathematical wave patterns to label token positions for any length.
- Adding positional encoding is simply elementwise addition to word embeddings before attention.

## Try It Yourself

Write code that generates sinusoidal positional encodings for a sequence length of 10 and embedding dimension of 8. Create random word embeddings of matching shape, add the positional encodings elementwise, and print the combined result. Observe how the positional encoding alters the original embeddings.

## Further Resources

- 🎥 [L‑5 | Positional Encoding in Transformers Explained](https://www.youtube.com/watch?v=kLHQGN6xmM0)
- 📘 [Positional Encoding in Transformers — Neuromatch Academy Tutorial](https://deeplearning.neuromatch.io/tutorials/W3D1_AttentionAndTransformers/student/W3D1_Tutorial1.html)
- 📄 [“Position Information in Transformers: An Overview” (survey paper)](https://arxiv.org/abs/2102.11090)

---

**Coming up on Day 30:** The Transformer Encoder Architecture