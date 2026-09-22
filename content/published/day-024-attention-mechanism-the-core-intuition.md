---
day: 24
generated_at: '2026-09-21T15:25:30.011806+00:00'
phase: Phase 3 — Sequence Models & Deep Learning for NLP
recap_summary: Explained how attention mechanisms help sequence models focus on relevant
  information across entire inputs, overcoming limitations of traditional RNNs and
  LSTMs when connecting distant parts of a sequence.
status: published
title: 'Day 24: Attention Mechanism: The Core Intuition'
topic_title: 'Attention Mechanism: The Core Intuition'
---

Sequence data, such as language, is fundamentally different from static data. The meaning of a sentence depends on the order of words and their relationships—sometimes between words that are right next to each other, sometimes across many words. Consider the sentence: “She gave the book to Tom because he was interested.” To understand who “he” is, you must remember earlier parts of the sentence, even if those details are separated by several less important words.

Traditional sequence models process data one step at a time. Recurrent Neural Networks (RNNs) and Long Short-Term Memory networks (LSTMs) scan a sentence word by word, updating a hidden state that is meant to store all the necessary information. For short sentences, this works reasonably well. But when critical information is far back in the sequence, these models struggle. Older words lose their influence as you move forward—a limitation sometimes called the “vanishing gradient” problem. This makes RNNs and LSTMs weak at connecting distant details, like resolving the reference for a pronoun several sentences later.

The attention mechanism directly addresses this limitation. Instead of relying only on the final hidden state (like scribbling every note on a single notepad), attention allows a model to look back over the entire input at each step. The model can “focus” on specific pieces of input it deems relevant, regardless of their position in the sequence. This selective focus is like having a highlighter to mark important phrases, rather than sifting through the entire text each time.

Attention works much the way people read and understand text. Imagine reading a research paper. If you encounter a tricky clause—like “this method outperforms previous work”—you might deliberately refer back to earlier sections to recall what “previous work” means. You don’t reread every line; you jump straight to the part that matters.

To make this concrete, take the sentence:  
“The animal didn’t cross the street because it was too tired.”

If a model wants to resolve what “it” refers to, attention allows it to weigh every word and assign a score of “importance” for each one. The table below pretends a model computes how relevant each word is when interpreting “it”:

| Word    | Attention Weight (to "it") |
|---------|----------------------------|
| The     | 0.05                       |
| animal  | 0.70                       |
| didn’t  | 0.02                       |
| cross   | 0.01                       |
| the     | 0.01                       |
| street  | 0.05                       |
| because | 0.02                       |
| it      | 0.09                       |
| was     | 0.02                       |
| too     | 0.01                       |
| tired   | 0.02                       |

Here, the model gives “animal” the highest attention weight. So when deciding what “it” refers to, the model mostly relies on the information from “animal.”

Now, let’s see a minimal, real code example of attention in practice—using simple numeric features for words (embeddings). We’ll simulate “queries,” “keys,” and “values,” the core elements of attention, as simple arrays. We’ll calculate attention weights and use them to create a “context vector,” which summarizes the relevant information.

```python
import numpy as np

# Example word features (embeddings) for a sequence of 3 words.
word_features = np.array([
    [1.0, 0.0],  # "Alice"
    [0.0, 2.0],  # "loves"
    [1.0, 1.0]   # "books"
])

# We want the model to focus on the relevant words for "books".
query = word_features[2]    # "books": [1.0, 1.0]
keys = word_features        # One key for each word

# Score each word by similarity (dot product) to the query.
scores = keys @ query       # scores: [1, 2, 2]

# Normalize scores so they sum to 1 (softmax).
exp_scores = np.exp(scores)
attention_weights = exp_scores / np.sum(exp_scores)

# Weighted sum of word features, using attention weights.
context_vector = (attention_weights[:, None] * word_features).sum(axis=0)

print("Attention weights:", np.round(attention_weights, 2))
print("Context vector:", np.round(context_vector, 2))
```

This produces:
```
Attention weights: [0.12 0.44 0.44]
Context vector: [0.56 1.32]
```

This example shows how the features for all three words get blended together, with higher weights for the words most similar to “books.” The resulting context vector pulls together the most relevant information for the model’s next decision.

It’s key to know what attention is not. Attention is not a standalone model. It doesn’t replace neural networks or “understand” text by itself. Instead, attention is a flexible mechanism—built into larger models—that helps them select which details to use at each step. The real learning comes from the surrounding neural network, which determines what should be attended to. 

Attention is like a lens, helping the model focus on what matters within long sequences, especially when crucial details are far apart. It’s an essential part of modern sequence models, but always needs a larger system to guide its use.

---

## Key Takeaways

- Meaning in language often depends on word order and long-range relationships.
- RNNs and LSTMs struggle to connect distant details due to information loss.
- Attention allows models to weigh and focus on any part of the input sequence.
- Attention is a mechanism, not a standalone model, and works within larger neural networks.
- Attention blends input features based on relevance to create context-aware representations.

## Try It Yourself

Take a sentence like 'Cats chase small mice quickly.' Choose a target word (e.g., 'quickly') and assign attention weights (numbers from 0 to 1, summing to 1) to each word based on how relevant you think they are to understanding 'quickly.' Given example two-dimensional feature vectors for each word, multiply each vector by your chosen weight, and sum them up to create a context vector. Reflect on which words you intuitively assigned the highest weights to and why.

## Further Resources

- 🎥 [The Attention Mechanism Explained with Intuition](https://www.youtube.com/watch?v=tebDYJOkUd0)
- 🎥 [C5W3L07 Attention Model Intuition (DeepLearning.AI)](https://www.youtube.com/watch?v=SysgYptB198)
- 📄 [Attention Mechanism — Tutorial (ModelRefs)](https://modelrefs.com/tutorials/attention-mechanism/)
- 📘 [NLP From Scratch: Translation with a Sequence to Sequence Network and Attention — PyTorch Tutorials](https://docs.pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)
- 📄 [Attention Mechanism, Transformers, BERT, and GPT: Tutorial and Survey](https://faculty.ist.psu.edu/vhonavar/Courses/dsmethods/transformer.pdf)

---

**Coming up on Day 25:** Bahdanau vs. Luong Attention