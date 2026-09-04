---
day: 11
generated_at: '2026-09-03T13:04:47.414643+00:00'
phase: Phase 2 — Word Representations & Embeddings
recap_summary: Explained one-hot encoding as a method for representing words as sparse
  vectors, highlighted its limitations regarding similarity and efficiency, and introduced
  the concept of word embeddings as an improvement.
status: pending_review
title: 'Day 11: Why One-Hot Encoding Fails: The Case for Embeddings'
topic_title: 'Why One-Hot Encoding Fails: The Case for Embeddings'
---

What is One-Hot Encoding?

Computers need numbers, not words. To process language, we first turn words into numbers. One of the earliest ways to do this is **one-hot encoding**.

A one-hot encoding represents each word in your vocabulary with a vector—a list of numbers—whose length matches the vocabulary size. All entries in this list are zero, except for a single one: the “hot” bit, which marks the position of that word in your list.

This approach is simple to compute. It lets you plug words into early machine learning algorithms. One-hot encoding was a workhorse for the first language models.

Example: One-Hot Vectors

Suppose your vocabulary is just three animals: `cat`, `dog`, and `fish`. Assign each a position:

- `cat` is at position 0  
- `dog` is at position 1  
- `fish` is at position 2

Their one-hot encodings are:

- `cat` → [1, 0, 0]  
- `dog` → [0, 1, 0]  
- `fish` → [0, 0, 1]

If you see `[0, 1, 0]`, you know it’s `dog` because only the middle spot is “hot.”

Drawbacks of One-Hot Encoding

Now scale up: your vocabulary has tens of thousands of words. Each word becomes a long vector with mostly zeros and a single one.

This causes several issues:

**1. No Sense of Similarity**

One-hot vectors don’t reflect meaning or relationships. For the computer, `cat` and `dog` look just as unrelated as `cat` and `fish`. There’s no sense that dogs and cats are both pets or mammals.

- The **dot product**—a way to measure how much two vectors point in the same direction—is zero for any pair of different one-hot vectors.  
- The **distance** between any two distinct one-hot vectors is always the same.

Example in code:

```python
import numpy as np

# Vocabulary and lookup
vocab = ['cat', 'dog', 'fish']
word_to_index = {word: idx for idx, word in enumerate(vocab)}

# Function to create a one-hot vector
def one_hot(word):
    vec = np.zeros(len(vocab))
    vec[word_to_index[word]] = 1
    return vec

cat_vec = one_hot('cat')
dog_vec = one_hot('dog')
fish_vec = one_hot('fish')

# Dot products
print(np.dot(cat_vec, dog_vec))   # 0.0
print(np.dot(cat_vec, fish_vec))  # 0.0

# Euclidean distances
print(np.linalg.norm(cat_vec - dog_vec))   # 1.414...
print(np.linalg.norm(cat_vec - fish_vec))  # 1.414...
```

No matter which two words you compare, the numbers are identical. One-hot vectors can't show that some words are more related than others.

**2. Inefficient Representation**

Real vocabularies are large. Each one-hot vector is mostly zeros—these are called **sparse vectors**. If you have 10,000 words, each word uses a vector 10,000 entries long. Storing these wastes memory and computation.

Why This Matters

Modern language tasks depend on recognizing relationships between words. Search engines, translation systems, and sentiment analysis systems need to recognize that words like `happy` and `joyful` are connected in meaning.

With one-hot encoding, swapping a word for a synonym changes the vector completely. The computer sees no connection. This makes it hard for models to learn about patterns, context, or subtle meanings.

If a system can’t detect that two words mean nearly the same thing, it will struggle to find similar documents, suggest synonyms, or identify related concepts.

A Glimpse at Embeddings

To address these problems, we use **embeddings**.

Embeddings also represent words as vectors. But instead of being all zeros except for one, embeddings use vectors filled with real values—often dozens or hundreds of numbers per word. Words with similar meanings get vectors that are close together. `Cat` and `dog` embeddings are similar; `cat` and `fish` are not as close.

This lets computers detect patterns, similarity, and context. You’ll learn how embeddings work, and how they power modern language models, in upcoming lessons. For now: embeddings let machines measure similarity and capture relationships between words, something one-hot encoding can’t do.

---

## Key Takeaways

- One-hot encoding maps each word to a unique sparse vector with a single one.
- All one-hot vectors are equally distant; they show no word similarity.
- One-hot representations become inefficient with large vocabularies.
- Embeddings create dense vectors that capture relationships and similarity between words.

## Try It Yourself

Choose five words and assign each a position in your list. Write their one-hot vector representations either by hand or in code. Compare the vectors for each word pair—are any pairs more similar than others? Explain your findings.

## Further Resources

- 🎥 [What Are Word Embeddings?](https://www.youtube.com/watch?v=hVM8qGRTaOA)
- 🎥 [why dense embeddings outshine one‑hot vector encoding: Intuitive explanation](https://www.youtube.com/watch?v=chF3BoGeSG8)
- 📄 [Embeddings | Transformer 101](https://www.transformer101.com/embeddings)
- 📄 [Embedding Layer | Neel Mishra](https://neelmishra.github.io/blog/dl/transformers/building-blocks/embedding-layer.html)
- 📄 [Word Representations and the Foundations of Word Embeddings](https://insightful-data-lab.com/2025/07/15/word-representations-and-the-foundations-of-word-embeddings/)

---

**Coming up on Day 12:** Word2Vec Explained: Skip-Gram and CBOW