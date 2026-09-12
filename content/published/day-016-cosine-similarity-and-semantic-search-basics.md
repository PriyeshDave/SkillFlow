---
day: 16
generated_at: '2026-09-10T13:11:17.587154+00:00'
phase: Phase 2 — Word Representations & Embeddings
recap_summary: Explained cosine similarity, how it measures the direction-based similarity
  between word embedding vectors, and its use in semantic search to find the most
  meaningfully related items based on embeddings rather than exact keywords.
status: published
title: 'Day 16: Cosine Similarity and Semantic Search Basics'
topic_title: Cosine Similarity and Semantic Search Basics
---

**Previously, on Day 15:** Explained why visualizing word embeddings is useful, introduced the challenges of high-dimensional data, and compared two popular dimensionality reduction techniques—PCA and t-SNE—for plotting embeddings, with code examples for practical visualization.

---

### What Cosine Similarity Measures

Cosine similarity measures how closely two vectors point in the same direction, regardless of their length.

Picture a vector as an arrow from the origin in space. Cosine similarity looks at the angle between two such arrows. If they point almost the same way, the angle is small, and cosine similarity is close to 1. If they’re at right angles (90 degrees), similarity is 0. If they point in opposite directions, similarity is -1.

For word embeddings, this is important. Embeddings for similar words—like "dog" and "puppy"—point in similar directions, even if their lengths are different. Cosine similarity captures this. By contrast, Euclidean distance measures the straight-line distance between the tips of the arrows. That means two vectors can be far apart (large Euclidean distance) but still point in nearly the same direction (high cosine similarity).

In natural language processing (NLP), direction typically represents meaning. Magnitude is usually less important.

---

### How to Compute Cosine Similarity

Given two vectors, A and B:

- Cosine similarity = (A · B) / (||A|| * ||B||)
  - The dot product (A · B) multiplies each pair of components and adds up the results.
  - ||A|| means the norm (length) of vector A: take each component, square it, add them up, then take the square root.
  - The denominator normalizes by length, so only the angle matters.

Think of two people tossing darts from the center of a dartboard. If their darts go in nearly the same direction, cosine similarity is high—no matter how far they throw.

The result ranges from -1 (opposite directions) to 1 (same direction). For word embeddings, values tend to be above 0 unless the words are totally unrelated or opposites.

Why normalize? Without normalization, longer vectors would always seem more similar to everything else, regardless of actual direction.

#### Step-by-Step Example

Suppose:
- A = [1, 2]
- B = [2, 4]

1. **Dot product:** (1 × 2) + (2 × 4) = 2 + 8 = 10  
2. **Norm of A:** sqrt(1² + 2²) = sqrt(1 + 4) = sqrt(5) ≈ 2.236  
3. **Norm of B:** sqrt(2² + 4²) = sqrt(4 + 16) = sqrt(20) ≈ 4.472  
4. **Divide:** 10 / (2.236 × 4.472) ≈ 10 / 10 = 1

Score of 1: arrows point exactly the same way.

---

### Example: Comparing Words with Embeddings

Suppose you have simple 2D vectors for some words:

- “cat” = [1, 2]
- “dog” = [0.9, 2.1]
- “car” = [-2, 0.5]
- “apple” = [0.2, -1]

To find the word most similar to "cat," compute cosine similarities:

- cat vs dog: dot = (1 × 0.9) + (2 × 2.1) = 0.9 + 4.2 = 5.1  
  norms: sqrt(1²+2²) ≈ 2.236, sqrt(0.9²+2.1²) ≈ 2.293  
  similarity = 5.1 / (2.236 × 2.293) ≈ 5.1 / 5.12 ≈ 0.996

- cat vs car: dot = (1 × -2) + (2 × 0.5) = -2 + 1 = -1  
  norms: 2.236, sqrt(4 + 0.25) ≈ 2.06  
  similarity = -1 / (2.236 × 2.06) ≈ -1 / 4.605 ≈ -0.217

- cat vs apple: (1 × 0.2) + (2 × -1) = 0.2 - 2 = -1.8  
  norms: 2.236, sqrt(0.2² + 1²) = sqrt(0.04 + 1) ≈ 1.02  
  similarity = -1.8 / (2.236 × 1.02) ≈ -1.8 / 2.28 ≈ -0.789

“Dog” is almost perfectly aligned with “cat.” “Car” and “apple” are much less similar, with negative scores showing they point in different or even opposite directions.

---

### Semantic Search Basics

Traditional keyword search matches exact words or phrases.

Semantic search uses embeddings—vectors that represent words, sentences, or documents. It finds items whose embeddings are close in direction (high cosine similarity). This means you can find results that match the meaning, not just the words.

Example:  
When searching "quick animal," a keyword search falls short. A semantic search matches "fast dog," "speedy fox," or "cheetah" because their embeddings are nearby in vector space.

Semantic search works at many levels:
- **Word-level:** Find synonyms or related words.
- **Sentence/document-level:** Retrieve passages with related meaning, even if phrased differently.

The key mechanism is comparing embedding vectors, almost always with cosine similarity.

---

### Real-World Example: Simple Semantic Search in Python

Here’s a minimal Python function to find the most semantically similar words to a query, using cosine similarity and small sample embeddings:

```python
import numpy as np

def most_similar(query_word, vocab, embeddings, top_n=3):
    # Get vector for the query word
    if query_word not in vocab:
        raise ValueError('Query word not in vocabulary!')
    query_vec = embeddings[vocab.index(query_word)]
    # Normalize query vector
    query_vec = query_vec / np.linalg.norm(query_vec)
    similarities = []
    for word, vec in zip(vocab, embeddings):
        # Normalize each embedding
        vec_norm = vec / np.linalg.norm(vec)
        # Cosine similarity (since normalized)
        sim = np.dot(query_vec, vec_norm)
        similarities.append((word, sim))
    # Sort by similarity, descending, skip the query word itself
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    return [w for w, s in similarities if w != query_word][:top_n]

# Example usage:
vocab = ['cat', 'dog', 'car', 'apple']
embeddings = np.array([
    [1, 2],      # cat
    [0.9, 2.1],  # dog
    [-2, 0.5],   # car
    [0.2, -1],   # apple
])

print(most_similar('cat', vocab, embeddings, top_n=2))
# Output: ['dog', 'car']
```

This function:
- Looks up the queried word’s vector.
- Computes cosine similarity with every other word.
- Returns the top-N most similar words by meaning.

The same logic, scaled up, forms the engine of modern semantic search systems. The only difference is bigger vectors and much larger vocabularies.

---

## Key Takeaways

- Cosine similarity measures the angle between vectors, ignoring magnitude.
- Word embeddings with similar meaning point in similar directions in vector space.
- Semantic search uses cosine similarity to match meanings, not just keywords.
- Negative cosine scores indicate opposite or unrelated meanings between embeddings.
- Cosine similarity is commonly used for comparing sentences, documents, or words in NLP.

## Try It Yourself

Create an array of at least four sample word embeddings—either randomly generated or manually assigned—and define a corresponding vocabulary list. Write a short Python script that, given any word in your vocabulary, computes cosine similarity with all other words and prints out the three most semantically similar words. Try this for at least two different query words.

## Further Resources

- 🎥 [Semantic Search using Embeddings in Python | Cosine Similarity Explained](https://www.youtube.com/watch?v=vu2gtfe2oHo)
- 🎥 [Semantic Search Fundamentals (Video)](https://mixpeek.com/education/videos/semantic-search-fundamentals)
- 📄 [How to Build Semantic Search with Embeddings: a Practical Walkthrough](https://www.andergrove.com/tools/cosine-similarity/guide/)
- 📘 [Let’s Understand Search — Qdrant (Module 1)](https://qdrant.tech/course/beginners/module-1/)
- 📄 [Embeddings and Semantic Search — Praveen T N](https://praveentn.live/learn/concepts/embeddings-semantic-search)

---

**Coming up on Day 17:** Neural Networks Refresher for NLP Engineers