---
day: 13
generated_at: '2026-09-07T14:32:55.235873+00:00'
phase: Phase 2 — Word Representations & Embeddings
recap_summary: Explained the motivation and workings of GloVe word embeddings, including
  how they leverage both local context and global word co-occurrence statistics, described
  the model's loss function and practical use of pretrained vectors, and compared
  GloVe to Word2Vec in terms of methodology and use cases.
status: pending_review
title: 'Day 13: GloVe: Global Vectors for Word Representation'
topic_title: 'GloVe: Global Vectors for Word Representation'
---

**Previously, on Day 12:** Explained how Word2Vec models generate word embeddings using skip-gram and CBOW techniques, compared their methods and trade-offs, and introduced a minimal example using Gensim.

---

## Why GloVe? The Motivation Behind Global Word Embeddings

Word2Vec is a powerful method for learning word vectors. It trains these vectors so that words appearing in similar *local contexts*—words that show up near the same neighbors—end up close together in the vector space. The model does this by predicting a word from its neighbors (the CBOW method) or by predicting a word’s neighbors from itself (the Skip-gram method). All training happens inside narrow “windows” of text.

Focusing only on local context leaves gaps. If two words never appear close together in those windows but have similar broad patterns across the corpus, Word2Vec might miss that relationship.

GloVe (Global Vectors for Word Representation) addresses this limitation. GloVe captures both local context and *global statistics*—patterns from the entire dataset. Instead of just “Who is next to ‘king’ in a sentence?”, GloVe asks: “Across everything, how often do ‘king’ and ‘queen’ co-occur with the same words? How does this compare to ‘king’ and ‘man’?” GloVe uses information from all word pairs across the dataset, not just small windows, to learn these connections.

The main insight is that *how often* two words appear together, compared to how often they appear apart, reveals word meaning. For example, if “ice” appears with “cold” much more than “steam” does—and “steam” with “hot” much more than “ice”—those ratios reveal important facts about the words.

## How GloVe Works: Counting Co-occurrences and Building Vectors

GloVe builds on counting global word co-occurrences. Here’s the basic process:

1. **Build a vocabulary:** For example, pick the top 400,000 most frequent words in your data.
2. **Count co-occurrences:** For every word pair (`i`, `j`), count how many times they appear near each other inside a window (say, five words on each side). This creates a *co-occurrence matrix*.
    - Example: If "cat" appears within five words of "pet" 350 times, then the cell ("cat", "pet") in the matrix contains 350.
3. **Compute statistics:** For each word pair, you know: “Given word `i`, how likely is word `j` nearby?” You can turn this into a frequency, probability, or ratio.
4. **Learn word vectors:** Train each word’s vector so relationships between vectors match these statistics (details in the next section).

Think of the co-occurrence matrix as an enormous spreadsheet. Each row and column is a word; each cell tells you how many times those words appear near each other anywhere in your data. (For efficiency, GloVe usually works just with the nonzero counts.)

## The GloVe Objective: What the Model Learns

What does GloVe actually *optimize*? The model tries to make the dot product of two word vectors (plus bias terms) match the logarithm of their co-occurrence count. In other words, the geometry of the vectors should reflect the global counts from your data.

Here’s the simplified loss function:

\[
J = \sum_{i,j=1}^{V} f(X_{ij})\, \left(w_i^T \cdot w_j + b_i + b_j - \log(X_{ij})\right)^2
\]

Term by term:

- `X_{ij}`: How often word `j` appears near word `i` in the corpus.
- `w_i` and `w_j`: Vectors for words `i` and `j`.
- `b_i` and `b_j`: Bias values for each word—just single numbers.
- `f(X_{ij})`: A weighting function so rare or extremely frequent pairs don’t overly affect the loss.

For each pair of words, the model pushes their dot product (plus biases) to be as close as possible to the log of their co-occurrence count.

**Concrete Example:**
- If "ice" and "cold" appear together 500 times (`X_{ice,cold}=500`) and "ice" and "steam" only 12 times, then `log(500)` is about 6.2 and `log(12)` about 2.5.
- GloVe trains the vectors for "ice" and "cold" so their dot product (+ biases) is close to 6.2, and "ice" + "steam" near 2.5.
- If the model achieves this match for all word pairs, the words’ positions in vector space encode much of the useful statistical information in the corpus.

Why use logs and not raw counts? Logarithms compress a huge range of counts into more manageable values and make optimization easier.

## GloVe Embeddings in Practice: Downloading and Using Pretrained Vectors

Most people use *pretrained* GloVe vectors. These are available for English (and some other languages) from datasets like Wikipedia, Common Crawl (large web scrape), and news sources.

A typical GloVe embedding file is a text file. Each line contains a word and its vector of numbers:
```
word_1 0.418 0.24968 ... -0.68568
word_2 0.123 -0.289 ... 1.459
...
```

Pretrained GloVe sets most commonly come in 50, 100, 200, or 300 dimensions. Vocabularies are very large—up to 400,000 words. You rarely need all of them; load only what you need for your project. The file format is simple and works almost interchangeably with Word2Vec’s text format.

**When are pretrained GloVe files useful?**
- When you need word vectors quickly without heavy computation.
- When working in restricted data environments (medical, legal, or proprietary data), you can fine-tune these vectors on your domain, instead of training from scratch.

## Quickstart: Loading and Using GloVe Embeddings in Python

Below is a simple script to download a small GloVe file, load the vectors, and compute the similarity between two words.

```python
import numpy as np
import urllib.request
import zipfile
import os

# Download a small GloVe embedding set (50 dimensions, ~70MB zipped)
glove_url = "http://nlp.stanford.edu/data/glove.6B.zip"
glove_zip_path = "glove.6B.zip"

if not os.path.exists(glove_zip_path):
    urllib.request.urlretrieve(glove_url, glove_zip_path)

# Extract the smallest file (glove.6B.50d.txt)
with zipfile.ZipFile(glove_zip_path, 'r') as z:
    if not os.path.exists("glove.6B.50d.txt"):
        z.extract("glove.6B.50d.txt")

# Load GloVe vectors into a dictionary
embeddings = {}
with open("glove.6B.50d.txt", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split()
        word = parts[0]
        vec = np.array(parts[1:], dtype=np.float32)
        embeddings[word] = vec

# Compute cosine similarity between two words
def cosine_similarity(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

word1, word2 = "king", "queen"
vec1 = embeddings[word1]
vec2 = embeddings[word2]
sim = cosine_similarity(vec1, vec2)
print(f"Cosine similarity between '{word1}' and '{word2}': {sim:.3f}")
```

This script downloads 50-dimensional GloVe vectors, loads them into a Python dictionary keyed by word, and computes cosine similarity—a common measure of how similar two vectors are—between "king" and "queen".

## Comparing GloVe and Word2Vec: When To Use Each

**How they work:**  
- **Word2Vec** learns word vectors using local text windows, optimizing to predict words from nearby words (or vice versa).
- **GloVe** constructs a large matrix of global word co-occurrence counts, then learns vectors so relationships between them reflect these global patterns.
- Both produce one fixed-length vector per word.

**Practical differences:**  
- GloVe tends to encode broader global relationships, since it “sees” the whole dataset at once.
- Word2Vec is faster to train on smaller datasets or when training must run continuously on incoming text.
- Pretrained GloVe vectors often work well out of the box for analogy and similarity tasks, especially where less frequent but globally insightful connections matter.

**When to prefer each:**  
- Use **GloVe** when: You want to leverage richer global context, need high-quality pretrained vectors, or your task benefits from word relationships visible at the corpus level.
- Use **Word2Vec** when: You’re working with a smaller in-house dataset, want to adapt as new data arrives, or need easy updates and control over local window settings.

In practice, both methods generate useful embeddings for modern NLP projects. Many teams try both and keep whichever works best.

---

## Key Takeaways

- GloVe captures both local context and global co-occurrence statistics from the entire corpus.
- The model optimizes word vectors so their dot products approximate the logarithm of word co-occurrence counts.
- Pretrained GloVe embeddings can be easily used or fine-tuned for various NLP tasks.
- GloVe and Word2Vec differ in their approach: GloVe uses global statistics, Word2Vec focuses on local windows.
- GloVe vectors are often preferred when broad, corpus-level relationships are important.

## Try It Yourself

Download a different pretrained GloVe vector set, such as the Twitter embeddings. Load the vectors and write code to compute the five words most similar to 'king' using cosine similarity. Reflect on how these nearest neighbors compare to those produced by Word2Vec in a previous exercise.

## Further Resources

- 🎥 [Lecture 3 | GloVe: Global Vectors for Word Representation (Stanford CS224n lecture)](https://www.youtube.com/watch?v=ASn7ExxLZws)
- 📘 [GloVe: Global Vectors for Word Representation (Stanford NLP project page)](https://nlp.stanford.edu/projects/glove/)

---

**Coming up on Day 14:** FastText and Subword Embeddings