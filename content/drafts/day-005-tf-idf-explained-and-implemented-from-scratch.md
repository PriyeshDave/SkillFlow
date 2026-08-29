---
day: 5
generated_at: '2026-08-28T20:28:25.155953+00:00'
phase: Phase 1 — NLP Foundations
recap_summary: Explained TF-IDF as a method to identify the most important words in
  documents by combining term frequency and inverse document frequency, contrasting
  it with Bag-of-Words, and demonstrating its calculation and implementation in Python.
status: pending_review
title: 'Day 5: TF-IDF Explained and Implemented from Scratch'
topic_title: TF-IDF Explained and Implemented from Scratch
---

**Previously, on Day 4:** Explained text vectorization in Natural Language Processing, focusing on the Bag-of-Words model for converting text into numerical vectors by counting word occurrences.

---

## What is TF-IDF and Why Do We Need It?

Bag-of-Words (BoW) is a basic way to turn documents into numbers. It counts how often each word occurs in a document. This makes it possible to compare documents, train models, or look for patterns in text.

However, BoW treats every word as equally important. This is a problem. Common words like "the," "is," and "and" show up in almost every document, but they don’t help us figure out what a document is really about. If BoW sees "the" fifty times in one document and twice in another, it treats that difference as meaningful—even though it usually isn’t.

Instead, we want a method that:
- **Highlights** words that are especially important or unique to a specific document.
- **Downplays** words that are common everywhere, since they don't help us tell documents apart.

This is the motivation for **TF-IDF**.

TF-IDF stands for **Term Frequency–Inverse Document Frequency**. It scores each word based on how frequent it is in one document, compared to how rare it is in the whole set. If a word appears often in one document but rarely elsewhere, it gets a high TF-IDF score.

## Breaking Down TF-IDF: Term Frequency and Inverse Document Frequency

TF-IDF has two parts: **Term Frequency (TF)** and **Inverse Document Frequency (IDF)**.

### Term Frequency (TF)

Term Frequency measures how often a word appears in a single document. Take the count of that word in the document and divide by the total words in that document.

**Formula:**
```
TF = (Number of times word w appears in document d) / (Total words in document d)
```

**Example:**
- Document: `"the cat sat on the mat"`
- TF for `"cat"`:
    - "cat" appears 1 time.
    - There are 6 words total.
    - TF = 1/6 ≈ 0.17

### Inverse Document Frequency (IDF)

Inverse Document Frequency measures how rare a word is across all documents. If a word appears in many documents, it gets a low IDF. If it appears in few documents, it gets a high IDF.

**Formula (common version):**
```
IDF = log(N / df)
```
- _N_ = total number of documents.
- _df_ = number of documents containing the word.

**Example:**
- Documents:
    1. `"the cat sat on the mat"`
    2. `"the dog sat on the log"`
    3. `"the cow jumped over the moon"`

- `"cat"` appears in Document 1 only, so df = 1.
- N = 3.
- IDF for "cat": log(3 / 1) ≈ 1.10

Note that some formulas use `1 + df` in the denominator to avoid division by zero. For words that appear in all documents, IDF becomes zero or close to zero depending on the version.

### TF-IDF (Combining the Parts)

The TF-IDF score is just the product:
```
TF-IDF = TF * IDF
```
A word that appears often in a document, but not in others, gets the highest scores.

## Step-By-Step Example

**Documents:**
1. `"the cat sat"`
2. `"the dog sat"`
3. `"the cat lay"`

Let's calculate the TF-IDF for `"cat"` in Document 1.

**Step 1: TF for "cat" in Document 1**
- Tokens: `["the", "cat", "sat"]`
- "cat" count = 1
- Total words = 3
- TF = 1/3 ≈ 0.33

**Step 2: IDF for "cat"**
- "cat" is in Document 1 and 3. So df = 2.
- N = 3.
- IDF = log(3 / 2) ≈ 0.405

**Step 3: TF-IDF**
- TF-IDF = 0.33 * 0.405 ≈ 0.134

We can repeat this for every word in each document to build a TF-IDF matrix, showing the importance of each word per document.

## Implementing TF-IDF from Scratch in Python

Below is a direct, runnable example. This uses no extra libraries beyond `math`.

Suppose we use these three documents:

```python
documents = [
    "the cat sat",
    "the dog sat",
    "the cat lay"
]
```

Here's a minimal implementation:

```python
import math

# Tokenize: split by spaces and lowercase everything
def tokenize(doc):
    return doc.lower().split()

# Build full vocabulary
def build_vocabulary(docs):
    vocab = set()
    for doc in docs:
        vocab.update(tokenize(doc))
    return sorted(vocab)

# Compute TF for one doc
def compute_tf(doc_tokens, vocabulary):
    tf = {}
    total_terms = len(doc_tokens)
    for term in vocabulary:
        tf[term] = doc_tokens.count(term) / total_terms
    return tf

# Compute document frequencies
def compute_df(tokenized_docs, vocabulary):
    df = {}
    for term in vocabulary:
        df[term] = sum(1 for doc in tokenized_docs if term in doc)
    return df

# Compute IDF for each term
def compute_idf(df, N):
    idf = {}
    for term, freq in df.items():
        idf[term] = math.log(N / freq) if freq else 0.0  # Avoid division by zero
    return idf

# Combine to compute TF-IDF
def compute_tfidf(documents):
    tokenized_docs = [tokenize(doc) for doc in documents]
    vocabulary = build_vocabulary(documents)
    dfs = compute_df(tokenized_docs, vocabulary)
    idfs = compute_idf(dfs, len(documents))
    
    tfidf_vectors = []
    for doc_tokens in tokenized_docs:
        tf = compute_tf(doc_tokens, vocabulary)
        tfidf = {term: tf[term] * idfs[term] for term in vocabulary}
        tfidf_vectors.append(tfidf)
    return tfidf_vectors, vocabulary

# Run example
documents = [
    "the cat sat",
    "the dog sat",
    "the cat lay"
]

tfidf_vectors, vocabulary = compute_tfidf(documents)

for idx, vec in enumerate(tfidf_vectors):
    print(f"Document {idx+1}:")
    for term in vocabulary:
        print(f"  {term:>5}: {vec[term]:.3f}")
    print()
```

This script lowercases and splits each document, builds the vocabulary, counts term and document frequencies, then calculates TF-IDF values for each word and document. Each document is represented as a dictionary mapping words to their TF-IDF values.

## TF-IDF Versus Bag-of-Words: Why It Matters

BoW counts all words equally. If one document uses "the" or "sat" more, BoW thinks it’s important—even though these are common words.

TF-IDF reduces the weight of words that appear everywhere (like "the" and "sat"). It increases the score for words that are more specific to each document ("cat," "dog," "lay"). If documents are about similar topics (for example, different types of animals), TF-IDF helps highlight what’s unique about each.

**When to use which:**
- Use Bag-of-Words if every word is meaningful and jargon is key.
- Use TF-IDF if you want to focus on the words that actually separate one document from others.

TF-IDF is a foundation for many search engines, text-matching systems, and automatic topic detectors. Understanding it—and how to implement it—unlocks much of modern text analysis.

---

## Key Takeaways

- Bag-of-Words treats every word equally, often overvaluing common words.
- TF-IDF highlights words unique or significant to a document by downweighting common terms.
- Term Frequency measures how often a word appears in a document.
- Inverse Document Frequency measures how rare a word is across all documents.
- TF-IDF is widely used in search engines and text analysis to identify distinguishing words.

## Try It Yourself

Write a Python function that takes two short text documents and calculates TF-IDF scores for each word. For each document, print the words with the highest and lowest TF-IDF values. Then, briefly explain in your own words why the high-score words are important for that document and what the low-score words represent.

## Further Resources

- 🎥 [TF‑IDF Explained: Document Similarity Search in NLP with Python Code](https://www.youtube.com/watch?v=JfMEUkCNfsc)
- 📄 [A Guide to TF‑IDF (Built In)](https://builtin.com/articles/tf-idf)
- 📄 [Python for NLP: Creating TF‑IDF Model from Scratch](https://stackabuse.com/python-for-nlp-creating-tf-idf-model-from-scratch/)
- 📄 [Introduction to tf‑idf (Jake Tae)](https://jaketae.github.io/study/tf-idf/)

---

**Coming up on Day 6:** N-Grams and Statistical Language Modeling