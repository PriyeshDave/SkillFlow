---
day: 4
generated_at: '2026-08-27T19:18:41.298489+00:00'
phase: Phase 1 — NLP Foundations
recap_summary: Explained text vectorization in Natural Language Processing, focusing
  on the Bag-of-Words model for converting text into numerical vectors by counting
  word occurrences.
status: pending_review
title: 'Day 4: Bag-of-Words and Text Vectorization'
topic_title: Bag-of-Words and Text Vectorization
---

**Previously, on Day 3:** Explained stopword removal, stemming, and lemmatization in NLP, including how they simplify and normalize text for analysis using practical examples and Python code.

---

# Text Vectorization: Turning Words into Numbers

Computers work with numbers, not text. To handle language, a Natural Language Processing (NLP) system must convert words, sentences, or documents into numerical data. Usually, this means turning them into vectors—ordered arrays of numbers. This process is called **text vectorization**.

A vector is a mathematical summary of a piece of text. The details and meaning behind the numbers depend on which vectorization method is used, but all serve a common purpose: to translate language into something a machine can process.

For example, imagine building a program to filter spam emails. The program can't directly understand words like "WINNER" or "sale." Every word must be mapped to a number before the program can look for patterns in messages.

## What is the Bag-of-Words Model?

**Bag-of-Words** (BoW) is the simplest and most common way to vectorize text. BoW ignores grammar and word order. It treats each document as a "bag" containing words, just counting how many times each word appears.

For example, the sentences "dog bites man" and "man bites dog" will produce the same vector in a BoW system. Both have the words "dog," "bites," and "man," each once. The meaning is very different to a human, but to BoW, they're identical.

This straightforward approach makes BoW fast and effective for many tasks, especially where quickly spotting key words is enough—for example, spam detection.

## From Words to Vectors: Building a Vocabulary

The first step in BoW is to build a **vocabulary**. This is a list of all unique words seen across your dataset (called a "corpus").

Suppose your dataset contains two sentences:
- "cat sat on the mat"
- "dog sat on the log"

List all unique words:
["cat", "sat", "on", "the", "mat", "dog", "log"]

The word order in the vocabulary doesn't matter, as long as you're consistent. Most systems sort words alphabetically, or keep them in the order they're first seen.

Assign each word a unique index. For example:

| Index | Word |
|-------|------|
|   0   | cat  |
|   1   | sat  |
|   2   | on   |
|   3   | the  |
|   4   | mat  |
|   5   | dog  |
|   6   | log  |

## Vectorizing Individual Texts

For each new document, you create a vector that's as long as your vocabulary. Each position in the vector matches a word from your vocabulary.

Count how many times each vocabulary word appears in the document. Place that count in the corresponding slot.

Let’s see this with our example sentences:

1. "cat sat on the mat"

| "cat" | "sat" | "on" | "the" | "mat" | "dog" | "log" |
|-------|-------|------|-------|-------|-------|-------|
|   1   |   1   |  1   |   1   |   1   |   0   |   0   |

2. "dog sat on the log"

| "cat" | "sat" | "on" | "the" | "mat" | "dog" | "log" |
|-------|-------|------|-------|-------|-------|-------|
|   0   |   1   |  1   |   1   |   0   |   1   |   1   |

For each document, you get a simple list of numbers—telling you how often each word from your vocabulary appears.

## Example: Bag-of-Words in Python

Here’s how to do this with a few lines of Python.

**Manual Vocabulary Building and Vectorization**

```python
texts = [
    "cat sat on the mat",
    "dog sat on the log"
]

# Step 1: Build the vocabulary
vocab = sorted(set(" ".join(texts).split()))  # Sort for reproducibility
word_to_index = {word: idx for idx, word in enumerate(vocab)}

print("Vocabulary:", vocab)

# Step 2: Convert texts to vectors
def text_to_bow(text, vocab, word_to_index):
    vector = [0] * len(vocab)
    for word in text.split():
        if word in word_to_index:
            vector[word_to_index[word]] += 1
    return vector

vectors = [text_to_bow(text, vocab, word_to_index) for text in texts]
for text, vec in zip(texts, vectors):
    print(f"Text: '{text}' --> {vec}")
```

**Result Output**
```
Vocabulary: ['cat', 'dog', 'log', 'mat', 'on', 'sat', 'the']
Text: 'cat sat on the mat' --> [1, 0, 0, 1, 1, 1, 1]
Text: 'dog sat on the log' --> [0, 1, 1, 0, 1, 1, 1]
```
The vocabulary order comes from sorting—if you don't sort the words, the order may change.

**Using scikit-learn for Bag-of-Words**

For real projects, you’ll typically use a library. Python’s [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) includes a class called `CountVectorizer` that builds the vocabulary and vectors for you.

```python
from sklearn.feature_extraction.text import CountVectorizer

texts = [
    "cat sat on the mat",
    "dog sat on the log"
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

print("Vocabulary:", vectorizer.get_feature_names_out())
print("Vectors:\n", X.toarray())
```

**Result Output**
```
Vocabulary: ['cat' 'dog' 'log' 'mat' 'on' 'sat' 'the']
Vectors:
 [[1 0 0 1 1 1 1]
  [0 1 1 0 1 1 1]]
```

Each row in the array is the vector for a text; each column matches a word from the vocabulary.

## Strengths and Weaknesses of Bag-of-Words

Bag-of-Words is fast, simple, and effective when word presence is more important than word order or context. The result is easy to interpret: each vector is just a set of word counts.

But it has clear limits:

- **Word order is ignored:** "dog bites man" and "man bites dog" look the same.
- **Loss of context:** It can't tell if "bank" means a riverbank or a financial bank.
- **New words:** Words not already in the vocabulary are ignored or get special handling.

Most modern text vectorization methods—like word embeddings and transformers—were designed to solve these problems. But Bag-of-Words is the starting point. Knowing how it works and where it falls short is essential before moving on in Natural Language Processing.

---

## Key Takeaways

- Text must be converted to numbers for machine processing.
- Bag-of-Words treats texts as unordered word collections and counts word frequencies.
- Building a vocabulary is the first step to vectorize text.
- Bag-of-Words ignores word order and context.
- Libraries like scikit-learn simplify Bag-of-Words vectorization.

## Try It Yourself

Choose any three short sentences—either your own or from news headlines. List all unique words to build a vocabulary, then manually create a Bag-of-Words vector for each sentence by counting word occurrences. Afterwards, use Python's sklearn CountVectorizer to generate the vectors and check if your manual results match the library output.

## Further Resources

- 🎥 [Bag of Words](https://www.youtube.com/watch?v=eCdYyaDtjjQ)
- 📘 [8.2 Feature extraction — scikit‑learn documentation (Bag of Words)](https://scikit-learn.org/stable/modules/feature_extraction.html?highlight=tfidfvectorizer)
- 📄 [Basic Bag‑of‑Words | Natural Language Processing Demystified](https://www.nlpdemystified.org/course/basic-bag-of-words)

---

**Coming up on Day 5:** TF-IDF Explained and Implemented from Scratch