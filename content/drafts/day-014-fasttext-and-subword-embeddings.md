---
day: 14
generated_at: '2026-09-08T13:08:36.515693+00:00'
phase: Phase 2 — Word Representations & Embeddings
recap_summary: Explained how word embeddings represent words as vectors, the limitations
  of classic approaches like one-hot encoding, and how FastText uses subword units
  to create robust embeddings that handle rare, new, or misspelled words.
status: pending_review
title: 'Day 14: FastText and Subword Embeddings'
topic_title: FastText and Subword Embeddings
---

### What Are Word Embeddings?

Words like "cat," "run," or "happiness" are just symbols to a computer. Computers operate on numbers, not symbols. To process language, we need a way to turn words into numbers—specifically, into mathematical objects that reflect their meaning and usage.

Word embeddings solve this. An *embedding* is simply a list of numbers—a vector—that represents a word. If two words are used in similar contexts, their vectors should be close together. For example, "cat" and "dog" would have similar vectors; both are common animals, and often appear in similar situations.

Word embeddings make it possible for machines to:
- Compare word meanings mathematically. (For example, using cosine similarity, which measures the angle between two vectors.)
- Cluster similar words together in this new number-based space.
- Use these vectors as input features for machine learning models, including neural networks.

Before word embeddings, earlier approaches represented words in more primitive ways.

---

### The Limits of Classic Embeddings

The most basic way to represent a word as a number is *one-hot encoding*. In one-hot encoding, each word in the vocabulary gets its own vector. This vector contains a single "1" in the position for that word, and "0" everywhere else. If your vocabulary has 10,000 words, every word gets a 10,000-dimensional vector, with just one non-zero entry.

This approach has major problems:
- Wastes memory. Almost every entry is zero.
- Shows no relationships between words. "Cat" and "dog" are as far apart as "cat" and "microwave."
- Cannot handle words that were not seen during training time. These are called "out-of-vocabulary" (OOV) words.
- Cannot handle typos. The model treats “cat” and “cta” as unrelated.
- Struggles with languages that have many forms of the same root word. Each form is treated as unrelated.

Word embeddings such as Word2Vec or GloVe improved this by learning shorter, dense vectors for each word. Similar words have similar embeddings. But these classic embeddings still assign one unique vector per word. If a word is rare or was never seen in training, these models cannot create a vector for it.

---

### FastText: Embeddings With Subwords

FastText, created by Facebook’s AI Research lab, addresses these gaps by looking inside words. Instead of treating a word as a single chunk, FastText represents each word as a bag of *subword units*, also called *n-grams*.

A subword unit is just a short piece of the word. For example, for the word “playing,” some 3-character n-grams are “pla,” “lay,” “ayi,” “yin,” and “ing.” FastText splits each word into overlapping character n-grams, usually between 3 and 6 characters long. Each n-gram gets its own vector.

To build the final embedding for a word, FastText combines the vectors for the word itself and all of its n-grams—typically by averaging or summing them.

For example, “jumping” would be represented by:
- The embedding for the whole word “jumping”
- The embeddings for “jum,” “ump,” “mpi,” “pin,” “ing” (and possibly others, depending on n-gram size)

Earlier approaches had no understanding of parts within a word. FastText fixes this.

---

### Why Subword Embeddings Matter

Suppose the training data never included the word “jumped.” With classic embeddings, there is no vector for this word. The model cannot process it.

With FastText, "jumped" is split into n-grams such as "jum," "ump," "mpe," "ped." Even if “jumped” never showed up during training, its n-grams might have, from words like “jump,” “jumping,” or even “stumped.” FastText can build a vector for "jumped" by combining those parts.

This approach has important advantages:
- **Robust to typos.** “jumping” and a typo like “jum ping” share many n-grams, so their vectors are similar.
- **Handles rare or invented words.** If a rare word like “quixotically” shares letter chunks with common words, it still gets a reasonable vector.
- **Works with morphologically rich languages.** Many languages use lots of word endings to change meaning. FastText can represent all these forms, even if it never saw them in training.

FastText is much less brittle than previous models, and handles natural, messy language well.

---

### Working With FastText: A Minimal Example

Here’s how to use FastText in practice. We’ll use Gensim, a Python library, to train a tiny FastText model. Then we’ll look up vectors for a common word, a rare word, and a totally new word.

Install Gensim:

```bash
pip install gensim
```

Minimal FastText code:

```python
from gensim.models import FastText

# Simple dataset: each sentence is a list of words.
corpus = [
    ["cat", "sat", "on", "the", "mat"],
    ["dog", "sat", "beside", "the", "cat"],
    ["dogs", "and", "cats", "are", "friends"],
    ["the", "dog", "jumped", "over", "the", "log"]
]

# Train a FastText model. Kept small for clarity.
model = FastText(sentences=corpus, vector_size=10, window=3, min_count=1)

# Common word
print("Vector for 'cat':", model.wv['cat'])

# Rare word (seen once)
print("Vector for 'jumped':", model.wv['jumped'])

# Plausible word never seen in training
print("Vector for 'jumping':", model.wv['jumping'])

# Made-up or misspelled word
print("Vector for 'cattz':", model.wv['cattz'])
```

Key points:
- “jumping” and “cattz” were never in the training set. But FastText can produce vectors for them, by combining vector pieces from their n-grams.
- Classic Word2Vec or GloVe models cannot handle such words—they only know the words seen during training.

---

### When (and When Not) to Use FastText

FastText’s subword approach is most useful when:
- Your text is messy, with typos, slang, or creative spelling (like in chat logs or social media).
- You are working with languages that have many word forms due to rich grammar (morphology).
- Your domain or dataset has many rare, specialized words.

Situations where FastText may be less helpful:
- FastText is slower. At prediction time, it needs to look up and combine several vectors for each word.
- Sometimes, the similarities it finds are harder to explain, since words sharing n-grams might not actually be related in meaning.
- For clean, well-edited text in a language like English, and when you have plenty of training data, simpler embeddings like Word2Vec or GloVe are often sufficient.

If your data is human-authored, noisy, or has lots of words outside your training set, FastText is a strong choice. For very large projects, it’s worth testing both classic and subword-based embeddings to see what works better for your needs.

---

## Key Takeaways

- Word embeddings translate words into numerical vectors capturing semantic similarity.
- Classic embeddings like one-hot cannot capture word relationships or handle unseen words.
- FastText creates embeddings using character n-grams, enabling vector creation for out-of-vocabulary words.
- Subword embeddings are robust to typos, rare words, and morphologically rich languages.
- FastText trades some speed and interpretability for improved handling of noisy, diverse data.

## Try It Yourself

Write a list of 5 words, including at least one made-up word and one misspelled word. Train a small FastText model on a simple dataset using Gensim, then extract vectors for your test words. Compare the vectors of the made-up and misspelled words to those of semantically or orthographically similar real words, and note how well FastText handles these out-of-vocabulary inputs.

## Further Resources

- 🎥 [FastText Explained | Subword Embeddings in NLP](https://www.youtube.com/watch?v=-K5QQIMV5F4)
- 📘 [FastText embeddings, Word representations tutorial (fastText official documentation)](https://fasttext.cc/docs/en/unsupervised-tutorial.html)
- 📄 [Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606)
- 📄 [Intro to FastText Subword Embeddings (Kaggle notebook)](https://www.kaggle.com/code/lorenzoscaturchio/intro-to-fasttext-subword-embeddings)

---

**Coming up on Day 15:** Visualizing Embeddings with t-SNE and PCA