---
day: 12
generated_at: '2026-09-04T12:59:25.083296+00:00'
phase: Phase 2 — Word Representations & Embeddings
recap_summary: Explained how Word2Vec models generate word embeddings using skip-gram
  and CBOW techniques, compared their methods and trade-offs, and introduced a minimal
  example using Gensim.
status: pending_review
title: 'Day 12: Word2Vec Explained: Skip-Gram and CBOW'
topic_title: 'Word2Vec Explained: Skip-Gram and CBOW'
---

**Previously, on Day 11:** Explained one-hot encoding as a method for representing words as sparse vectors, highlighted its limitations regarding similarity and efficiency, and introduced the concept of word embeddings as an improvement.

---

## What Are Word2Vec Models?

One-hot vectors represent words for a computer using long lists of zeros with a single one—like a light switch for each word. Each word has a unique position. Every other position is zero. This approach is simple but has two big drawbacks. First, the vectors are extremely sparse—almost all zeros. Second, they contain no information about meaning. The one-hot vectors for “cat” and “dog” are just as different as those for “cat” and “banana,” even though “cat” and “dog” are related.

**Word2Vec** is a technique that addresses both problems. It learns short, dense vectors called **embeddings**. Each word gets its own embedding—a list of a few dozen or a few hundred decimal numbers. These embeddings are not hand-written. They are trained from large amounts of real text. Words with similar meanings end up with similar embeddings.

## The Skip-Gram Model: Predicting Context from a Target Word

The **Skip-Gram** model is one of two main approaches in Word2Vec. Its idea: _Given a single target word, can we guess which words are likely to appear near it in sentences?_

**Example:**

Take the sentence:

    "The quick brown fox jumps over the lazy dog"

Pick the word “fox.” Imagine a window that covers two words to the left and two to the right. The context words are: “quick,” “brown,” “jumps,” “over.”

The skip-gram model’s task is to take “fox” and try to predict each of those context words.

During training, the model is shown pairs like (“fox”, “quick”), (“fox”, “brown”), and so on. But it doesn’t just memorize the examples. Each word is mapped to a vector, and the model tweaks these vectors so the right target-context pairs become more likely, over millions of examples.

Over time, words like “fox,” “dog,” and “wolf” often show up in similar contexts. Their embeddings shift closer together in the vector space.

## The CBOW Model: Predicting a Target Word from Its Context

**Continuous Bag of Words (CBOW)** is the reverse of skip-gram. Instead of predicting context from a word, it predicts a single missing word given its surroundings.

**Example:**

Consider the same sentence:

    "The quick brown fox jumps over the lazy dog"

Blank out the word “fox.” Now the model sees: “quick,” “brown,” “jumps,” “over.” Its job is to predict that the missing word is “fox.”

CBOW combines the embeddings for all the context words (by averaging or adding them) and uses that combined vector to predict the most likely target word. During training, it adjusts all the embeddings so that, given surrounding words, the correct word is predicted as often as possible.

Both skip-gram and CBOW learn useful word embeddings as a side effect of trying to solve their prediction tasks.

## Comparing Skip-Gram and CBOW with Concrete Examples

Suppose we use the sentence:

    "The quick brown fox jumps"

with a context window of 2.

**Skip-Gram examples:**
- Input: “brown” → Output: “quick”, “fox”
- Input: “fox” → Output: “brown”, “jumps”

You feed the model each word in the middle and try to guess nearby words.

**CBOW examples:**
- Input: “quick”, “fox” → Output: “brown”
- Input: “brown”, “jumps” → Output: “fox”

You use the neighboring words as input and try to guess the middle word.

Key points:
- Skip-gram generates more training pairs (since every word predicts several context words), but each pair is simple—one input, one output.
- CBOW creates fewer pairs, but each combines multiple input words into a single example.

## How Training Adjusts Word Embeddings

When the model predicts incorrectly—for example, guessing “wolf” instead of “fox”—it slightly changes the embeddings to reduce the error. If “quick,” “brown,” and “jumps” often occur near both “fox” and “wolf,” the embeddings for “fox” and “wolf” will become more similar.

Words that appear in many environments, like “the” or “and,” get embeddings that reflect their generic context.

Word2Vec training is entirely driven by the patterns in real text. Nothing about meaning is written in by hand. The frequencies and co-occurrences in the data teach the model what words often appear together. The resulting math places similar-meaning words near each other in the embedding space.

Over many examples, clusters emerge. “King” and “queen” end up close together. “Walk” and “run” are neighbors. Sometimes, the difference between vectors is also meaningful: the vector from “man” to “woman” is similar to the vector from “king” to “queen.”

## When to Use Skip-Gram or CBOW

Both models learn good word vectors, but each has specific strengths.

- **Skip-Gram:** Stronger for rare words. Every rare word gets a chance to predict its context in each appearance, so the model can learn their nuances even from limited data.
- **CBOW:** Faster to train. It predicts a word from several context words combined into one example, making it more efficient. It works well for common words but can blur details for rare or unusual words.

Choose skip-gram if your data contains many unique or rare words and you care about precise representations. Choose CBOW if you have huge amounts of text and want fast, general embeddings.

## Code Example: Minimal Skip-Gram with Gensim

Here’s a simple example using `gensim`, a Python library for Word2Vec. This code trains a skip-gram model on a tiny dataset and finds the most similar words to “fox.”

```python
from gensim.models import Word2Vec

# Small demo corpus
sentences = [
    ["the", "quick", "brown", "fox", "jumps"],
    ["the", "lazy", "dog", "sleeps"],
    ["the", "fox", "outsmarts", "the", "dog"],
]

# Train a skip-gram Word2Vec model
model = Word2Vec(
    sentences,       # list of tokenized sentences
    vector_size=50,  # embedding dimensions
    window=2,        # context window size
    min_count=1,     # include all words
    sg=1,            # 1 for skip-gram; 0 for CBOW
)

# Find most similar words to “fox”
similar_words = model.wv.most_similar("fox", topn=3)
for word, score in similar_words:
    print(word, score)
```

In this snippet:
- Each sentence is tokenized as a list of words.
- `sg=1` chooses skip-gram. Switch to `sg=0` for CBOW.
- After training, we ask for the three words closest to “fox.”
- With such a tiny dataset, results are limited, but you’ll see “dog” or "brown" may appear—showing which embeddings are close even from a small set.

Word2Vec can easily scale to millions of words and much larger texts. The process for training and looking up similar words is just the same. With enough data, the model finds meaningful relationships between words automatically.

---

## Key Takeaways

- One-hot vectors lack meaning and are sparse, while Word2Vec embeddings are dense and semantic.
- Skip-gram predicts context words from a target word, generating multiple simple training pairs.
- CBOW predicts a missing word from its context, combining surrounding words in each example.
- Skip-gram excels with rare words; CBOW trains faster but can blur rare word distinctions.
- Word2Vec learns word meanings entirely from text data using co-occurrence patterns.

## Try It Yourself

Take the sentence 'The cat sat on the mat.' For a context window of 1, write out all the Skip-Gram training pairs by hand: for each word, list the words directly before and after it as outputs. This will help you see how Skip-Gram training data is structured.

## Further Resources

- 🎥 [Word2Vec Explained - CBOW & Skip‑Gram Models (YouTube)](https://www.youtube.com/watch?v=QYrhJUBWJwA)
- 📘 [TensorFlow Word2Vec Tutorial](https://www.tensorflow.org/text/tutorials/word2vec)
- 📄 [Word2Vec Demystified: From One‑Hot to Vector Magic (CBOW & Skip‑Gram Explained)](https://blog.subhampanda.com/word2vec-demystified-from-one-hot-to-vector-magic-cbow-skip-gram-explained)
- 📄 [word2vec Parameter Learning Explained (Xin Rong)](https://arxiv.org/abs/1411.2738)

---

**Coming up on Day 13:** GloVe: Global Vectors for Word Representation