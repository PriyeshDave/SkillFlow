---
day: 6
generated_at: '2026-08-30T11:55:45.960211+00:00'
phase: Phase 1 — NLP Foundations
recap_summary: Explained statistical language models and how n-grams are used to predict
  word sequences, including calculating probabilities using bigram counts and understanding
  basic limitations of n-gram models.
status: published
title: 'Day 6: N-Grams and Statistical Language Modeling'
topic_title: N-Grams and Statistical Language Modeling
---

**Previously, on Day 5:** Explained TF-IDF as a method to identify the most important words in documents by combining term frequency and inverse document frequency, contrasting it with Bag-of-Words, and demonstrating its calculation and implementation in Python.

---

Statistical language models help computers estimate how likely a sequence of words is. When your phone predicts the next word as you type, it uses a language model behind the scenes. These models enable text prediction, speech recognition, and machine translation by guessing which words make sense in context.

A language model works by assigning a probability to a sequence of words. Sequences with higher probabilities are more “natural”—they occur more often in real language. If a computer can accurately predict the next word, it can “understand” enough about text to be useful in real applications.

## N-Grams: The Basic Unit

A straightforward way to model text is to look at short, fixed-length chunks of words and learn which ones usually appear together. These chunks are called **n-grams**.

- An **n-gram** is just a sequence of 'n' words from the text.
- A **unigram** (n=1) is a single word:  
  Examples: `"the"`, `"cat"`, `"sat"`
- A **bigram** (n=2) is a pair of words that appear next to each other:  
  Examples: `"the cat"`, `"cat sat"`
- A **trigram** (n=3) is three words in a row:  
  Example: `"the cat sat"`

Breaking text into n-grams helps us spot patterns. For instance, `"New York"` often goes together, but `"York city"` is less common. This hints to the model that after seeing `"New"`, the word `"York"` is very likely to come next.

## Counting N-Grams: Turning Text into Numbers

To make n-grams useful for a computer, we need to count how often each n-gram appears in our text.

Suppose our text is:  
`"the cat sat on the mat"`

The bigrams (pairs of consecutive words) are:
- `"the cat"`
- `"cat sat"`
- `"sat on"`
- `"on the"`
- `"the mat"`

We count each bigram. In this short example, every bigram shows up exactly once.

With larger texts, some n-grams appear many times, and others not at all. This counting gives us a frequency table: a simple list showing which n-grams are most common in the data.

Even with these basic counts, we can already predict or suggest likely next words by checking which words tend to follow each other.

## From Counts to Probabilities: The Simple Language Model

The next step is turning these counts into probabilities. We want to know: how likely is word B to follow word A?

Using our bigram counts, the probability of `"cat"` after `"the"`—written as P(`"cat"`|`"the"`)—is:

\[
P(\text{"cat"}|\text{"the"}) = \frac{\text{Count("the cat")}}{\text{Count("the")}}
\]

- The top part counts how often we see "the cat".
- The bottom part counts how often "the" appears as the first word in any bigram.

The result is the chance of seeing "cat" just after "the." This is the heart of a statistical language model.

This approach uses the **Markov assumption**, sometimes called the **Markov property**. This means our model only looks at a short history—just the last word or two—instead of the entire previous sentence. It greatly reduces the complexity and makes the system practical, even for big datasets.

### Example: Calculating Bigram Probabilities With Python

Here’s a runnable example:

```python
from collections import Counter

# Sample text
text = "the cat sat on the mat"

# Split the text into words
words = text.split()

# Collect all bigrams (pairs of adjacent words)
bigrams = [(words[i], words[i+1]) for i in range(len(words) - 1)]

# Count how often each bigram and each word occurs
bigram_counts = Counter(bigrams)
unigram_counts = Counter(words)

# Calculate the probability that 'cat' follows 'the'
bigram = ('the', 'cat')
count_bigram = bigram_counts[bigram]
count_unigram = unigram_counts[bigram[0]]

if count_unigram > 0:
    probability = count_bigram / count_unigram
else:
    probability = 0

print(f"P('cat' | 'the') = {probability:.2f}")
```

This code:
- Splits the text into words.
- Finds all bigrams.
- Counts how often each bigram and word appear.
- Calculates P('cat' | 'the'), the probability that "cat" comes after "the".

## What N-Gram Models Can't Do

N-gram models are foundational, but they have clear weaknesses:

- **Data sparsity:** As you make n-grams longer (bigger n), most possible word combinations never appear in even large datasets. Their probabilities end up as zero because the model never saw them.
- **Limited context:** N-gram models only use a narrow window of previous words. If important context is farther away in the sentence, an n-gram model misses it.
- **No real understanding:** N-grams treat language as just word sequences. They don’t know about meaning, grammar, or synonyms—just counts.

Even with these drawbacks, n-gram models powered much of early natural language processing (NLP). They are still essential ideas. Many of today’s most advanced models, including deep neural networks, build on the basic logic of n-gram statistics. Mastering n-gram models sets the stage for understanding every language model that followed.

---

## Key Takeaways

- A language model estimates the likelihood of word sequences by assigning probabilities.
- N-grams are fixed-length word sequences, such as unigrams, bigrams, and trigrams.
- Counting n-grams lets us predict which words are likely to follow others.
- Bigram probabilities are calculated by dividing bigram counts by unigram counts.
- N-gram models are limited by data sparsity and their reliance on short context windows.

## Try It Yourself

Take a short paragraph (2-3 sentences). Write out all the bigrams (pairs of consecutive words) in the text and count how many times each occurs. Then, choose two different first words and calculate the probability of a specific next word following each, using the bigram over unigram formula introduced in the lesson.

## Further Resources

- 🎥 [N-gram Language Modeling | Theory, Math, Code](https://www.youtube.com/watch?v=Vc2C1NZkH0E)
- 📄 [Modeling Natural Language with N‑Gram Models](https://sookocheff.com/post/nlp/n-gram-modeling/)
- 📘 [NLTK Language Modeling Module (nltk.lm)](https://www.nltk.org/api/nltk.lm.html)

---

**Coming up on Day 7:** Part-of-Speech Tagging