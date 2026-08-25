---
day: 2
generated_at: '2026-08-25T09:23:22.730385+00:00'
phase: Phase 1 — NLP Foundations
recap_summary: Explained the importance of text preprocessing in NLP, focusing on
  tokenization and normalization techniques to make raw text manageable for machine
  learning models.
status: pending_review
title: 'Day 2: Text Preprocessing: Tokenization and Normalization'
topic_title: 'Text Preprocessing: Tokenization and Normalization'
---

**Previously, on Day 1:** Introduced Natural Language Processing (NLP), explaining what natural language is, why NLP is challenging, and real-world applications of NLP in technology and daily life. Provided an overview of the upcoming lesson series and demonstrated a simple NLP task in Python.

---

Text preprocessing is the first step in most natural language processing (NLP) projects. Raw human language is messy. We write informally, use different spellings, punctuation, and slang. Computers can’t analyze text well until we clean and organize it into manageable pieces.

Preprocessing is like preparing ingredients before cooking. Chefs wash, peel, and chop before they cook. In NLP, we “prep” text before feeding it to algorithms.

**Tokenization: Splitting Text into Units**

Tokenization means breaking text into small pieces called tokens. A token can be a word, a sentence, a single character, or a chunk smaller than a word.

Take the sentence:  
> The quick brown fox jumps over the lazy dog.

To a computer, this is just a single string of characters. For NLP tasks, we break it up. The most common method is word tokenization: splitting the sentence into words.

Tokens: `['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog']`

Why use tokens? Most NLP models—and tools like chatbots or classifiers—are built to handle tokens. A computer can’t learn patterns from “quickbrownfox,” but it can build connections between “quick,” “brown,” and “fox” in context.

There are three main ways to tokenize:

- **Character tokenization**: Each character becomes a token.  
  Example: `"chat"` → `['c', 'h', 'a', 't']`
- **Word tokenization**: Each word (split by spaces or punctuation) becomes a token.  
  Example: `"chat bots are helpful"` → `['chat', 'bots', 'are', 'helpful']`
- **Subword tokenization**: A word splits into common chunks, such as “play,” “ing,” “ed.” This helps with rare or made-up words. We’ll explore subword tokenization in a later lesson. For now, focus on word tokenization.

**Normalization: Making Text Predictable**

Normalization means making text more consistent for computers. Human writing varies in capitalization, spelling, accents, and punctuation. Normalization reduces differences that don’t matter for meaning.

Suppose you see these three strings:

- `"Dog"`
- `"dog!"`
- `"DOGS"`

A person knows these all refer to “dog.” But a computer treats them as different. Normalization helps group similar forms.

Common normalization steps:

- **Lowercasing:** Turn everything to lowercase.  
  `"Dog," "DOG," "dog"` → `"dog"`
- **Removing punctuation:** Strip periods, commas, exclamation marks, etc.  
  `"dog!"` → `"dog"`
- **Stripping whitespace:** Remove extra spaces.  
  `"  dog  "` → `"dog"`
- **Handling numbers and special symbols:** Decide if numbers (like “123”) or mixed formats (“1st”) should be changed or kept as-is.

Normalization makes matching easier and cuts out distractions. After normalization, “Hello,” “HELLO!” and “hello” all become “hello.”

**Simple Tokenization and Normalization in Python**

You can normalize and tokenize text in Python with only the standard library.

Here’s a basic example:

```python
import string

text = "The quick brown fox, jumps over the lazy dog!"
# Step 1: Lowercase the text
text = text.lower()

# Step 2: Remove punctuation
text = text.translate(str.maketrans('', '', string.punctuation))

# Step 3: Split into tokens (words)
tokens = text.split()

print(tokens)
```

This code works as follows:

- `.lower()` makes every letter lowercase.
- `str.maketrans` and `.translate()` remove punctuation.
- `.split()` separates words using whitespace.

The output is:

```
['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog']
```

This format—a list of lowercase words with no punctuation—is what most NLP models expect as input.

**Limits and Pitfalls of Preprocessing**

Preprocessing isn’t problem-free. These simple steps don’t always fit every case:

- **Languages without spaces:** Some languages, like Chinese, Japanese, and Thai, don’t use spaces between words. Splitting on spaces won’t work for them.
- **Significant punctuation:** Sometimes, punctuation changes meaning.  
  For example:  
  “Let’s eat, grandma!” versus “Let’s eat grandma!”  
  Removing the comma erases an important distinction.
- **Meaningful capitalization:** In some languages, case changes meaning.  
  In German, “Gift” (“poison”) is not the same as “gift” (English word).
- **Special characters and emojis:** These can carry crucial information, especially in social media. Deleting them may erase sentiment or intent.

Preprocessing is powerful, but every choice is a tradeoff. Always adjust preprocessing steps for your data and your task. When building NLP systems, experiment and check how changes affect results.

---

## Key Takeaways

- Text preprocessing cleans and organizes raw text for NLP tasks.
- Tokenization splits text into smaller units like words or characters.
- Normalization standardizes text using lowercasing and punctuation removal.
- Not all preprocessing steps are appropriate for every language or scenario.
- Careful preprocessing choices affect model accuracy and performance.

## Try It Yourself

Write a Python function that takes a string and returns a list of normalized tokens by lowercasing, removing punctuation, and splitting on whitespace. Test your function on two or three sentences of your choice, and compare the outputs before and after normalization to see the effect.

---

**Coming up on Day 3:** Stopwords, Stemming, and Lemmatization