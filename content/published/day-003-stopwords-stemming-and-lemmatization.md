---
day: 3
generated_at: '2026-08-26T09:30:08.224024+00:00'
phase: Phase 1 — NLP Foundations
recap_summary: Explained stopword removal, stemming, and lemmatization in NLP, including
  how they simplify and normalize text for analysis using practical examples and Python
  code.
status: published
title: 'Day 3: Stopwords, Stemming, and Lemmatization'
topic_title: Stopwords, Stemming, and Lemmatization
---

**Previously, on Day 2:** Explained the importance of text preprocessing in NLP, focusing on tokenization and normalization techniques to make raw text manageable for machine learning models.

---

What Are Stopwords and Why Remove Them?
---------------------------------------------------

Stopwords are the most common words in a language—words like "the," "is," "in," "of," "and," and "to." These show up in almost every sentence, but on their own, they usually don’t carry much meaning in text analysis.

Picture looking for information about "machine learning." In sentences about this topic, you’ll keep seeing words like "the" or "is" no matter what the specific subject is. These words help sentences flow, but they rarely point to topics or ideas.

In Natural Language Processing (NLP)—the field of building algorithms to work with human language—stopwords can act as noise. They make your data bigger and harder to work with, but they don’t help your algorithms understand the meaning in the text. Removing stopwords helps you focus on the words with more value for analysis.

Example of Stopword Removal
--------------------------------------

Consider this sentence:

> "The quick brown fox jumps over the lazy dog"

If we remove common English stopwords ("the," "over"), it becomes:

> "quick brown fox jumps lazy dog"

The result is shorter, with only the important nouns and verbs left. This makes it easier to analyze the core meaning.

The exact stopword list depends on your language and task. Most NLP tools come with a default list you can adjust as needed.

What Is Stemming?
-----------------------

Stemming is a way to reduce words to a simpler form by chopping off their ends. The idea is to group related words together by turning them into a common "root," even if that root isn’t a real dictionary word.

For example, "connect," "connected," "connection," and "connecting" all point to a similar idea. Stemming reduces each of these to the same base string—often "connect." This approach makes it easier to match and count similar meanings, even if grammar is ignored.

Stemming in Practice
--------------------------

Here are some examples:

Original Word    | Stemmed Form
-----------------|-------------
playing          | play
played           | play
plays            | play
player           | player
happily          | happili
running          | run
generalization   | general

Note:

- "playing," "played," and "plays" all become "play." This helps you count similar operations together.
- But "happily" turns into "happili," which isn’t a real word. Sometimes, meaning gets lost or the output looks odd.

Stemming is simple and quick to run. It doesn’t look at meaning or grammar—just the word’s shape.

What Is Lemmatization?
------------------------------

Lemmatization is a more sophisticated way to normalize words. It uses vocabulary and grammar rules to find a word’s official base form, called its "lemma." The lemma is always a real word.

For instance, "better" is lemmatized to "good" (since "good" is the root adjective). "Running" becomes "run," and so on. Lemmatization uses the word’s meaning and sometimes its part of speech (noun, verb, etc.) to figure out the correct root.

Lemmatization takes more computation than stemming, but its results are more accurate and easier to read.

Examples: Lemmatization vs. Stemming
-------------------------------------------

Original Word      | Lemmatized Form
-------------------|----------------
am, are, is        | be
better             | good
playing            | play
played             | play
children           | child
leaves (noun)      | leaf
leaves (verb)      | leave
wolves             | wolf

Lemmatization returns proper words and makes choices based on context. For example, "leaves" can be "leaf" (if it's a noun) or "leave" (if it's a verb).

Comparing Stopword Removal, Stemming, and Lemmatization
-----------------------------------------------------------------

Each of these tools is useful for its own purpose:

- **Stopword removal** is almost always a helpful first step. It removes noise and highlights the words most likely to matter.
- **Stemming** is good for speed and simple grouping, when you care less about perfect grammar. Many search engines use stemming for fast lookup.
- **Lemmatization** gives cleaner, dictionary words and keeps meaning clear. If you’ll show results to users or need subtle distinctions, use lemmatization.

You don’t always need all three. Pick the combination that matches your task and data.

Python Example: Stopword Removal, Stemming, and Lemmatization
-------------------------------------------------------------------

Let’s see how these look in Python, using the [NLTK library](https://www.nltk.org/):

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

sentence = "The striped bats are hanging on their feet for best"

# 1. Tokenize the sentence into words
words = word_tokenize(sentence)

# 2. Remove stopwords
stop_words = set(stopwords.words('english'))
filtered_words = [w for w in words if w.lower() not in stop_words]

# 3. Stemming
stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(w) for w in filtered_words]

# 4. Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(w) for w in filtered_words]

print("Original:", words)
print("After stopword removal:", filtered_words)
print("After stemming:", stemmed_words)
print("After lemmatization:", lemmatized_words)
```

This code:

- Breaks the sentence into words.
- Removes common English stopwords.
- Applies stemming to the rest.
- Applies lemmatization to the same words.
- Prints each stage so you can see what changes.

Try using this on other sentences to see how stopwords, stemming, and lemmatization shape your text data.

---

## Key Takeaways

- Stopwords are common words that can be removed to reduce noise in text analysis.
- Stemming quickly reduces words to simple roots but may create non-dictionary forms.
- Lemmatization finds the correct base word using grammar and context, producing real dictionary words.
- Not all tasks require all three methods; the choice depends on the specific analysis needs.

## Try It Yourself

Choose a short paragraph of two or three sentences. Remove the stopwords, then apply stemming and lemmatization to the remaining words. List the stemmed and lemmatized outputs side by side, and note which method you find clearer or more useful and why.

## Further Resources

- 🎥 [3.4. Text Normalization 101: Stopwords, Stemming, and Lemmatization](https://www.youtube.com/watch?v=lXktcjDEjq8)
- 🎥 [NLP Demystified 3: Basic Preprocessing (case‑folding, stop words, stemming, lemmatization)](https://www.youtube.com/watch?v=I173TmCTxpk)
- 📘 [NLTK Book, Chapter 3 (Processing Raw Text) including Normalizing Text: stemming and lemmatization](https://www.nltk.org/book/ch03.html)
- 📘 [NLTK API Documentation for nltk.stem (stemmers) and WordNetLemmatizer](https://www.nltk.org/api/nltk.stem.html?highlight=lemmatizer)
- 📄 [Stemming and Lemmatization in Python | DataCamp tutorial (June 1, 2026)](https://www.datacamp.com/tutorial/stemming-lemmatization-python)

---

**Coming up on Day 4:** Bag-of-Words and Text Vectorization