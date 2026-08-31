---
day: 7
generated_at: '2026-08-31T15:26:34.150753+00:00'
phase: Phase 1 — NLP Foundations
recap_summary: Explained part-of-speech (POS) tagging, its importance in NLP, different
  approaches to tagging, common tagsets, and typical challenges such as ambiguity
  and unknown words.
status: pending_review
title: 'Day 7: Part-of-Speech Tagging'
topic_title: Part-of-Speech Tagging
---

**Previously, on Day 6:** Explained statistical language models and how n-grams are used to predict word sequences, including calculating probabilities using bigram counts and understanding basic limitations of n-gram models.

---

### What Is Part-of-Speech Tagging?

Part-of-speech tagging assigns each word in a sentence a label describing its role—such as noun, verb, or adjective. In simple terms, each word gets a tag that shows what job it’s doing in that sentence. For example, “run” can be a noun (“a morning run”) or a verb (“I run”). Tagging tells a computer which is which.

### Parts of Speech: A Quick Refresher

Here are the major parts of speech in English:

- **Noun:** Names a person, place, or thing.  
  Example: “cat”, “London”, “happiness”
- **Verb:** Describes an action or state.  
  Example: “run”, “is”, “think”
- **Adjective:** Describes a noun.  
  Example: “blue”, “quick”, “loud”
- **Adverb:** Modifies a verb, adjective, or other adverb.  
  Example: “quickly”, “very”, “well”
- **Pronoun:** Takes the place of a noun.  
  Example: “he”, “she”, “it”
- **Preposition:** Shows relationships in time or space.  
  Example: “in”, “on”, “at”
- **Conjunction:** Joins words or groups of words.  
  Example: “and”, “but”, “or”
- **Determiner:** Introduces a noun.  
  Example: “the”, “a”, “an”, “some”

Recognizing these roles helps computers—and us—understand how sentences work.

### Why Tag Parts of Speech?

POS tagging is a building block for many natural language processing (NLP) tasks:

- **Parsing sentence structure:** A computer needs to know which words have which roles to break down a sentence’s meaning.
- **Search and retrieval:** If you search for “read” as a noun (“the read was long”), you don’t want results about “read” as a verb.
- **Translation, text-to-speech, and more:** Knowing a word’s function helps machines translate or pronounce it correctly.
- **Other NLP tasks:** Named entity recognition, coreference resolution (tracking “he”/“she”), and sentiment analysis usually start with tagging.

Without POS tags, algorithms often can’t tell what a sentence really means.

### Manual vs. Automated Tagging

Humans can tag each word’s part of speech, but this is slow and tedious for anything longer than a short paragraph.

As NLP grew, automating tagging became essential. Algorithms can now process millions of sentences quickly and usually with high accuracy.

### How Automated POS Tagging Works

Automated tagging uses two main approaches:

- **Rule-based tagging:** Applies a list of rules about English. For example, “run” after “to” is likely to be a verb.
- **Statistical models:** Learns from lots of already-tagged text (“corpora,” the plural of “corpus,” meaning “text collection”). Early models use probability to pick the most likely tag for each word, based on its neighbors. For example, **Hidden Markov Models (HMMs)** predict tags by looking at which tags and words tend to appear together.

One of the biggest challenges is **ambiguity**—words like “can” might be a verb (“can you help?”) or a noun (“a can of soup”). Tagging depends heavily on context to resolve these cases.

### Walkthrough: Tagging a Sentence by Hand

Let’s tag this sentence:  
**"The can will rust."**

Word by word:

- **The** — Determiner (DT): Introduces a noun.
- **can** — Noun (NN): Here, “can” is a metal container, not the verb.
- **will** — Modal verb (MD): Shows future tense.
- **rust** — Verb (VB): The action.

Change the sentence to:  
**"We can fish,"**  
and “can” becomes a verb—meaning “to be able to.” Tagging always needs context.

### A Minimal Code Example: POS Tagging with NLTK

Python’s NLTK library can tag parts of speech automatically:

```python
import nltk

# Download resources needed for NLTK
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

sentence = "The can will rust."
tokens = nltk.word_tokenize(sentence)
tagged = nltk.pos_tag(tokens)

for word, tag in tagged:
    print(f"{word}\t{tag}")
```

Sample output:
```
The    DT
can    NN
will   MD
rust   VB
.      .
```

- **DT:** Determiner (“The”)
- **NN:** Noun (“can”)
- **MD:** Modal verb (“will”)
- **VB:** Verb (“rust”)
- **.**: Punctuation

NLTK handles both splitting the sentence into words (tokenization) and tagging. Each word gets paired with a code for its part of speech.

### Common Tags and Tagsets

A **tagset** is the set of all possible tags a system can use. The most common for English is the **Penn Treebank tagset**.

Some useful tags from Penn Treebank:

- **NN:** Noun, singular (“cat”)
- **NNS:** Noun, plural (“cats”)
- **VB:** Verb, base form (“run”)
- **VBD:** Verb, past tense (“ran”)
- **JJ:** Adjective (“quick”)
- **RB:** Adverb (“quickly”)
- **DT:** Determiner (“the”)
- **IN:** Preposition (“in”)
- **PRP:** Personal pronoun (“she”)
- **CC:** Coordinating conjunction (“and”)

The abbreviations also encode details like number (singular/plural) or tense.

### Limitations and Sources of Error

Automated part-of-speech tagging isn’t flawless. Some common issues are:

- **Unknown words:** Taggers may guess the role of a word they haven’t seen before, using spelling or context clues.
- **Ambiguity:** Words with multiple meanings (“can,” “lead,” “bark”) can confuse taggers if the context isn’t clear.
- **Unusual sentence patterns:** Poetry, slang, or highly technical writing can trip up taggers not trained on similar text.
- **Tokenization mistakes:** If a sentence isn’t split into words correctly—such as missing a punctuation mark—tagging will be off.

Despite these hurdles, automated POS tagging is reliable enough to kick-start more advanced NLP tasks. As algorithms and training data improve, so does tagging. Edge cases will always be tricky, but accuracy is high for typical text.

---

## Key Takeaways

- POS tagging labels each word with its grammatical role, like noun or verb.
- Automated tagging uses rule-based and statistical methods to assign tags.
- The Penn Treebank is the most widely used English tagset.
- Ambiguous or unknown words present challenges for tagging accuracy.
- Python's NLTK library can automatically tag words in a sentence.

## Try It Yourself

Write a Python script that asks for a sentence, then uses NLTK to tag each word by part of speech. In your output, pick any two tags assigned by NLTK, look up their meanings in the Penn Treebank tagset, and write a one-sentence explanation for each.

## Further Resources

- 🎥 [Part‑of‑Speech (POS) Tagging Tutorial: Teaching AI Grammar with NLTK & spaCy](https://www.youtube.com/watch?v=N5KRLY4es_A)
- 📘 [Categorizing and Tagging Words — NLTK Book, Chapter 5](https://www.nltk.org/book/ch05.html)
- 📘 [NLTK Tagging Module Documentation](https://www.nltk.org/api/nltk.tag)

---

**Coming up on Day 8:** Named Entity Recognition (NER) Fundamentals