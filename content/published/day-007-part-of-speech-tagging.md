---
day: 7
generated_at: '2026-08-31T13:49:14.622999+00:00'
phase: Phase 1 — NLP Foundations
recap_summary: Explained part-of-speech (POS) tagging, its importance in natural language
  processing, common parts of speech, ambiguity in tagging, tagging methods (rules,
  statistical models, neural networks), and demonstrated POS tagging using NLTK and
  the Penn Treebank tag set.
status: published
title: 'Day 7: Part-of-Speech Tagging'
topic_title: Part-of-Speech Tagging
---

**Previously, on Day 6:** Explained statistical language models and how n-grams are used to predict word sequences, including calculating probabilities using bigram counts and understanding basic limitations of n-gram models.

---

## What Is Part-of-Speech Tagging?

Part-of-speech tagging, or POS tagging, is the process of labeling each word in a sentence with its part of speech. A part of speech is a category describing how a word functions in a sentence—such as a noun, verb, or adjective.

POS tagging matters for natural language processing (NLP) because many tasks depend on understanding sentence grammar. For example, whether "run" is a noun ("a morning run") or a verb ("to run fast") changes the sentence's meaning. POS tagging is a foundational step in NLP pipelines, including language translation and sentiment analysis.

## A Refresher: What Are Parts of Speech?

English has several main parts of speech. Here are the most common:

- **Noun**: names a person, place, thing, or idea. Example: "cat," "happiness."
- **Verb**: expresses an action or state. Example: "run," "is."
- **Adjective**: describes a noun. Example: "happy," "blue."
- **Adverb**: modifies a verb, adjective, or other adverb. Example: "quickly," "very."
- **Pronoun**: replaces a noun. Example: "she," "it."
- **Preposition**: shows relationships between words. Example: "in," "on," "by."
- **Conjunction**: connects words or sentences. Example: "and," "but."
- **Determiner**: introduces a noun. Example: "the," "a," "some."

Most words belong mainly to one part of speech, but many can play different roles depending on context.

## How Does POS Tagging Work?

POS tagging starts with a sentence. The sentence is split into tokens—usually words, numbers, and punctuation. Each token is then assigned a part of speech tag.

The input is a list of tokens. The output is a list of pairs: (token, POS tag).

For example, for the sentence:  
*"The quick brown fox jumps over the lazy dog."*  
Tokenizing and tagging might give:  
`[("The", "DT"), ("quick", "JJ"), ("brown", "JJ"), ("fox", "NN"), ...]`

Tagging can be tricky because many words have more than one possible tag. For example, "book" can be a noun ("a book on the table") or a verb ("book a flight"). The correct tag depends on context.

## POS Tagging Approaches: Rules, Statistics, and Neural Methods

Early POS taggers used **rule-based systems**. These relied on hand-written patterns. For example, a rule might say "if a word follows 'the' and is not capitalized, it's probably a noun." These systems worked for simple sentences but missed many edge cases.

Statistical models, such as the **Hidden Markov Model (HMM)**, came next. These learn from large amounts of annotated text, estimating the probabilities of different tags for each word in context.

Modern POS taggers often use **machine learning**, including neural networks. These methods need large labeled datasets but can learn complex patterns and handle ambiguous cases better.

Despite changing methods—rules, statistics, or neural networks—the goal stays the same: find the right tag for each word in context.

## Ambiguity in POS Tagging: Why Context Matters

Many words have more than one possible part of speech. To pick the right one, taggers need context.

For example, the word "can":

- In "I can swim," "can" is a verb (in this case, an auxiliary verb helping "swim").
- In "Pass me a can," "can" is a noun (an object).

Or look at "flies":

- "Time flies quickly." Here, "flies" is a verb (what time does).
- "Fruit flies are pests." Here, "flies" is a noun (the insect).

Taggers use surrounding words to make the right choice. Without context, it's easy to pick the wrong tag.

## Hands-on: POS Tagging with NLTK

[NLTK (Natural Language Toolkit)](https://www.nltk.org/) is a popular Python library for working with language. It can tokenize and POS-tag English text with minimal code.

Here’s a basic example. This script takes a sentence, splits it into words, tags each word with its part of speech, and prints the results.

```python
import nltk

# Download these the first time you use NLTK (uncomment to run them)
# nltk.download("punkt")
# nltk.download("averaged_perceptron_tagger")

sentence = "Can you can a can as a canner can can a can?"
tokens = nltk.word_tokenize(sentence)  # Split the sentence into word tokens

# Tag each token with its part of speech
tagged = nltk.pos_tag(tokens)

for word, tag in tagged:
    print(f"{word:10} -> {tag}")
```

Line-by-line breakdown:
- `import nltk` loads the library.
- Download commands fetch pre-trained models (run once).
- `sentence` contains the text you want to tag.
- `nltk.word_tokenize(sentence)` splits the sentence into tokens.
- `nltk.pos_tag(tokens)` tags each token.
- The loop prints each word and its tag.

Change the example sentence to see how POS tags change in different contexts.

## Understanding the Output: POS Tag Sets

Taggers use a "tag set"—a list of labels for different parts of speech. NLTK’s default is the **Penn Treebank tag set**, widely used in American English NLP.

Some common tags you’ll see:

- **NN**: noun, singular ("cat")
- **NNS**: noun, plural ("cats")
- **VB**: verb, base form ("run")
- **VBD**: verb, past tense ("ran")
- **JJ**: adjective ("blue")
- **RB**: adverb ("quickly")
- **DT**: determiner ("the")
- **IN**: preposition or subordinating conjunction ("in", "because")

If you're unsure about a tag, look up the [Penn Treebank tag set online](https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html) for details.

In the code example above, "can" appears multiple times, in both noun and verb contexts. The tagger uses context to try to assign the right tag each time. This demonstrates how POS tagging is contextual and automatic, but can still make mistakes with tricky sentences or rare usages.

Understanding POS tagging is key for many downstream NLP tasks. It shows how computers start to interpret language—using patterns, categories, and context to make sense of text.

---

## Key Takeaways

- POS tagging labels each word in a sentence with its grammatical function.
- Tagging accuracy depends heavily on context due to word ambiguity.
- Modern POS taggers use statistical and neural methods for higher accuracy.
- NLTK in Python provides easy tools for POS tagging with the Penn Treebank tag set.
- Understanding output tag codes is crucial for interpreting POS tagging results.

## Try It Yourself

Pick three short English sentences—either provided or of your own choosing. Tokenize and POS-tag each sentence using NLTK. For any POS tags you don't recognize, consult the Penn Treebank tag guide to find their meaning.

## Further Resources

- 🎥 [Part Of Speech POS Tagging: NLP Tutorial For Beginners – codebasics (YouTube)](https://www.youtube.com/watch?v=gdHWoQWZGkk)
- 🎥 [POS Tagging | Part of Speech Tagging in NLP | Hidden Markov Models in NLP | Viterbi Algorithm in NLP – CampusX (YouTube)](https://www.youtube.com/watch?v=269IGagoJfs)
- 📘 [Tagger · spaCy API Documentation](https://spacy.io/api/tagger/)
- 📄 [5. Categorizing and Tagging Words – NLTK Book Chapter (NLTK)](https://www.nltk.org/book/ch05.html)
- 📄 [Part‑Of‑Speech Tagging With Hidden Markov Model – Baeldung](https://www.baeldung.com/cs/nlp-hmm-pos-tags)

---

**Coming up on Day 8:** Named Entity Recognition (NER) Fundamentals