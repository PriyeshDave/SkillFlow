---
day: 8
generated_at: '2026-09-01T10:55:47.956941+00:00'
phase: Phase 1 — NLP Foundations
recap_summary: Explained named entity recognition (NER) in NLP, covering entity types,
  the BIO annotation scheme, rule-based versus machine learning approaches, and practical
  NER using spaCy.
status: published
title: 'Day 8: Named Entity Recognition (NER) Fundamentals'
topic_title: Named Entity Recognition (NER) Fundamentals
---

**Previously, on Day 7:** Explained part-of-speech (POS) tagging, its importance in NLP, different approaches to tagging, common tagsets, and typical challenges such as ambiguity and unknown words.

---

### What Is Named Entity Recognition?

Named Entity Recognition (NER) is a key task in natural language processing (NLP). NER finds and labels "named entities" in text. A named entity is a real-world thing—such as a person, location, or organization—that has a specific name. For example, in “Alice went to Paris,” “Alice” is a person and “Paris” is a location.

NER is different from Part-of-Speech (POS) tagging. POS tagging labels each word’s role in the sentence (like noun, verb, or adjective). NER instead asks: is this word (or group of words) a name for a real-world thing, and what kind?

NER powers many applications. Search engines use it to find people or companies in news stories. Chatbots use it to spot product names or locations. In general, NER helps machines find the “important things with names” inside text.

Common entity examples include:
- Person names (e.g., "Elon Musk")
- Locations (e.g., "New York")
- Organizations (e.g., "Google")
- Dates (e.g., "April 27th")
- Money values (e.g., "$100")

The list can vary with the application.

### How NER Works: From Tokens to Entities

NER starts by splitting text into pieces called “tokens.” A token is usually a word or a punctuation mark. The NER system examines each token (and often nearby tokens) to decide if it forms part of a named entity, and if so, of what type.

Example:

- Input: `Barack Obama was born in Hawaii.`
- Tokenized: [`Barack`, `Obama`, `was`, `born`, `in`, `Hawaii`, `.`]
- Labeled: [`Barack` (begin person), `Obama` (inside person), `was` (O), `born` (O), `in` (O), `Hawaii` (begin location), `.` (O)]

Here, "Barack Obama" is recognized together as a person. "Hawaii" is labeled as a location. “O” means that token is outside any named entity.

NER’s challenge is to get these groupings right, using the words and their context.

### Common Entity Types and the BIO Annotation Scheme

Most NER systems use a standard set of entity types. The most common are:
- **PERSON:** A person, real or fictional
- **ORG (Organization):** A company or group
- **GPE (Geo-Political Entity):** A country, city, or state
- **LOC (Location):** A place that is not a GPE (like rivers, mountains)
- **DATE:** A date or period of time
- **MONEY:** A monetary value

Others are possible, but these cover many use cases.

To show where each entity starts and ends in the text, NER uses an annotation scheme. The most popular is **BIO**:
- **B (Begin):** First token of an entity
- **I (Inside):** Later token of the same entity
- **O (Outside):** Not part of any entity

For example:

Sentence: `Steve Jobs founded Apple in California.`
Annotated:
- Steve (B-PERSON)
- Jobs (I-PERSON)
- founded (O)
- Apple (B-ORG)
- in (O)
- California (B-GPE)
- . (O)

BIO tags help the system mark exactly which words belong to which entities, including names that are more than one word long.

### Rule-Based vs. Machine Learning NER

NER systems generally fall into two types: rule-based or machine learning-based.

**Rule-Based NER** uses patterns, dictionaries, or manual rules. For instance, if a word is in a list of cities, mark it as a location. Regular expressions can spot certain formats (like dates or money).

- **Strengths:** Good for narrow, clear tasks. Rules are easy to understand. High precision if cases are simple.
- **Weaknesses:** Can’t handle new or unusual names. Hard to maintain and scale. Not flexible for messy real-world data.

**Machine Learning-Based NER** trains a model using labeled examples. The model learns the patterns that signal the start, end, and type of entities. Most modern NER systems—including spaCy and Hugging Face—use this approach.

- **Strengths:** Can handle new entities if context is similar to training examples. Handles varied text better. Adapts to real-world complexity.
- **Weaknesses:** Needs lots of labeled data. Mistakes can be hard to understand or fix. Quality depends heavily on the training data.

Rule-based methods work if the domain is small and data is limited. For most real-world applications, machine learning is the default choice.

### Minimal NER Example with spaCy

Let’s see NER in action using spaCy, a powerful Python NLP library.

```python
import spacy

# Load spaCy's English NER model (download 'en_core_web_sm' first if needed)
nlp = spacy.load("en_core_web_sm")

text = "Elon Musk founded SpaceX in California in 2002."

doc = nlp(text)

for ent in doc.ents:
    print(ent.text, ent.label_)
```

This script:
- Loads a pre-trained English model that knows how to spot entities.
- Processes a simple sentence.
- Prints out each detected entity and its type, producing:
  ```
  Elon Musk PERSON
  SpaceX ORG
  California GPE
  2002 DATE
  ```

This is all it takes to extract named entities from text with spaCy’s out-of-the-box model.

### NER Limitations and Common Pitfalls

NER works well much of the time, but has real limits:

- **Ambiguous Names:** "Apple" can mean the fruit or the company. "Jordan" might be a country or a person. The system has to guess from context, and can get it wrong.
- **Unseen Entities:** New companies, rare places, or odd spellings (“SpceX” instead of “SpaceX”) might be missed.
- **Context Sensitivity:** The same word can mean different things in different contexts. "Jaguar" is an animal at a zoo, but a car brand in an auto review.
- **Nested or Overlapping Entities:** Sometimes one name is inside another or entities overlap. Most NER systems aren't built to handle this.

Many NER models are trained on news or “clean” text. They often perform worse on tweets, technical documents, or text with lots of slang or typos.

Keep these limitations in mind. NER is a powerful tool, but not foolproof. Understanding what it does (and doesn’t do) helps you use it effectively in real applications.

---

## Key Takeaways

- NER finds and labels real-world entities like people, places, and organizations in text.
- The BIO tagging scheme marks entity boundaries with Begin, Inside, and Outside labels.
- Rule-based NER uses hand-crafted patterns; machine learning-based NER learns from labeled examples.
- spaCy allows easy extraction of named entities with just a few lines of Python code.
- NER systems have limitations with ambiguous, unseen, or context-dependent names.

## Try It Yourself

Pick a few sentences from recent news articles or Wikipedia. Run them through the provided spaCy code and list all named entities detected, along with their types. For any entities that are missed or mislabeled, note what happened and think about possible reasons (such as ambiguity or unusual spelling).

## Further Resources

- 🎥 [Named Entity Recognition (NER): NLP Tutorial For Beginners - S1 E12](https://www.youtube.com/watch?v=2XUhKpH0p4M)
- 📘 [Chapter 1: Finding words, phrases, names and concepts · Advanced NLP with spaCy](https://course.spacy.io/en/chapter1/)
- 📄 [Named Entity Recognition: Essentials, Workings, Approaches, Challenges](https://blog.unitlab.ai/named-entity-recognition/)
- 📄 [Named Entity Recognition: How NLP Identifies People, Places & Organizations](https://fondralabs.com/blog/nlp-foundations/named-entity-recognition-how-nlp-identifies-people-places-organizations.html)

---

**Coming up on Day 9:** Text Classification with Naive Bayes and Logistic Regression