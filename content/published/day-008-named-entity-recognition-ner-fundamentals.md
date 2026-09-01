---
day: 8
generated_at: '2026-08-31T16:28:29.069925+00:00'
phase: Phase 1 — NLP Foundations
recap_summary: Explained Named Entity Recognition (NER) as an NLP technique for detecting
  and categorizing real-world entities (such as people, organizations, and locations)
  in text, including typical entity types, system approaches, and practical usage
  with spaCy.
status: published
title: 'Day 8: Named Entity Recognition (NER) Fundamentals'
topic_title: Named Entity Recognition (NER) Fundamentals
---

**Previously, on Day 7:** Explained part-of-speech (POS) tagging, its importance in NLP, different approaches to tagging, common tagsets, and typical challenges such as ambiguity and unknown words.

---

### What is Named Entity Recognition (NER)?

Named Entity Recognition (NER) is a key task in natural language processing (NLP). Its job is to automatically find and label “named entities” in text.

A **named entity** is a word or phrase referring to a specific person, place, organization, date, value, or other real-world item. These are usually proper nouns or names, such as “London,” “Microsoft,” or “January 1, 2022.” NER systems identify these spans of text and label what kind of entity each one is.

Some common categories for entities (called **entity types**) include:

- **Person** (names of people)
- **Organization** (companies, agencies, institutions)
- **Location** (cities, landmarks, continents)
- **Date/Time** (calendar dates, clock times)
- **Monetary values** (prices, amounts of money)
- **Percentages** (percent numbers)

NER is important because most of the useful information in text involves knowing *what* is mentioned, not just *which words* appear. For example, extracting all company names, people, or places from a document lets you answer questions and organize information in ways that raw text cannot.

### How NER Fits with Other NLP Tasks

NER is one building block among others in NLP. It focuses on extracting “things” mentioned in text, rather than analyzing grammar.

- **Tokenization** splits text into words and symbols. For example, “Mr. Smith went to Paris.” becomes `[Mr., ., Smith, went, to, Paris, .]`.
- **Part-of-Speech (POS) Tagging** labels words as nouns, verbs, etc. (Reminder: We covered this yesterday. For example, “went” is a verb, “Paris” is a noun.)
- **Chunking** groups words into phrases like “the tall building.” This captures structure, but doesn’t tell you what kind of thing the phrase is.

NER builds on these steps. It looks at the tokens and phrases to find meaningful real-world entities, and labels their type (person, place, organization, etc.). For instance, “Steve Jobs” is a noun phrase—but NER identifies it specifically as a PERSON entity.

### Common Entity Types in NER

Most NER tools use a handful of carefully chosen types:

- **Person (PERSON):** Specific people or fictional characters  
  *Example:* “Barack Obama”, “Harry Potter”
- **Organization (ORG):** Companies, groups, or institutions  
  *Example:* “Google”, “United Nations”
- **Location (LOC):** Physical places (cities, mountains, rivers)  
  *Example:* “Mount Everest”, “Nile River”
- **Geo-Political Entity (GPE):** Countries, states, cities  
  *Example:* “France”, “California”
- **Date (DATE):** Dates or ranges  
  *Example:* “April 10, 2022”, “last Friday”
- **Time (TIME):** Times or durations  
  *Example:* “8:00 PM”, “two hours”
- **Money (MONEY):** Monetary amounts  
  *Example:* “$100”, “fifty euros”
- **Percent (PERCENT):** Percentages  
  *Example:* “10%”, “half of the people”

Different tools or datasets might include more or fewer categories, but these are the most common.

### How NER Systems Work

NER systems have developed over time, moving from simple pattern matching to advanced machine learning:

- **Rule-Based NER** uses hand-coded patterns. For example, it might look for “Mr. [Capitalized Word]” to find personal names. This is limited and doesn’t adapt well.
- **Dictionary or Lookup-Based NER** checks for matches against lists of known entities, like cities or company names. This is fast, but misses new or misspelled names.
- **Statistical Models** like Hidden Markov Models (HMMs) or Conditional Random Fields (CRFs) learn from labeled examples. These can spot patterns in contexts rather than just matching strings exactly.
- **Neural NER** uses neural networks, including models like BERT. These systems learn from large datasets and can generalize to names and phrases they have never seen before.

In every approach, NER takes a chunk of text and returns a list of entity spans, with a label for each one.

For example, with the sentence “Tim Cook runs Apple in Cupertino.” an NER system might return:

- “Tim Cook” (PERSON)
- “Apple” (ORG)
- “Cupertino” (GPE)

### NER Input and Output

Before NER processes text, it’s usually prepared in a couple of steps:

- The text is split into sentences.
- Each sentence is split into tokens (words or punctuation marks).

NER usually labels each token using the **BIO** tag format (also called **IOB**):

- **B-XXX:** Beginning of an entity of type XXX (like B-PERSON)
- **I-XXX:** Inside an entity of type XXX
- **O:** Outside any named entity

Example with the sentence:  
“Steve Jobs founded Apple in California.”

| Token       | Tag        |
|:----------- |:----------|
| Steve       | B-PERSON  |
| Jobs        | I-PERSON  |
| founded     | O         |
| Apple       | B-ORG     |
| in          | O         |
| California  | B-GPE     |
| .           | O         |

Some NER tools also output the “span” (start and end token indices):

- Tokens 0–2: “Steve Jobs” (PERSON)
- Token 3: “Apple” (ORG)
- Token 5: “California” (GPE)

The core idea: the system identifies *which part* of the text is an entity, and *what kind* it is.

### Running NER with spaCy

You don’t need to build NER from scratch to see it work. Tools like **spaCy** offer pre-trained English NER models.

Here’s a minimal code example. Given this sentence:

“Apple was founded by Steve Jobs in California.”

Let’s run NER with spaCy:

```python
import spacy

# Load a small English model with NER support
nlp = spacy.load("en_core_web_sm")

text = "Apple was founded by Steve Jobs in California."

doc = nlp(text)

for ent in doc.ents:
    print(ent.text, ent.label_)
```

Output:
```
Apple ORG
Steve Jobs PERSON
California GPE
```

- “Apple” is identified as an ORG (organization).
- “Steve Jobs” as a PERSON.
- “California” as a GPE (geo-political entity, like a state).

You can run this if you have spaCy installed (`pip install spacy` and `python -m spacy download en_core_web_sm`). The important point is that NER gives back both the exact words (“spans”) and a label for each, showing which things are mentioned in text.

Knowing who, what, and where is foundational for building search engines, news apps, chatbots, and more. NER takes the raw stream of words and tells you what real-world things are there.

---

## Key Takeaways

- NER detects specific entities like people, places, and organizations in text.
- Common NER entity types include PERSON, ORG, LOC, GPE, DATE, TIME, MONEY, and PERCENT.
- NER can be implemented using rule-based, dictionary-based, statistical, or neural network methods.
- The BIO tag format is widely used to mark entity spans in tokenized text.
- spaCy provides pre-trained NER models for easy extraction of entities from English text.

## Try It Yourself

Write a short Python script using spaCy to analyze three sentences you invent. For each sentence, extract all named entities and print both the entity text and its predicted label. Experiment with different kinds of names (people, places, dates, etc.) to see which types are detected.

## Further Resources

- 🎥 [Named Entity Recognition (NER): NLP Tutorial For Beginners – S1 E12](https://www.youtube.com/watch?v=2XUhKpH0p4M)
- 📄 [Named Entity Recognition: How NLP Identifies People, Places & Organizations](https://fondralabs.com/blog/nlp-foundations/named-entity-recognition-how-nlp-identifies-people-places-organizations.html)
- 📄 [Named Entity Recognition | Baeldung on Computer Science](https://www.baeldung.com/cs/ner-nlp)

---

**Coming up on Day 9:** Text Classification with Naive Bayes and Logistic Regression