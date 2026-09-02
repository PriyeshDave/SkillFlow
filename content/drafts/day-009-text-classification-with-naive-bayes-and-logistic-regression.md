---
day: 9
generated_at: '2026-09-01T13:45:46.454754+00:00'
phase: Phase 1 — NLP Foundations
recap_summary: Explained the fundamentals of text classification, focusing on bag-of-words
  representations and comparing Naive Bayes and logistic regression classifiers for
  labeling text such as spam detection.
status: pending_review
title: 'Day 9: Text Classification with Naive Bayes and Logistic Regression'
topic_title: Text Classification with Naive Bayes and Logistic Regression
---

**Previously, on Day 8:** Explained Named Entity Recognition (NER), detailing its role in finding and categorizing named entities in text, standard entity types, rule-based and machine learning approaches, practical code examples, and real-world applications and challenges.

---

---
What is Text Classification?

Text classification is the process of automatically assigning categories or labels to text. For example, a system might learn to recognize whether an email is "spam" or "not spam," or determine whether a tweet has "positive," "negative," or "neutral" sentiment.

This goes beyond tasks like tokenization (splitting text into words) or named entity recognition (identifying names and dates). Those focus on extracting specific elements. In contrast, text classification tries to capture the overall meaning or purpose of the entire text and assign a single label to it.

Text classification appears in many places. Spam filters use it to block junk email. News sites use it to sort articles by topic. Apps use it to route support tickets to the correct department. If you want to sort text into buckets based on what it means, you need text classification.

Naive Bayes: A Simple, Classic Approach

Naive Bayes is one of the oldest and simplest algorithms for classification. It's a good starting point for text tasks.

The core idea is probability. The algorithm asks: "Given this text, what is the probability it belongs to each possible category, based on which words are present?" For instance, in spam detection: if a message contains "free," "win," and "money," what are the chances it's spam?

The "naive" part refers to a major simplifying assumption: it treats each feature—here, each word—as if it’s independent of the others, given the label. This means the presence of "free" is assumed unrelated to the presence of "money." In real text, that's rarely true, but the method often still works well. Word usage patterns are strong enough that this rough assumption still separates categories effectively.

Preparing Text Data for Naive Bayes

Machine learning models—including Naive Bayes—work with numbers, not raw text. So, each example needs to be converted into numbers. The most common approach for Naive Bayes is called the bag-of-words model.

Bag-of-words means counting how many times each word appears in a chunk of text, ignoring word order and grammar. For example:

- "Free money now!" → {"free": 1, "money": 1, "now": 1}

If we're building a spam filter, we scan all available spam and non-spam texts, count word occurrences for each group, and then compute the probability of each word appearing in spam vs. non-spam.

> **Note:** In spam detection, there are usually two labels:  
> - `"spam"`: unwanted, promotional, or fraudulent messages  
> - `"ham"`: a standard term for messages that are not spam (i.e., regular, wanted messages)

When a new message arrives, Naive Bayes looks at its words, combines the probabilities (using Bayes’ Rule), and chooses the most likely label.

Naive Bayes for Spam Detection: Minimal Example

Here’s a minimal, real code example using scikit-learn (sklearn). We'll use some sample SMS messages, convert them to bag-of-words vectors, and train/test a Naive Bayes classifier.

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Simple SMS data. For real tasks, use much more data!
texts = [
    "Win a free iPhone now",      
    "Call me as soon as you can", 
    "Cheap meds available online",
    "Hey, are we meeting today?",
    "Congratulations! You won cash.",
    "Let's catch up for lunch."
]
labels = ["spam", "ham", "spam", "ham", "spam", "ham"]

# Indices to keep track of which text is which
text_indices = list(range(len(texts)))

# Step 1: Convert text to bag-of-words counts
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# Show how a text line becomes a vector
sample = texts[0]
print(f"Sample text: '{sample}'")
sample_vec = vectorizer.transform([sample]).toarray()
print("Vector representation:", sample_vec)
print("Vocabulary mapping:", vectorizer.vocabulary_)

# Step 2: Train/test split (keep indices to trace back to text)
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, labels, text_indices, test_size=0.5, random_state=42
)

# Step 3: Train Naive Bayes model
nb_clf = MultinomialNB()
nb_clf.fit(X_train, y_train)

# Step 4: Make predictions and evaluate
y_pred = nb_clf.predict(X_test)
print("Naive Bayes accuracy:", accuracy_score(y_test, y_pred))

# Show predictions for each text
for i in range(len(X_test)):
    text = texts[idx_test[i]]
    label = y_test[i]
    pred = y_pred[i]
    print(f"MSG: '{text}' | TRUE: {label} | NB-PREDICTED: {pred}")
```

How it works:
- The vectorizer transforms each text into a vector of word counts (the bag-of-words).
- Splitting into train and test sets checks that the model generalizes.
- Training fits word probabilities for spam and ham.
- Prediction multiplies those probabilities using Bayes’ Rule to select the best label.

Logistic Regression: A Different Orientation

Logistic regression is another core method for classification. Despite its name, it’s used for classification (categorizing classes), not regression (predicting numbers). Naive Bayes models the probability of seeing this text if it's "spam"; logistic regression directly models the chance that this text *is* spam, given the features.

You can think of logistic regression as fitting a line (technically, a weighted sum plus a sigmoid function to squash the score between 0 and 1) to separate classes. Each word is given a “weight” that pushes a message toward "spam" or "ham." More of a spammy word increases the spam probability; less of it decreases it.

Why use logistic regression? It often achieves higher accuracy than Naive Bayes when you have a decent amount of data, because it can model subtle, overlapping patterns. Its output can be directly interpreted as predicted probabilities.

Logistic Regression Example

Let’s run nearly the same workflow as above, but swap in logistic regression. (You’ll need to have run the previous code block so all variables exist.)

```python
from sklearn.linear_model import LogisticRegression

lr_clf = LogisticRegression(max_iter=1000)
lr_clf.fit(X_train, y_train)

y_pred_lr = lr_clf.predict(X_test)
print("Logistic Regression accuracy:", accuracy_score(y_test, y_pred_lr))

# Compare predictions side by side
for i in range(len(X_test)):
    text = texts[idx_test[i]]
    label = y_test[i]
    nb_prediction = y_pred[i]
    lr_prediction = y_pred_lr[i]
    print(f"MSG: '{text}' | TRUE: {label} | NB: {nb_prediction} | LR: {lr_prediction}")
```

What’s happening:
- Text to numbers (vectorization) is identical to Naive Bayes.
- Logistic regression learns weights per word instead of word probabilities.
- The evaluation process is the same.

Predictions may be different when the labels are ambiguous. Logistic regression’s flexibility helps when classes overlap or the decision isn't obvious.

Comparing Naive Bayes and Logistic Regression

Both methods:
- Use simple numeric representations of text (like bag-of-words).
- Output predictions for each class (with confidence).
- Train and predict quickly on small or mid-sized data.

Key differences:
- **Assumptions**: Naive Bayes assumes that each word is independent given the label; logistic regression does not make this assumption.
- **What’s modeled**: Naive Bayes estimates P(words | label); logistic regression estimates P(label | words).
- **Speed**: Naive Bayes is typically faster and uses less memory, especially with very large vocabularies.
- **Interpretability**: Logistic regression’s weights for each word can be easier to read and explain.
- **Accuracy**: Logistic regression usually outperforms Naive Bayes when you have lots of data and complex boundaries. Naive Bayes works very well when word patterns are clear or the dataset is small.

When to use each:
- **Naive Bayes** is a solid baseline—simple, sturdy, and hard to overfit. Great for quick tests and noisy or limited data.
- **Logistic Regression** is often better for subtle separations or if you add richer features (like word pairs, or TF-IDF weightings).

Limits:
- Both ignore word order unless you create features to handle it (such as bigrams).
- Both struggle with subtleties like sarcasm or complex context that goes beyond words alone.

In practice: Start with Naive Bayes for a first pass. If you have more data or need more nuanced decisions, try logistic regression next. Both are solid foundations for more complex models you'll encounter later.
---

---

## Key Takeaways

- Text classification assigns entire texts to categories like 'spam' or 'ham.'
- Bag-of-words converts text into numeric vectors by counting word occurrences.
- Naive Bayes models word probabilities assuming word independence, making it fast and simple.
- Logistic regression assigns weights to words and often performs better with larger, complex datasets.
- Both methods ignore word order and may struggle with subtle language cues.

## Try It Yourself

Download a small labeled dataset of SMS or email messages. Split the data into training and test sets. Train both a Naive Bayes and a logistic regression classifier to identify spam. Compare their test accuracies and review some cases where the models give different predictions, reflecting on possible reasons for disagreement.

## Further Resources

- 🎥 [Text Classification Explained: Naive Bayes, Logistic Regression, & Fine‑Tuned BERT](https://www.youtube.com/watch?v=1jsTRXZTtq8)
- 📘 [scikit‑learn documentation: Naive Bayes (MultinomialNB, BernoulliNB, ComplementNB)](https://scikit-learn.org/stable/modules/naive_bayes.html)
- 📄 [Lecture 10 – Text Classification with Bag‑of‑Words and TF‑IDF in NLP](https://electuresai.com/lecture-10-text-classification-bag-of-words-tf-idf/)
- 📄 [Text Classification from Scratch: TF‑IDF and Naive Bayes](https://sesen.ai/blog/text-classification-tfidf-naive-bayes)

---

**Coming up on Day 10:** Evaluation Metrics for NLP: Precision, Recall, F1, Confusion Matrix