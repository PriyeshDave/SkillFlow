---
day: 10
generated_at: '2026-09-02T13:02:38.461894+00:00'
phase: Phase 1 — NLP Foundations
recap_summary: Explained core evaluation metrics for NLP classification tasks, including
  accuracy, precision, recall, F1 score, and the confusion matrix, highlighting their
  importance and pitfalls in real-world scenarios.
status: pending_review
title: 'Day 10: Evaluation Metrics for NLP: Precision, Recall, F1, Confusion Matrix'
topic_title: 'Evaluation Metrics for NLP: Precision, Recall, F1, Confusion Matrix'
---

**Previously, on Day 9:** Explained the fundamentals of text classification, focusing on bag-of-words representations and comparing Naive Bayes and logistic regression classifiers for labeling text such as spam detection.

---

Model evaluation means measuring how well your NLP system performs a specific task. These measurements, called metrics, offer concrete evidence of your model’s strengths and weaknesses. For engineers, metrics are not just numbers—research, engineering choices, and business decisions all depend on them.

**Accuracy: The Basic Metric**

Accuracy tells you what fraction of predictions were correct. But accuracy alone can mislead, especially when certain mistakes cost more, or when your data is imbalanced.

Take a spam filter. If only 5% of emails are spam, a model that always predicts “not spam” scores 95% accuracy. It looks good on paper, but in reality, the system fails at catching spam.

Accuracy is just a starting point. Most real-world problems need more nuanced metrics.

---

**Precision: Correctness of Positive Predictions**

Precision measures how many of the items your model labeled as positive were actually positive.

In plain terms: if your spam filter marks 10 emails as spam, and only 8 are really spam, your model made 2 mistakes—it blocked good email.

Precision focuses only on "is spam" predictions. Out of everything marked as spam, how many truly were?

**Precision = (True Positives) / (True Positives + False Positives)**

Spam filter example:
- 8 spam emails correctly blocked (true positives)
- 2 valid emails wrongly blocked (false positives)
  
Precision = 8 / (8 + 2) = 0.8, or 80%.

High precision means that when your model says "spam," it’s probably right. Systems tuned for high precision rarely block important messages—but may let some spam through.

---

**Recall: Coverage of Real Positives**

Recall measures how many actual positive items your model found.

Returning to the spam filter. Suppose there were 12 spam emails in your inbox, but your model only caught 8.

**Recall = (True Positives) / (True Positives + False Negatives)**

Here:
- True positives: 8 spam emails caught
- False negatives: 4 spam emails missed and let through

Recall = 8 / (8 + 4) = 8 / 12 ≈ 0.67, or 67%.

High recall means the model finds nearly all the spam. But if recall is too high at the cost of precision, it may block good mail as well.

Precision and recall often work against each other. Improving one can lower the other.

---

**F1 Score: The Balance Point**

When you want a single measure that balances precision and recall, use the F1 score. It’s the harmonic mean—if either precision or recall is low, the F1 score drops sharply.

**F1 = 2 × (Precision × Recall) / (Precision + Recall)**

With our example:
- Precision = 0.8
- Recall = 0.67

F1 = 2 × (0.8 × 0.67) / (0.8 + 0.67)  
F1 = 2 × 0.536 / 1.47 ≈ 1.072 / 1.47 ≈ 0.73, or 73%.

If either precision or recall is very low, the F1 score will also be low.

---

**Confusion Matrix: The Error Breakdown**

Metrics can be abstract. The confusion matrix makes results easy to see and analyze.

A confusion matrix is a table showing the count of true and false outcomes for each class. For binary problems (like spam vs. not spam), it looks like this:

|                 | Predicted: Spam | Predicted: Not Spam |
|-----------------|----------------|--------------------|
| **Actual: Spam**     | True Positive (TP)   | False Negative (FN)      |
| **Actual: Not Spam** | False Positive (FP)  | True Negative (TN)       |

- **True Positive (TP):** Spam predicted as spam.
- **False Positive (FP):** Not-spam predicted as spam.
- **True Negative (TN):** Not-spam predicted as not-spam.
- **False Negative (FN):** Spam predicted as not-spam.

From this table, you can always calculate:
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)

Here’s a bare-bones Python example:

```python
# Sample labels (1=spam, 0=not spam)
actual    = [1, 0, 1, 1, 0, 0, 1, 0, 1, 0]
predicted = [1, 0, 1, 0, 0, 1, 1, 0, 0, 0]

# Compute confusion matrix entries
TP = sum(1 for a, p in zip(actual, predicted) if a == 1 and p == 1)
FP = sum(1 for a, p in zip(actual, predicted) if a == 0 and p == 1)
TN = sum(1 for a, p in zip(actual, predicted) if a == 0 and p == 0)
FN = sum(1 for a, p in zip(actual, predicted) if a == 1 and p == 0)

print(f"TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}")

# Now calculate the metrics
precision = TP / (TP + FP) if (TP + FP) else 0
recall = TP / (TP + FN) if (TP + FN) else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
```

This prints the confusion matrix numbers, then precision, recall, and F1.

---

**Metric Pitfalls: When Numbers Mislead**

Metrics are tools. The wrong tool can lead to the wrong decisions.

If your data has 1,000 emails, but only 10 are spam, always predicting "not spam" means 99% accuracy. But you'll miss all the spam—accuracy is useless here.

In medical tests, recall might matter more (catch all disease cases). In email, high precision may matter more (never lose real mail). Sometimes you want a balance (F1); sometimes you care much more about one type of error than another.

Always connect your metrics to your real-world goals. Analyze your confusion matrix to understand your model’s errors. Never trust a single number without asking whether it matches your needs.

Metrics measure, compare, and drive improvement. But their value always comes from how thoughtfully you choose, compute, and interpret them.

---

## Key Takeaways

- Accuracy alone can be misleading, especially with imbalanced data.
- Precision measures how often positive predictions are actually correct.
- Recall reflects how many real positives your model identifies.
- The F1 score balances precision and recall into a single metric.
- A confusion matrix shows detailed error counts, revealing strengths and weaknesses.

## Try It Yourself

Take these actual labels: [1, 0, 1, 1, 0, 0, 1, 0, 1, 0] and predictions: [1, 0, 1, 0, 0, 1, 1, 0, 0, 0]. Manually fill in a confusion matrix by counting true positives, true negatives, false positives, and false negatives. Then calculate precision, recall, and F1 score for this data set, showing your steps.

## Further Resources

- 🎥 [Beginner's tutorial on Precision Recall and F1 Score for Machine Learning Models | Confusion Matrix](https://www.youtube.com/watch?v=JYQupddZkzc)
- 📄 [Evaluating Classifiers: Confusion Matrix, Precision, Recall, and F1](https://intuitivetutorial.com/2026/08/25/evaluating-classifiers-confusion-matrix-precision-recall-and-f1/)
- 📘 [Model Evaluation in Scikit‑Learn — Metrics: precision, recall, F‑beta, F1, confusion matrix](https://scikit-learn.org/stable/modules/model_evaluation.html)
- 📘 [6. Learning to Classify Text (NLTK book chapter)](https://www.nltk.org/book/ch06.html)

---

**Coming up on Day 11:** Why One-Hot Encoding Fails: The Case for Embeddings