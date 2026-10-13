---
day: 35
generated_at: '2026-10-12T18:18:44.622688+00:00'
phase: Phase 5 — Pretraining & Language Models
recap_summary: Explained how BERT uses bidirectional transformer encoders and masked
  language modeling to create contextualized word representations, enabling state-of-the-art
  understanding and prediction in NLP tasks. Introduced the concepts of tokenization,
  attention, fine-tuning, and provided a code example for masked token prediction.
status: published
title: 'Day 35: BERT: Architecture and Masked Language Modeling'
topic_title: 'BERT: Architecture and Masked Language Modeling'
---

**Previously, on Day 34:** Explained the concept of pretraining in NLP, highlighted why it is foundational, introduced transfer learning, and demonstrated how pretrained models are used in practice with Hugging Face Transformers.

---

What is BERT? Contextualized Language Representations
-----------------------------------------------------

BERT stands for **Bidirectional Encoder Representations from Transformers**. It's a language model—a machine learning system trained to read, understand, and generate human language.

Earlier models read text in only one direction: either left to right or right to left. If you asked them to predict a missing word, they could only use the words before or after the blank—not both at once.

BERT changed this with **bidirectionality**. BERT reads text in both directions at the same time. When BERT tries to understand or predict a word, it looks at the whole context—everything before and after the target word. In natural language processing (NLP), this is called **contextualization**: the meaning of a word is adjusted based on the words around it.

For example, “bank” in “river bank” is different from “bank” in “savings bank.” Earlier models sometimes missed this. BERT uses the whole sentence to find the right meaning for every word, every time. This context-aware reasoning is a big reason for BERT’s accuracy and popularity.

BERT’s Architecture: Key Building Blocks
----------------------------------------

BERT is built from a stack of **transformer encoder blocks**. These blocks process the entire sentence at once, letting every word send and receive information from every other word.

Here’s what goes into BERT processing:

- **Tokens**: BERT doesn’t work with full words. It works with *tokens*—often pieces of words, called subwords. A tokenizer breaks text into tokens and converts each token to a unique number.

- **Segments**: BERT can process two partitions of text together, such as a question and its answer. Each token gets a label marking which segment it comes from.

- **Positions**: Transformers have no built-in sense of order. So BERT adds a *position embedding* to each token to indicate where it appears in the sentence.

- **Encoder blocks & Attention**: Each encoder block uses **attention**. Attention means that, for every token, the model decides how much to “pay attention” to every other token. These attention weights help BERT understand connections, even between far-apart words.

Imagine BERT's input like a grid: each row is a token, labeled with its segment and position, with attention creating links between all pairs of tokens in each layer.

Masked Language Modeling (MLM)
------------------------------

Traditional language models predict the *next* word in a sentence. BERT uses **Masked Language Modeling** (MLM), which works differently.

Here’s how MLM training works:

- Some tokens in the input are randomly replaced with a special token: `[MASK]`.
- BERT doesn’t know what `[MASK]` means. Its task is to use the rest of the sentence to guess the original word.
- For example:
  - Input: “The cat [MASK] on the mat.”
  - BERT needs to predict that `[MASK]` should be “sat”.

Why mask random words?

1. Unlike left-to-right models, BERT uses context from both left and right to make its guess.
2. This forces BERT to actually understand the whole sentence—where every word matters, not just the next word.

This approach teaches BERT to model words in any position, based on their full context.

A Step-by-Step Example: How BERT Learns
----------------------------------------

Let’s follow what happens on a single training step:

Consider the real sentence:

> “The [MASK] barked loudly in the park.”

Let’s say “[MASK]” covers the word “dog.”

**1. Tokenization and Masking**
- Split into tokens: [“The”, “[MASK]”, “barked”, “loudly”, “in”, “the”, “park”, “.”]
- Each token is mapped to a unique integer.

**2. Input Assembly**
- Each token is converted into an embedding (a list of numbers).
- Segment and position information are added.

**3. Encoding**
- Embeddings flow through the stack of encoder blocks.
- Attention lets tokens (like “[MASK]”) use information from all the other words, both before and after.

**4. Mask Prediction**
- At `[MASK]`, BERT combines the contexts and outputs a guess—a probability for each possible word.
- In our example, “dog” should have the highest probability.

**5. Loss and Update**
- BERT compares its guess to the correct word (“dog”).
- It computes a loss (a measure of how wrong it was), then tweaks its internal weights to do better next time.

Repeat this process millions of times, and BERT becomes great at filling in blanks and understanding all kinds of language patterns.

Fine-Tuning: Using BERT for Real Tasks
--------------------------------------

MLM pretraining gives BERT strong general language skills, but practical NLP tasks—like classifying sentences or extracting names—are different.

To make BERT useful for a specific task, we **fine-tune** it: start with the general pretrained BERT, then adjust (train) it on a small, labeled dataset for your target problem.

The process:

- Add a simple new task-specific layer on top (like a classifier).
- Train on labeled data for a few more steps.
- BERT keeps its language skills; the new layer learns how to use them for your task.

This *transfer learning* method was a breakthrough. With BERT, almost any NLP task can see big improvements, often with minimal extra training.

Minimal Code Example: Masked Token Prediction with BERT
-------------------------------------------------------

Here’s how BERT predicts a masked word using the Hugging Face Transformers library:

```python
from transformers import BertTokenizer, BertForMaskedLM
import torch

# Load pre-trained BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

# Example sentence with a masked token
sentence = "The quick brown [MASK] jumps over the lazy dog."
inputs = tokenizer(sentence, return_tensors='pt')

# Find the position of [MASK]
mask_token_index = torch.where(inputs["input_ids"][0] == tokenizer.mask_token_id)[0]

# Pass the sentence through BERT
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# Get BERT's top predictions for [MASK]
mask_token_logits = logits[0, mask_token_index, :]
top_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()

print("Top predictions for [MASK]:")
for token in top_tokens:
    print(tokenizer.decode([token]))
```

This code masks the word “fox” in a familiar sentence. BERT examines the context on both sides and predicts the missing word. The top guesses almost always include “fox,” showing BERT’s strong understanding of how language works.

Summary
-------

BERT introduced a powerful new approach for machine understanding of language: using bidirectional, contextualized representations built with transformers and attention. By learning to predict masked words using all available context, BERT sets a foundation for most modern NLP systems. Its bidirectional context-awareness is now the standard for how machines should interpret language.

---

## Key Takeaways

- BERT reads text bidirectionally, using context from both sides of each word.
- Tokenization, segment and position embeddings, and attention enable contextual understanding.
- Masked Language Modeling trains BERT to predict masked words using the full sentence context.
- Fine-tuning allows BERT to adapt to specific NLP tasks with minimal new training.
- BERT's transformer architecture is now foundational for many modern NLP systems.

## Try It Yourself

Choose a short sentence (such as 'The sun rises in the east.') and manually replace one word with [MASK]. Use Hugging Face Transformers to encode the sentence and predict the masked token with BERT, as shown in the lesson's code example. Compare the top predictions to your expectations and reflect on how the context affects BERT's choices.

## Further Resources

- 🎥 [BERT Explained Like Never Before: Architecture + Masked Language Modeling (Full Guide)](https://www.youtube.com/watch?v=XiNw__ozYGQ)
- 🎥 [Bert‑ masked language modelling & applications — Lecture by IIT Madras](https://www.youtube.com/watch?v=0q18FupeywU)
- 📘 [Hugging Face Documentation: BERT model](https://huggingface.co/docs/transformers/model_doc/bert)
- 📘 [Masked language modeling — Hugging Face Transformers guide](https://huggingface.co/docs/transformers/v4.27.0/en/tasks/masked_language_modeling)
- 📄 [Dive into Deep Learning, Chapter 15.8: BERT Pretraining Tasks (Masked Language Modeling & NSP)](https://d2l.ai/chapter_natural-language-processing-pretraining/bert.html)

---

**Coming up on Day 36:** Fine-Tuning BERT for Downstream Tasks