---
day: 38
generated_at: '2026-10-15T13:46:13.292000+00:00'
phase: Phase 5 — Pretraining & Language Models
recap_summary: Explained the structure and evolution of GPT models, detailing the
  shift from GPT-1 to GPT-3, the significance of scale, and how these models generate
  coherent text based on next word prediction using transformers.
status: pending_review
title: 'Day 38: GPT-1 to GPT-3: The Evolution of Scale'
topic_title: 'GPT-1 to GPT-3: The Evolution of Scale'
---

## What Are GPT Models?

GPT stands for **Generative Pretrained Transformer**. Each part of this name tells you something important.

- **Generative:** The model creates things. Here, that means writing new text—sentences, paragraphs, or full articles—rather than just classifying or picking existing text.
- **Pretrained:** The model is first trained very broadly, reading huge amounts of text (like books and websites). Only after that is it fine-tuned for specific tasks.
- **Transformer:** This is a type of neural network introduced in 2017. Unlike older models that read text bit by bit, transformers can look at all the words in a sentence at once, capturing more context at once.

A GPT model reads massive datasets, spots patterns in how words appear together, and uses those patterns to generate text that sounds natural—sometimes uncannily so.

## The Core Idea Behind GPT

Think of a fill-in-the-blank game:

_"The cat sat on the \_\_\_."_

You might guess "mat," using your intuition for what fits.

GPT models, during training, play this game at enormous scale. But rather than filling blanks, they predict the next word in a sentence. So if you give:

_"In the early morning, the birds started to"_

the model needs to generate the next likely word, such as "sing."

Doing this billions of times, GPT learns subtle patterns in language. This guessing-the-next-word skill, known as **next word prediction**, is the core trick it uses for writing, summarizing, translating, and answering questions.

## The Leap: GPT-1

Before GPT-1, most language models used simple tricks like counting word pairs (called **n-grams**), or small neural networks. These older models saw only a narrow window of text at a time.

**GPT-1** changed the game in 2018. It used the transformer structure, so it could take in much more context from each sentence. GPT-1 had **117 million parameters**. Each parameter is like a small dial adjusted during training that helps the model get better at matching real text.

GPT-1 was trained on the BooksCorpus dataset—about 7,000 unpublished books. Instead of retraining from scratch for every task, GPT-1 could now be **fine-tuned**: given small extra “practice” to specialize for new jobs. This made it far more flexible.

## Scaling Up: GPT-2

GPT-2 arrived in 2019, taking the same basic design and making it much larger.

- **Parameters:** 1.5 billion (over 10 times GPT-1)
- **Training Data:** About 40 GB of public internet text (much wider variety than just books)

This jump in size and data led to writing that was much more convincing. GPT-2 could now write news articles, answer questions, and summarize while mostly avoiding the awkward, abrupt sentences you’d see from earlier models.

**Example:**

**With GPT-1:**
```
Prompt: The robot entered the room and
AI: turned on the light. It was happy. Book is interesting.
```
Notice how the last sentence doesn’t really fit.

**With GPT-2:**
```
Prompt: The robot entered the room and
AI: turned on the light, scanning the area for signs of movement. Its sensors flickered as it crept forward.
```
Here, the model keeps the story consistent and logical.

## The Power of Massive Scale: GPT-3

GPT-3 launched in 2020. Its scale is staggering:

- **Parameters:** 175 billion (about 100 times more than GPT-2)
- **Training Data:** Hundreds of billions of words from all over the internet and many book sources

With this much knowledge, GPT-3 could do things its predecessors simply couldn’t. It demonstrated **zero-shot learning** (solving new tasks without any examples) and **few-shot learning** (using just a few examples in the prompt).

For example, you could simply prompt:

**Prompt:** Translate to French: "How are you?"

**GPT-3 Output:** "Comment ça va ?"

It can also write code, answer trivia, summarize text, even perform math or logic steps—just by drawing on the patterns it saw during training.

Compared to GPT-2, GPT-3 writes longer, more logical, more accurate text and can keep up context over several paragraphs. Its writing can mimic various tones or roles, not just churn out generic sentences.

## Bigger Isn’t Just Bigger—Why Scale Matters

Making the model larger doesn’t just make it a better parrot. The jump in scale is more like going from a travel phrasebook to true fluency.

Small models are like tourists: they can order coffee, but miss jokes or cultural references. Huge models, with billions of parameters, start to notice idioms, rhythms, and subtle connections in language.

That’s why a big jump in scale, like GPT-3, brings new abilities. The model isn’t learning isolated patterns anymore. It starts to infer general rules about language, tasks, and even reasoning—just by seeing enough examples.

This is why GPT-3 can solve problems and follow instructions it was never specifically trained for.

## Quick Tour: Using GPT-2 and GPT-3 with Code

You can use GPT-2 on your own computer quickly, thanks to the HuggingFace Transformers library. (GPT-3 is accessed via an API, but works similarly.)

First, install the required libraries:

```bash
pip install transformers torch
```

Here’s a minimal Python example to prompt GPT-2 and get its continuation:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

prompt = "The robot entered the room and"
inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(inputs.input_ids, max_length=30, do_sample=True)
completion = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(completion)
```

The `.generate()` call asks the model to keep predicting the next word for up to 30 tokens. You can change the `prompt` or `max_length` to experiment.

For GPT-3, you’d use OpenAI’s API instead of downloading the model. But the process is simple: send a prompt, get a continuation.

These tools let you access language models that have read a significant slice of the world’s text—all packed into models you can use with just a few lines of code.

---

## Key Takeaways

- GPT models generate new text by predicting the next word in a sequence.
- Transformers allow the model to capture wide context in sentences.
- Scaling up model size and training data improves text quality and versatility.
- GPT-3 exhibits abilities like zero-shot and few-shot learning.
- GPT-2 can be used locally with simple Python code; GPT-3 is accessed via API.

## Try It Yourself

Run the provided GPT-2 code example. Change the input prompt to three distinctly different sentences, such as a question, a story opening, and a random sentence fragment. For each prompt, run the code, observe how the model's output differs, and write a short note on the variations you see.

## Further Resources

- 📄 [Language Models are Few‑Shot Learners](https://openai.com/index/language-models-are-few-shot-learners/)
- 📄 [GPT‑3: Language Models are Few‑shot Learners (ArXiv)](https://arxiv.org/abs/2005.14165)
- 🎥 [GPT‑3: Language Models are Few‑Shot Learners — Talk by Aman Singhal](https://amansinghal927.github.io/talks/gpt3-few-shot-learners/)

---

**Coming up on Day 39:** T5 and the Text-to-Text Framework