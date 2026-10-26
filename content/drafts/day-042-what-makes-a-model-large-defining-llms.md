---
day: 42
generated_at: '2026-10-22T14:10:41.580807+00:00'
phase: Phase 6 — The Rise of LLMs
recap_summary: Explained what determines a machine learning model's size—specifically
  the number of learnable parameters—and how this impacts capacity, complexity, resource
  needs, and real-world definitions of 'large' in language models.
status: pending_review
title: 'Day 42: What Makes a Model "Large"? Defining LLMs'
topic_title: What Makes a Model "Large"? Defining LLMs
---

## Why Models Have 'Size' at All

A machine learning model’s “size” is its capacity to store learned information. This is measured by the number of **parameters** in the model.

A parameter is a value the model learns from data, like a dial it adjusts to fit patterns. In a basic neural network, these are called weights and biases. Training updates these values to make predictions more accurate. The more parameters, the more precisely a model can represent relationships in the data.

Think of each parameter as an adjustable knob. A model with just a few knobs can represent only simple behaviors or shapes. A model with millions—or billions—of knobs can represent much more complexity, such as human language.

Model size is not about the amount of data a model processes, or the number of words it knows. It is about how many individual values it can fine-tune to transform inputs into outputs.

Why does size matter? Bigger models can represent more details, capture more dependencies, and make richer predictions. But they also demand more computing power, memory, and training time.

## What Counts as “Large”?

The definition of “large” shifts over time as hardware and techniques improve.

A **small model** in natural language tasks might have tens of thousands, or a few million, parameters. Early neural networks for things like sentiment analysis or translation lived in this range.

**Large models** have hundreds of millions or billions of parameters. For context:

- OpenAI’s GPT-1 (2018): 117 million parameters
- Google’s BERT base: 110 million; BERT large: 340 million
- GPT-2: 1.5 billion parameters
- GPT-3: 175 billion parameters

A rough breakdown:

- Small: Under 100 million parameters
- Medium: 100 million to 1 billion
- Large: Over 1 billion
- “LLMs” (Large Language Models): Usually several billion to hundreds of billions of parameters

These boundaries keep rising as new models push what’s possible.

## The Anatomy of a Large Language Model (LLM)

An LLM is a neural network designed for language. Its size—how many parameters it has—comes from a few main places:

**1. Vocabulary Embeddings**  
Each word or token in the vocabulary is assigned a vector—a bundle of numbers, known as an embedding. Every number in every embedding is a parameter. More tokens or higher-dimensional embeddings mean more parameters.

**2. Network Layers**  
LLMs use stacks of layers. Each layer contains weights and biases—parameters the model adjusts as it learns. Adding more layers or making each layer wider increases parameter count.

**3. Output Head**  
To generate a prediction (such as the next word), a final set of parameters converts the model’s output into the chosen token.

Note: **Training data size** is related but different. While more data helps models make sense of all those parameters, it is not itself a parameter. LLMs are trained on vast corpora—often trillions of words—so the many parameters don’t just memorize, but generalize.

Change the vocabulary size, the number and width of layers, or the model architecture, and the parameter count—and model size—changes too.

## More Parameters, More Power?

Increasing parameters enables a model to learn more detailed and nuanced patterns.

A model with few parameters only picks up basic, broad features. As the parameter count grows, the model can represent more subtle relationships and dependencies in the data. LLMs with billions of parameters can track complex language details, like how “bank” differs in “river bank” versus “bank account.”

But there’s a point of **diminishing returns**. Doubling the number of parameters does not double the model’s skill. If a model is too big for the amount of training data, it may just memorize rather than generalize—this is called overfitting. Large models also need far more computing resources and energy.

Past a certain scale, making a model ten times larger may make it ten times harder to run, but only give a slight improvement for most tasks.

Still, larger models often show “emergent” abilities that simply don’t appear in smaller ones, such as in-context learning (solving tasks based on instructions, not pure memorization) or writing code.

## LLM Examples in the Wild

Recent large models include:

- **GPT-2** (OpenAI, 2019): 1.5 billion parameters
- **GPT-3** (OpenAI, 2020): 175 billion parameters
- **BERT (large)** (Google, 2018): 340 million parameters
- **PaLM 2** (Google, 2023): Hundreds of billions of parameters (varies by version)
- **Llama 2** (Meta, 2023): 7 billion to 70 billion parameters
- **Falcon 180B** (TII, 2023): 180 billion parameters

For contrast: a basic spam filter may use just 100,000 to a few million parameters.

## Code Example: Model Size as Parameter Count

Here’s a minimal example in Python to show how model “size” relates to parameter count. Each number in the list represents a parameter. We can then measure the approximate memory used by small and large models.

```python
import sys

# Tiny model: 10,000 parameters
tiny_model = [0.5] * 10_000

# "Large" model: 10,000,000 parameters
large_model = [0.5] * 10_000_000

def memory_mb(obj):
    return sys.getsizeof(obj) / (1024 * 1024)

print(f"Tiny model size: {memory_mb(tiny_model):.2f} MB")
print(f"Large model size: {memory_mb(large_model):.2f} MB")
```

This script prints the memory size of each model, based solely on how many parameters (numbers) it holds. In real neural networks, parameters are organized with complex connectivity, but the principle is the same: more parameters mean a larger, heavier model.

The key is not how much data was seen during training, or how many example sentences the model knows. It is the count of learnable values the model contains. That is what defines a model as “large.”

---

## Key Takeaways

- Model size is defined by the number of learnable parameters.
- Parameters include vocabulary embeddings, network layers, and output heads.
- Larger models capture more complex patterns but require more computation.
- Parameter counts are orders of magnitude higher in LLMs versus traditional models.
- Increasing parameters has diminishing returns beyond a certain scale.

## Try It Yourself

Find the parameter count for a well-known large language model (e.g., GPT-3 or BERT) and for a classic machine learning model (like logistic regression or a small decision tree). Write down both numbers and briefly note how many times larger the language model is compared to the traditional model.

## Further Resources

- 📘 [What is a large language model (LLM)? Complete guide | Elastic](https://www.elastic.co/what-is/large-language-models)
- 📘 [LLM Fundamentals | Microsoft Learn](https://learn.microsoft.com/en-us/agent-framework/journey/llm-fundamentals)
- 🎥 [Large Language Models: From Next-Token Prediction to Reasoning Agents](https://nova-lincs.di.fct.unl.pt/seminars/large-language-models-from-next-token-prediction-to-reasoning-agents/)

---

**Coming up on Day 43:** In-Context Learning Explained