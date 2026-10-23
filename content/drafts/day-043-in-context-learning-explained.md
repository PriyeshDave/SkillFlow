---
day: 43
generated_at: '2026-10-23T13:54:15.740187+00:00'
phase: Phase 6 — The Rise of LLMs
recap_summary: Explained in-context learning, where language models adapt to new tasks
  by imitating examples in the input prompt rather than updating their internal parameters,
  and contrasted this approach with traditional fine-tuning.
status: pending_review
title: 'Day 43: In-Context Learning Explained'
topic_title: In-Context Learning Explained
---

## What Is In-Context Learning?

In-context learning is a way to teach a language model a new task by providing examples directly in the input prompt. You do not change anything inside the model. You just show it the pattern you want to see.

Traditional machine learning works differently. If you want a model to perform a new task, you usually retrain it or fine-tune it with new data. This changes the model’s parameters. Parameters are the internal settings the model learned from previous data.

In in-context learning, the model’s parameters stay frozen. The model instead looks at the examples in your prompt and figures out the desired behavior for that specific input. This is like demonstrating a task to someone right at the moment you want them to perform it—no retraining, just showing.

This is possible because large language models (LLMs), such as GPT-3 or Llama 2, are trained on huge amounts of varied text. During training, they see countless patterns, tasks, and instructions. They become good at recognizing and imitating new instructions presented in context.

## Why In-Context Learning Matters

In-context learning makes LLMs flexible and practical. You don’t need to retrain or fine-tune a large model—both time-consuming and costly—for each new use case. Instead, anyone can adapt the same model to a new task by crafting a clear prompt with examples.

This approach is valuable when:
- You need the model to do something it wasn’t trained on directly.
- The task changes often.
- You don't have enough data to properly train a model from scratch.

It’s also accessible. If you can write concrete instructions and show a few examples, you can customize the model’s behavior, without any special software or machine learning background.

## How LLMs Learn from Context

Imagine you want to teach a friend a new card game. You give them a playing sheet showing three example rounds. They see: "If you get this card, play that card." The friend then tries to mimic what you showed, based on your examples.

LLMs do something similar with text. When you prompt an LLM like this:

```
Country: France
Capital: Paris

Country: Japan
Capital: Tokyo

Country: Egypt
Capital: 
```

The model reads the examples in order. It notices the pattern: country names are matched with capital cities. It predicts the answer for Egypt by following the same pattern.

No internal changes happen in the model. It does not update its parameters. It just runs through the task using what it sees in the prompt.

## Concrete Example: Few-Shot Prompting

Let’s see a real code example using the OpenAI API. This will demonstrate "few-shot prompting," where you show a few examples and ask the model to follow them.

Suppose you want to find the capital of a country. Here’s a minimal example with two demonstrations in the prompt, then a new question:

```python
import openai

# (Replace YOUR_API_KEY with your actual key)
openai.api_key = 'YOUR_API_KEY'

prompt = """\
Country: Italy
Capital: Rome

Country: Canada
Capital: Ottawa

Country: Brazil
Capital:"""

response = openai.Completion.create(
    engine="text-davinci-003",
    prompt=prompt,
    max_tokens=5,
    temperature=0  # Set to 0 for consistent output
)

print(response['choices'][0]['text'].strip())
```

Here’s what’s happening:
- Two example pairs: Italy–Rome and Canada–Ottawa.
- For Brazil, the answer is left blank.
- The model matches the input (Brazil) and fills in “Brasília,” following the established pattern.

All the "learning" happens in the prompt. If you want to change the task—country codes, synonyms, translations—just swap out the examples.

## Limitations and Common Confusions

In-context learning has clear boundaries.

First, the model isn’t truly learning in a permanent way. Humans form lasting memories. Here, the model is only doing one-time imitation. Remove the examples, and the model forgets. Each new prompt needs new demonstrations.

Second, success depends on your examples. If the task is tricky, or if you need many examples, your prompt might become too large. LLMs have maximum input lengths—usually thousands of tokens, not unlimited.

People often confuse in-context learning with fine-tuning. Fine-tuning means feeding many examples to the model and updating its parameters, making permanent changes. In-context learning only uses the input for that one session. No learning persists after the prompt.

LLMs are surprisingly strong at picking up new tasks with just a few examples. But the system is not perfect. If the pattern is subtle, messy, or totally unfamiliar, the model can fail or guess.

To summarize: in-context learning means teaching the model what you need it to do, on the fly, using prompt examples. It’s a powerful, flexible tool for rapid, temporary customization—but it can’t replace retraining for complicated, high-stakes, or nuanced tasks.

---

## Key Takeaways

- In-context learning adapts models to new tasks using prompt examples, not parameter changes.
- No retraining or fine-tuning is needed—behavior is guided by the input prompt alone.
- Each new task demonstration is temporary; the model does not retain new abilities across prompts.
- Prompt size limitations and example clarity affect in-context learning success.
- In-context learning is flexible but not a substitute for full model retraining in complex cases.

## Try It Yourself

Create a prompt that demonstrates how to convert Celsius temperatures to Fahrenheit. Include two example pairs (such as 'Celsius: 0; Fahrenheit: 32' and 'Celsius: 100; Fahrenheit: 212'), then leave a third Celsius value (for example, 25) for the model to answer. Enter this prompt into a language model and see if it returns the correct Fahrenheit value.

## Further Resources

- 🎥 [Emergence of In-Context Learning in Small Transformer Models (YouTube talk)](https://www.classcentral.com/course/youtube-emergence-of-in-context-learning-in-small-transformer-models-by-gautam-reddy-482946)
- 📄 [An Explanation of In‑context Learning as Implicit Bayesian Inference (Xie et al., 2021)](https://arxiv.org/abs/2111.02080)
- 📄 [What Do Language Models Learn in Context? The Structured Task Hypothesis (ACL 2024)](https://aclanthology.org/2024.acl-long.669/)
- 📄 [Rethinking the Role of Demonstrations: What Makes In‑Context Learning Work? (EMNLP 2022)](https://aclanthology.org/2022.emnlp-main.759/)

---

**Coming up on Day 44:** Zero-Shot, One-Shot, and Few-Shot Learning