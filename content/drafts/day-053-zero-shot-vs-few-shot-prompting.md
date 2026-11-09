---
day: 53
generated_at: '2026-11-06T16:25:25.819302+00:00'
phase: Phase 7 — Prompt Engineering
recap_summary: Explained the concepts of zero-shot and few-shot prompting in language
  models, comparing their use cases, strengths, and practical differences for text
  classification tasks.
status: pending_review
title: 'Day 53: Zero-Shot vs. Few-Shot Prompting'
topic_title: Zero-Shot vs. Few-Shot Prompting
---

# Prompting: The Basics

In natural language processing (NLP) and generative AI (GenAI), a **prompt** is any text or instruction you give to a language model—like GPT-4, LLaMA, or Claude—to get it to do a task. Think of prompting as simply telling the model, in plain English (or another supported language), what you want done.

A prompt can be as short as a question—  
`What’s the weather like in Paris?`  
—or as detailed as a set of instructions with rules and constraints. Anything you type before the model responds is part of the prompt.

**Prompt engineering** means designing, refining, and experimenting with prompts to get better or more reliable results from a language model.

---

## Zero-Shot Prompting

**Zero-shot prompting** means giving the language model a direct instruction to do a task, but without providing any examples of how the task should be performed. The model uses your instructions alone to figure out what to do.

For example, if you want to classify the sentiment (positive, negative, or neutral) of a sentence, a zero-shot prompt would be:

```
Classify the sentiment of the following sentence as Positive, Negative, or Neutral.
Sentence: “I’m really looking forward to the weekend.”
```

Here, you haven’t shown any sample answers. The model relies on its training data and general knowledge of language.

**Zero-shot prompting works well when:**
- Your instructions are clear and leave no room for misunderstanding.
- The task is common, so the model has likely seen similar instructions before.
- You don’t have labeled data (examples with correct answers).

---

## Few-Shot Prompting

**Few-shot prompting** is when you add a few examples to your instructions. Each example pairs a task with its desired answer. The model learns from these examples “in context”—it observes the pattern and applies it to your new question.

Continuing with sentiment classification, here’s a few-shot prompt:

```
Classify the sentiment of each sentence as Positive, Negative, or Neutral.

Sentence: “I love this new phone.”
Sentiment: Positive

Sentence: “The food was disappointing.”
Sentiment: Negative

Sentence: “It’s an ordinary Monday.”
Sentiment: Neutral

Sentence: “I’m really looking forward to the weekend.”
Sentiment:
```

By showing examples, you demonstrate the type of answer, the labels you want, and how they relate to each input. When the model reaches the new sentence, it’s expected to keep following the pattern.

**Few-shot prompting is useful when:**
- The task or answer labels are unique to your application.
- You’re asking for unusual instructions or a specific output format.
- Basic instructions might be interpreted in several ways, and you need to be precise.

---

## Zero-Shot vs. Few-Shot: Practical Comparison

Here’s how outputs can differ.

**Zero-shot prompt:**
```
Classify the sentiment of the following sentence as Positive, Negative, or Neutral.
Sentence: “The customer service was adequate, nothing more.”
```
Possible output:  
`Neutral`

**Few-shot prompt:**
```
Classify the sentiment of each sentence as Positive, Negative, or Neutral.

Sentence: “The movie was fantastic!”
Sentiment: Positive

Sentence: “I wish the product lasted longer.”
Sentiment: Negative

Sentence: “The customer service was adequate, nothing more.”
Sentiment:
```
Possible output:  
`Neutral`

For this common task, both zero-shot and few-shot are likely to work. The benefit of few-shot grows for less standard labels or tricky tasks.

A more nuanced example:

**Zero-shot prompt:**
```
Label the following review as Supportive or Critical.
Review: “They did their best, but the outcome wasn’t ideal.”
```
Possible output:  
`Critical`

**Few-shot prompt:**
```
Label each review as Supportive or Critical.

Review: “She provided helpful advice throughout.”
Label: Supportive

Review: “Many mistakes were made by the team.”
Label: Critical

Review: “They did their best, but the outcome wasn’t ideal.”
Label:
```
Possible output:  
`Critical`

Here, your label names (“Supportive”, “Critical”) don’t match common sentiment labels. With a few-shot prompt, the model clearly sees the expected format and interpretation, making its answer more reliable.

**Few-shot prompting is especially reliable when:**
- The task is nonstandard (labels like “Supportive” or “Critical” instead of just “Positive”/“Negative”).
- Output format needs to follow a template, not just a plain answer.
- The model misinterprets your zero-shot prompt.

Zero-shot is fast and works for many standard problems. Few-shot is more robust and flexible when your needs are custom, or if you want consistent output.

---

## Choosing Zero-Shot vs. Few-Shot

**Zero-shot prompting is usually enough when:**
- The task is standard, like summarizing text, answering questions, or general classification.
- You want a quick answer and don’t care about strict formatting.
- You don’t have example data—or don’t want to use it upfront.

**Few-shot prompting is a better fit when:**
- The task uses custom or niche labels/output formats.
- You want more predictable, reliable results—critical for automation.
- Zero-shot prompts aren’t producing what you need.
- Instructions could be interpreted in multiple, conflicting ways.

**Trade-offs:**
- *Complexity:* Few-shot prompts are longer, and picking “good” examples may take some work up front.
- *Reliability:* Few-shot prompts usually give more consistent results, especially for unusual or challenging tasks.
- *Example requirements:* You need labeled examples—real or made up—for few-shot. Zero-shot only needs instructions.

Most people start with zero-shot prompting. If that works, keep it simple. If results aren’t what you want, add one or more examples as a few-shot prompt.

---

## Python Example: Zero-Shot vs. Few-Shot Prompting with OpenAI API

Here’s a minimal script using OpenAI’s GPT API (works with GPT-3.5, GPT-4, or compatible local models). It compares zero-shot and few-shot prompts for sentiment classification.

```python
import openai

openai.api_key = "your-api-key"

def classify_sentiment(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Or "gpt-4", or your compatible local model
        messages=[{"role": "user", "content": prompt}],
        max_tokens=10,
        n=1,
        stop=None,
        temperature=0  # Low temperature for predictable output
    )
    return response.choices[0].message['content'].strip()

# Zero-shot prompt
zero_shot_prompt = (
    "Classify the sentiment of the following sentence as Positive, Negative, or Neutral. "
    'Sentence: "I’m really looking forward to the weekend."'
)

# Few-shot prompt
few_shot_prompt = (
    "Classify the sentiment of each sentence as Positive, Negative, or Neutral.\n"
    'Sentence: "I love this new phone."\n'
    "Sentiment: Positive\n\n"
    'Sentence: "The food was disappointing."\n'
    "Sentiment: Negative\n\n"
    'Sentence: "It’s an ordinary Monday."\n'
    "Sentiment: Neutral\n\n"
    'Sentence: "I’m really looking forward to the weekend."\n'
    "Sentiment:"
)

print("Zero-shot result:", classify_sentiment(zero_shot_prompt))
print("Few-shot result:", classify_sentiment(few_shot_prompt))
```

This code sends both a zero-shot and a few-shot prompt to the model. For each one, the model is asked to classify the same sentence’s sentiment. You can compare the results and adjust prompts as needed.

Replace `"your-api-key"` with your actual OpenAI API key before running.

---

This approach—starting with zero-shot, and moving to few-shot if needed—is the foundation for more advanced prompt engineering. Nearly all real-world uses start here, and build up as needs and requirements become clearer.

---

## Key Takeaways

- A prompt is the instruction or input provided to a language model.
- Zero-shot prompting gives instructions without examples; few-shot prompting includes labeled examples.
- Few-shot prompting improves reliability for uncommon tasks or custom output formats.
- Zero-shot prompting works well for standard, common tasks without special requirements.
- Choosing the right prompting strategy depends on task clarity, complexity, and desired output consistency.

## Try It Yourself

Create two prompts for a simple text classification task, such as deciding if a review is positive or negative. Write one using zero-shot prompting (just instructions), and another using few-shot prompting (include a few labeled examples). Run both prompts through a language model of your choice and compare which one gives more reliable or useful results.

## Further Resources

- 🎥 [Mastering Prompt Engineering: Zero‑Shot & Few‑Shot Prompting | Prompt Engineering : Part 2](https://www.youtube.com/watch?v=M_ZXSNaYJB0)
- 📘 [Prompt engineering | OpenAI API prompt‑engineering guide](https://developers.openai.com/api/docs/guides/prompt-engineering)
- 📘 [Prompt engineering techniques - Microsoft Foundry (Azure) documentation](https://learn.microsoft.com/en-us/azure/foundry/openai/concepts/prompt-engineering)
- 📄 [Zero‑Shot vs. Few‑Shot Prompting tutorial — LING3401 (CUHK) tutorial PDF](https://yigechen.com/files/sp25/ling3401_sp25_w15_tut_s_p.pdf)
- 📄 [Zero‑Shot and Few‑Shot Prompting — SuperML.org Prompt Engineering Fundamentals](https://superml.org/courses/prompt-engineering-fundamentals/zero-shot-few-shot-prompting)

---

**Coming up on Day 54:** Chain-of-Thought Prompting