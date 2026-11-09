---
day: 54
generated_at: '2026-11-09T19:17:23.287806+00:00'
phase: Phase 7 — Prompt Engineering
recap_summary: Explained the concept of Chain-of-Thought (CoT) prompting, highlighting
  how it encourages language models to show step-by-step reasoning for improved accuracy
  and transparency, and compared it to direct-answer prompting with practical examples.
status: pending_review
title: 'Day 54: Chain-of-Thought Prompting'
topic_title: Chain-of-Thought Prompting
---

## What is Chain-of-Thought Prompting?

Chain-of-Thought (CoT) prompting is a way to help language models solve problems by working through each reasoning step, rather than jumping straight to an answer.

Normally, when you ask a question, the model just gives a final answer. With CoT prompting, you write your prompt so the model explains its thought process as it solves the problem. The model "thinks out loud," showing the steps it takes before reaching a conclusion.

This makes its reasoning transparent. For questions that require multiple logical steps, it often leads to more accurate answers.


## Why Use Chain-of-Thought Prompting?

Large language models (LLMs), like GPT-4, are powerful. But they're not perfect. If you only ask for an answer, the model may skip important steps, make leaps in logic, or ignore hidden assumptions.

Chain-of-Thought prompting slows the model down, encouraging it to consider each step. This helps the model:
- Avoid mistakes that show up only in the intermediate steps,
- Handle multi-part questions or questions that require connecting several pieces of information,
- Show its work, so you can spot where it went wrong if it makes a mistake.

This technique is especially useful for math problems, logic puzzles, code analysis, and decision-making tasks with multiple steps.


## How Chain-of-Thought Prompting Works

There are two main ways to prompt an LLM:

**1. Direct-Answer Prompting**

This is the default style. You ask only for the result, with no explanation.

Example:
```
Q: If a train leaves New York at 3 PM, travels for 4 hours, and arrives in Boston, what time does it arrive?
A:
```
Model's answer:
```
7 PM.
```
You just get the outcome.

**2. Chain-of-Thought Prompting**

Here, you explicitly ask the model to explain its process.

Example:
```
Q: If a train leaves New York at 3 PM, travels for 4 hours, and arrives in Boston, what time does it arrive?
Let's think step by step.
A:
```

Model's answer:
```
The train leaves New York at 3 PM. It travels for 4 hours. Adding 4 hours to 3 PM brings us to 7 PM. The train arrives in Boston at 7 PM.
```

Asking for step-by-step reasoning—using phrases like "Let's think step by step," "Explain your reasoning," or "Show your work"—nudges the model to make its thought process visible.


## Examples of Chain-of-Thought Prompts

Let's see how CoT changes the model's responses in real examples.

### Example 1: Simple Math

**Direct Prompt:**
```
Q: If a jar has 4 red balls and 7 blue balls, how many balls are there in total?
A:
```
Model's answer:
```
11
```

**Chain-of-Thought Prompt:**
```
Q: If a jar has 4 red balls and 7 blue balls, how many balls are there in total? Let's think step by step.
A:
```
Model's answer:
```
There are 4 red balls and 7 blue balls in the jar. 4 plus 7 equals 11. So, there are 11 balls in total.
```

For easy questions, both prompt types work. In the CoT version, you see each part of the problem referenced. If there were a mistake, it would be easier to find.

### Example 2: Multi-Step Logical Reasoning

**Direct Prompt:**
```
Q: Tom is three years older than Sam. Sam is two years older than Lily. If Lily is 10 years old, how old is Tom?
A:
```
Model's answer:
```
15
```
You see just a number with no explanation.

**Chain-of-Thought Prompt:**
```
Q: Tom is three years older than Sam. Sam is two years older than Lily. If Lily is 10 years old, how old is Tom? Let's think step by step.
A:
```
Model's answer:
```
Lily is 10 years old. Sam is two years older than Lily, so Sam is 12 years old. Tom is three years older than Sam, so Tom is 15 years old.
```

Now each logical step is spelled out. If an error occurred, you could trace it to the faulty step.

### Annotated Comparison Table

| Prompt Style        | Output                                 | Transparency      | Error Detection |
|---------------------|----------------------------------------|-------------------|----------------|
| Direct-Answer       | “15”                                   | None              | Hard           |
| Chain-of-Thought    | "Lily is 10... Tom is 12... Tom is 15."| Every step shown  | Easy           |

## When to Use (and Not Use) Chain-of-Thought Prompting

**Use CoT prompting when:**
- The problem needs several reasoning steps.
- The answer depends on intermediate facts or calculations.
- You are debugging a prompt and need to see where the model may go wrong.
- You need to know how the answer was generated (interpretability is important).

**CoT prompting is less helpful when:**
- The question is very simple or doesn't require reasoning (for example, “What color is the sky?”).
- Short responses are needed—for speed or output length limits.
- Transparency is not important and only the answer matters.

If you want to see the "work shown," use CoT. For quick, simple answers without explanation, direct prompting is enough.


## Minimal Python Example: Comparing CoT and Direct Prompting

Here’s a Python example that compares both methods with the same question. It sends one direct prompt and one CoT prompt to the language model, then prints both responses.

Replace `"YOUR_API_KEY"` with your OpenAI API key (or another LLM provider as appropriate).

```python
import openai

openai.api_key = "YOUR_API_KEY"

question = "Tom is three years older than Sam. Sam is two years older than Lily. If Lily is 10 years old, how old is Tom?"

direct_prompt = f"Q: {question}\nA:"
cot_prompt = f"Q: {question} Let's think step by step.\nA:"

def ask(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
        temperature=0,
    )
    return response['choices'][0]['message']['content'].strip()

print("Direct Prompt Output:")
print(ask(direct_prompt))
print("\nChain-of-Thought Prompt Output:")
print(ask(cot_prompt))
```

Run this code to see how the two prompting approaches change the model’s response. The CoT answer typically shows all the reasoning steps. The direct answer will usually just be a number.

Chain-of-Thought prompting is a practical tool for getting better, more interpretable answers from language models. When step-by-step reasoning matters, it’s a technique worth reaching for.

---

## Key Takeaways

- Chain-of-Thought prompting makes language models show their reasoning process step by step.
- CoT prompts improve transparency and help identify errors in multi-step problems.
- Direct-answer prompting only yields final answers without explanations.
- CoT is most useful for complex, multi-part questions or debugging model logic.
- Short, simple questions often don't benefit from CoT prompting.

## Try It Yourself

Choose a multi-step reasoning problem, such as a word math problem or logic puzzle. Write both a direct prompt and a Chain-of-Thought prompt for the same question, then run them through a public language model interface (like OpenAI Playground). Compare the responses and note how the explanations differ between the two methods.

## Further Resources

- 🎥 [Chain‑of‑Thought Prompting Explained in English | Zero‑Shot, Few‑Shot & AI Reasoning Models](https://www.youtube.com/watch?v=HLQqPwOag3U)
- 📄 [Chain‑of‑Thought Prompting Elicits Reasoning in Large Language Models (arXiv)](https://arxiv.org/abs/2201.11903)
- 📄 [Prompt Engineering in Practice — Chapter 9: Advanced prompt patterns (Manning preview)](https://www.manning.com/preview/prompt-engineering-in-practice/chapter-9)
- 📘 [Chain‑of‑Thought (CoT) prompting — CMU LLM Documentation Guide: Prompting Techniques](https://guides.library.cmu.edu/LLMDocumentationGuide/Techniques)
- 📄 [What Is Chain of Thought (CoT) Prompting? | NVIDIA Glossary](https://www.nvidia.com/en-us/glossary/cot-prompting/)

---

**Coming up on Day 55:** Self-Consistency and Tree-of-Thought Prompting