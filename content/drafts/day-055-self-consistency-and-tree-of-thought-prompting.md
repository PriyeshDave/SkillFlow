---
day: 55
generated_at: '2026-11-10T16:39:31.839794+00:00'
phase: Phase 7 — Prompt Engineering
recap_summary: Explained prompt engineering for large language models, focusing on
  techniques to improve reliability and reasoning such as single prompts, self-consistency,
  and Tree-of-Thought prompting.
status: pending_review
title: 'Day 55: Self-Consistency and Tree-of-Thought Prompting'
topic_title: Self-Consistency and Tree-of-Thought Prompting
---

### What Is Prompt Engineering?

Prompt engineering means designing the input, called a *prompt*, given to a large language model (LLM) to guide its output toward useful, accurate, or creative results.

A *prompt* is the text or question you type in for the model. For example:  
“What is the capital of France?”  
That sentence is your prompt.

An LLM (large language model) is a type of neural network that has been trained on a large collection of text from many sources. Most use a transformer architecture. When you give it a prompt, it predicts and generates the next words, one at a time.

Prompt engineering shapes how the model responds. You do this by carefully choosing your words, the structure, and even the layout of your prompt. Sometimes, adding context or detailed instructions can get much better results.

### Why Simple Prompts Are Sometimes Not Enough

Simple prompts work for straightforward questions.  
Example:
```
Q: What is 2+2?
A: 4
```
But not all tasks are this easy. Take this:
```
Q: Alice has 3 apples. She buys 2 more, then gives 1 to Bob. How many does she have left?
```
A model might answer “5”, “4”, or “3”, depending on tiny wording changes or even random chance. It may skip steps or get confused when the question is more complicated.

LLMs are probabilistic—this means they create answers by picking likely next words, not calculating absolute facts. That leads to inconsistencies, especially with reasoning problems that require showing each step.

### Introducing Self-Consistency

One solution is *self-consistency*. This uses the model’s randomness to our advantage.

1. Ask the same question several times, letting the model produce slightly different answers.
2. Collect all the responses.
3. Pick the answer that comes up most frequently.

This helps for questions with a single clear answer where the LLM’s first try isn’t always reliable. Think of it as polling a group of experts. The most common answer is likely to be correct, even if some individual answers are off.

### Tree-of-Thought (ToT) Prompting: Making Reasoning Explicit

Some problems trip up LLMs no matter how many times we ask. The issue isn’t randomness, but a missed step—or the model “taking a shortcut” that skips logic.

*Tree-of-Thought (ToT) prompting* solves this by telling the model to walk through the process step by step and consider alternatives.

**Analogy:** Imagine solving a tricky puzzle on paper. You jot down each part (“First, I buy 2 apples. Then I give one away…”) and sometimes even track what happens if you tried things a different way. Each decision can branch out, forming a “tree” of thought.

ToT prompting asks the model to do the same: “Show your reasoning for each step.” This exposes errors, makes thinking visible, and allows partial solutions to be checked.

### Comparing Approaches: Single, Self-Consistent, and ToT Prompts

Let’s compare all three with a simple word problem:

**Task:**  
*"Sarah has 5 candies. She buys 3 more. She gives 2 to Tom. How many does Sarah have now?"*

#### 1. Single Prompt

We just ask the question:

```python
import openai

def single_prompt(question):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=question,
        max_tokens=20,
        temperature=0  # Deterministic: same answer every time
    )
    return response.choices[0].text.strip()

question = "Sarah has 5 candies. She buys 3 more. She gives 2 to Tom. How many does Sarah have now?"
print(single_prompt(question))
```
Often correct for easy problems. Less reliable for tricky ones.

#### 2. Self-Consistency

Ask several times, with a bit of randomness, then take the most common answer:

```python
from collections import Counter

def self_consistent_prompt(question, num_samples=5):
    answers = []
    for _ in range(num_samples):
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=question,
            max_tokens=20,
            temperature=0.8,  # Adds variety to output
            n=1
        )
        answer = response.choices[0].text.strip()
        answers.append(answer)
    most_common = Counter(answers).most_common(1)[0][0]
    return most_common

print(self_consistent_prompt(question))
```
This method helps when the LLM's answers are inconsistent due to randomness.

#### 3. Tree-of-Thought Prompt

Make the model show its work:

```python
def tree_of_thought_prompt(question):
    prompt = f"""Let's solve the problem step by step.

Problem: {question}

Step 1: How many candies did Sarah start with?
Step 2: How many candies did she have after buying more?
Step 3: How many did she have after giving some away?
Answer:"""
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=100,
        temperature=0
    )
    return response.choices[0].text.strip()

print(tree_of_thought_prompt(question))
```
This nudge makes the model much more reliable for multi-step reasoning.

#### Method Comparison Table

| Approach          | How It Works                  | Strengths                         | Drawbacks                               |
|-------------------|------------------------------|-----------------------------------|-----------------------------------------|
| Single Prompt     | Ask once, get one answer      | Fast, simple, inexpensive         | Inconsistent or incomplete results      |
| Self-Consistency  | Ask multiple, pick consensus | Smoother, more reliable for basics| Needs more calls, no extra reasoning    |
| Tree-of-Thought   | Ask for step-by-step solution| Clear logic, handles hard tasks   | More tokens, slower, needs structure    |

### Choosing the Right Tool: When and Why

Use **self-consistency** for tasks that:
- Have a clear, single answer
- Are straightforward, but the model is shaky on first try
- Examples: fact recall, arithmetic, classification

Use **Tree-of-Thought prompting** when:
- The task needs multi-step or logical thinking
- You want to see the model’s reasoning (for debugging, explanations, or trust)
- Examples: math word problems, puzzles, structured analysis

**Things to watch out for:**  
Both techniques cost more (in time and money) than simple prompts. Self-consistency means running the model several times. Tree-of-Thought prompts produce longer answers, using more tokens.

For easy tasks, “single prompt” is fastest. As the difficulty or importance increases, or when you need rock-solid reliability or explainability, move up to self-consistency or ToT prompting.

You’ll get better at picking the right approach through experimentation. Start simple, then add these tools when you see a single prompt is no longer enough.

---

## Key Takeaways

- Prompt engineering shapes how language models respond to inputs.
- Simple prompts often work for basic queries but may fail on complex tasks.
- Self-consistency improves answer reliability by synthesizing multiple model outputs.
- Tree-of-Thought prompting guides models to show step-by-step reasoning.
- Choosing the right prompting method depends on task complexity and need for explainability.

## Try It Yourself

Write your own Tree-of-Thought style prompt for a non-math problem, like planning a weekend trip or solving a riddle. Test both a single prompt and your ToT prompt with an LLM, then compare the outputs. Observe how the reasoning and detail differ between the two approaches.

## Further Resources

- 🎥 [Tree of Thought, ReAct & Self‑Consistency – Advanced Prompting Masterclass | Sukrid LearnHub](https://www.youtube.com/watch?v=2AdoXG8EdfA)
- 📄 [Self‑Consistency Improves Chain of Thought Reasoning in Language Models (Wang et al. 2022)](https://arxiv.org/abs/2203.11171)
- 📄 [Tree of Thoughts: Deliberate Problem Solving with Large Language Models (Yao et al. 2023)](https://arxiv.org/abs/2305.10601)
- 📘 [Techniques to Improve Reliability — OpenAI Cookbook (Self‑Consistency)](https://github.com/openai/openai-cookbook/blob/main/articles/techniques_to_improve_reliability.md)
- 📄 [Self‑Consistency, Tree of Thoughts, and ReAct — WriterDock blog post](https://writerdock.in/learning/prompt-engineering/self-consistency-tree-of-thoughts-and-react)

---

**Coming up on Day 56:** ReAct Prompting: Reasoning + Acting