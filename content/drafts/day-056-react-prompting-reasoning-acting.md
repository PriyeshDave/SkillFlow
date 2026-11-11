---
day: 56
generated_at: '2026-11-11T16:43:58.987399+00:00'
phase: Phase 7 — Prompt Engineering
recap_summary: Explained the ReAct prompting technique, which alternates structured
  reasoning and tool-based actions to guide language models through multi-step tasks
  transparently and reliably.
status: pending_review
title: 'Day 56: ReAct Prompting: Reasoning + Acting'
topic_title: 'ReAct Prompting: Reasoning + Acting'
---

### What is ReAct Prompting?

ReAct prompting is a prompt engineering technique that tells a language model to switch between two steps: **reasoning** and **acting**.

**Reasoning** means the model explains its thought process out loud. It lays out its logic, breaks down a problem, or says what it *thinks* needs to happen next.

**Acting** means the model does something in the outside world. Usually, this means calling a tool—like a calculator, a web search, or an external API—to get information it can't just invent.

ReAct prompting structures the model’s work into a repeating sequence:
1. **Reason:** Think about the situation.
2. **Act:** Take an action, like using a tool.
3. **Repeat:** Use the action’s result to reason further. Then, act again if needed.

The name comes from combining **Re**asoning and **Act**ion.

---

### Why Combine Reasoning with Acting?

Picture yourself as a detective working on a mystery, with a smartphone at your side:

- You *talk through* the clues (reasoning).
- You *look up* facts or *do a calculation* on your phone (acting).
- You use what you find to update your thinking.
- You repeat, alternating between thinking and doing.

If you only reasoned and never used your phone, you'd get stuck on facts you can't remember. If you only used your phone without thinking, you'd waste time and miss connections.

ReAct prompting gives language models this same back-and-forth approach. By clearly marking when to reason and when to act, we help the model organize its steps like a capable worker: think, do, think again.

This makes models better at multi-step tasks and tool use. It also helps reduce mistakes by keeping every step visible.

---

### The Structure of a ReAct Prompt

A typical ReAct prompt alternates between two moves:
- **Thought:** The model says what it’s thinking.
- **Action:** The model calls a tool, shown as a clear command.

Thought and Action happen in pairs. Each *Thought* justifies the *Action* that follows. Each *Action* returns a result, which is used for the next *Thought*.

Here's a fictional ReAct exchange for a trivia question using a search tool:

```
Question: What is the height of Mount Everest in meters plus the length of the Golden Gate Bridge in meters?

Thought: I need the height of Mount Everest.
Action: Search("height of Mount Everest in meters")
Observation: The height is 8,848 meters.

Thought: Now I need the length of the Golden Gate Bridge.
Action: Search("length of the Golden Gate Bridge in meters")
Observation: The length is 2,737 meters.

Thought: I can add the two numbers.
Action: Calculate(8848 + 2737)
Observation: The sum is 11,585.

Thought: The answer is 11,585 meters.
```

Key details:
- **Each Thought** is a clear step in planning.
- **Each Action** is explicit and can be picked up by another system.
- The model doesn’t skip steps or act without a reason.

---

### Concrete Example: Using a Calculator Tool

Let’s see a simple example where the model uses a calculator tool to solve a math problem.

Suppose the question is: *“What is 7 times 8, plus 12?”*

A language model using ReAct prompting will reason step by step, using the calculator tool at each “Action.”

Here's the sequence:

```
Question: What is 7 times 8, plus 12?

Thought: First, I need to calculate 7 times 8.
Action: Calculate(7 * 8)
Observation: The result is 56.

Thought: Next, I need to add 12 to 56.
Action: Calculate(56 + 12)
Observation: The result is 68.

Thought: The final answer is 68.
```

- **Observation** shows what came back from the calculator.
- Every step is visible and recoverable.

#### Minimal Python Example

Let’s wire up a minimal version in Python. This simulates what an LLM agent framework might do: read lines from the prompt, call the calculator when it sees an "Action:", and print results.

This is not production code—just a demo:

```python
def calculator(expr):
    # Simple calculator to evaluate math expressions
    return eval(expr)

prompt_lines = [
    "Question: What is 7 times 8, plus 12?",
    "Thought: First, I need to calculate 7 times 8.",
    "Action: Calculate(7 * 8)",
    "Observation: The result is 56.",
    "Thought: Next, I need to add 12 to 56.",
    "Action: Calculate(56 + 12)",
    "Observation: The result is 68.",
    "Thought: The final answer is 68."
]

for line in prompt_lines:
    print(line)
    if line.startswith("Action: Calculate("):
        expr = line[len("Action: Calculate("):-1]
        result = calculator(expr)
        print(f"Observation: The result is {result}.")
```

Output:

```
Question: What is 7 times 8, plus 12?
Thought: First, I need to calculate 7 times 8.
Action: Calculate(7 * 8)
Observation: The result is 56.
Thought: Next, I need to add 12 to 56.
Action: Calculate(56 + 12)
Observation: The result is 68.
Thought: The final answer is 68.
```

A real LLM agent system (like LangChain or OpenAI Functions) can generate *Thought* and *Action* lines as it works. The core pattern—alternating reasoning and tool use—remains the same.

---

### How To Write Your Own ReAct Prompt

To design a ReAct prompt:

1. **Start with a clear question.**
2. **Guide the model to alternate:** Use “Thought: ...” (reasoning), then “Action: ...” (tool use).
3. **Explain what tools are available, and the format for each Action.**
4. **Have the model note each tool result (“Observation: ...”), and continue.**
5. **Finish with a summary answer (“Thought: ...”).**

For example, suppose you want a model to use a temperature converter:

```
Question: If it is 68 degrees Fahrenheit, what is that in Celsius, plus 5?

Thought: First, I need to convert 68 F to Celsius.
Action: ConvertFtoC(68)
Observation: The result is 20.0.

Thought: Next, I need to add 5 to 20.0.
Action: Calculate(20.0 + 5)
Observation: The result is 25.0.

Thought: The answer is 25.0 degrees Celsius.
```

Tips:
- Keep Thoughts short—just one reasoning step at a time.
- Each Action should map clearly to an available tool.
- Each Observation should follow immediately after the related Action.
- Repeat until you reach an answer.

The ReAct structure keeps everything transparent, step by step. It helps language models reliably use tools and outside data, acting like a well-organized human solving a problem.

---

## Key Takeaways

- ReAct prompting combines explicit reasoning and tool use in clear, alternating steps.
- Each 'Thought' line reveals the model's current reasoning.
- Each 'Action' line triggers a specific tool or function, followed by an 'Observation' of the result.
- This approach keeps the problem-solving process transparent and stepwise.
- ReAct prompts are especially useful for multi-step tasks requiring external information or calculations.

## Try It Yourself

Choose a real-world question that requires an external fact (like today's weather or a currency conversion). Write a ReAct-style prompt for it, alternating 'Thought:' and 'Action:' lines and adding 'Observation:' with imagined tool results, simulating how an LLM agent would reason and act through the task step by step.

## Further Resources

- 📄 [ReAct: Synergizing Reasoning and Acting in Language Models (original paper)](https://arxiv.org/abs/2210.03629)
- 📄 [ReAct — Complete Guide | Prompt Engineering Tutorial](https://www.toolliyo.com/tutorials/prompt-engineering-tutorial-react-prompting-complete-guide)
- 📄 [ReAct in Prompt Engineering in Practice (liveBook)](https://livebook.manning.com/learn/concepts/react--technique/prompt-engineering-in-practice)

---

**Coming up on Day 57:** System Prompts vs. User Prompts