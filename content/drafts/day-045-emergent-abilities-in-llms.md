---
day: 45
generated_at: '2026-10-27T14:59:39.670085+00:00'
phase: Phase 6 — The Rise of LLMs
recap_summary: Explained emergent abilities in large language models, describing how
  new skills—such as arithmetic, translation, code generation, and in-context learning—can
  appear unexpectedly when models reach a certain scale, with examples from nature
  and experiments in AI.
status: pending_review
title: 'Day 45: Emergent Abilities in LLMs'
topic_title: Emergent Abilities in LLMs
---

## What Are Emergent Abilities?

Emergent abilities are skills or behaviors that a system unexpectedly develops once it becomes complex enough—even though these abilities weren’t programmed in directly.

Take water as an example. A single hydrogen or oxygen atom doesn’t have the property of “wetness.” But combine billions of them to make water, and wetness emerges. Wetness can’t be traced to any single atom; it shows up only when there are enough atoms interacting together.

In large language models (LLMs), emergent abilities show up in a similar way. Nobody codes in a “math skill” setting or flips a translation switch. Instead, once a language model crosses a certain size or training level, it suddenly starts demonstrating new skills all by itself. For many engineers, this is surprising—even counterintuitive. Abilities can sharply appear that simply didn’t exist in smaller versions of the model.

## How Emergent Abilities Appear in Large Language Models (LLMs)

When building LLMs, researchers typically start with small models. These can have millions of parameters. A *parameter* is like one simple decision the model can adjust as it learns patterns in text.

At this scale, small models can do basic tasks: autocomplete the next word or echo simple patterns they’ve seen before.

As models grow—training on more data, adding more parameters, and taking longer to learn—they don’t just improve little by little. Sometimes, as the size crosses a certain threshold, entirely new abilities show up without warning.

For example, a model with 100 million parameters might be completely confused by a math word problem. Yet a model with 10 billion parameters can suddenly solve many of them. No engineer added explicit “math logic”—the ability emerged on its own at scale.

This isn’t unique to AI. In nature, an ant colony is guided by thousands of individual ants following simple rules. But when enough ants gather, surprising group behaviors emerge: bridge building, food storage, and more. No single ant “knows” the full plan—the collective behavior appears as the system grows.

## Concrete Examples of Emergent Abilities

Researchers have observed specific skills popping up in large language models only after they become big enough:

- **Arithmetic**: Small models cannot reliably add two numbers stated in a sentence. Increase size, and the model can solve two- and three-digit addition problems.
- **Translation**: Early models would only mimic translation for language pairs seen during training. Larger models can translate between language pairs with little or no direct training data.
- **Code Generation**: Small models repeat programming terms awkwardly. Larger models can produce correct, working code from a natural-language instruction.
- **In-context Learning**: When given a handful of examples (a method called “few-shot prompting”), only large LLMs can extend the pattern and generalize—the smaller ones can’t.

A clear example is arithmetic. Ask a small language model:

"Add 17 and 35."

A tiny model may return something like:

"17 and 35 is a big number."

Meanwhile, a larger model, once past a certain size, suddenly answers correctly:

"17 + 35 = 52."

This shift isn’t because someone added a math plugin to the model. The skill simply **emerges**.

## Why Size Matters: Scaling and Thresholds

A machine learning model’s *size* usually refers to the number of parameters it has. Parameters are the basic “knobs” a model adjusts as it learns from data. Small models have fewer knobs and learn fewer details. Large models—often with billions of parameters—can represent far more complex concepts.

You might expect bigger models to get equally better at all skills. But emergent abilities show up in a non-linear way. There can be long stretches where the skill is completely absent, and then, at a certain size, it suddenly appears.

Imagine piling sand on a table. The pile gets taller in a smooth way—then, suddenly, an avalanche forms. The shift isn’t gradual—it just happens when the pile crosses the right angle.

Similarly, some LLM skills—like basic spelling or finding the next word in a sentence—improve gradually as the model grows. But other abilities—arithmetic, analogies, logic—can appear abruptly, right after the model passes a size threshold.

This means scaling up a model is not just about making existing skills more accurate. It’s about *unlocking* entirely new classes of behavior that never existed in the smaller system.

## Key Takeaways and Open Questions

Emergent abilities show that large language models can go far beyond simple pattern matching. As they grow, they develop surprising new skills—without needing explicit instructions from engineers.

The fact that some skills turn on suddenly, only after reaching a certain scale, tells us something important. We can’t always predict what a large model will be able to do just by looking at smaller versions. Emergence breaks the expectation that every improvement is gradual and predictable.

This unpredictability also shows the limits of our current understanding. We don’t yet know *why* certain abilities appear at some model sizes but not others, or why some tasks require much larger models to become possible. Research is ongoing to understand these leaps—and what sets the thresholds for new skills to emerge.

What is clear: as LLMs grow in size and data, expect not just better performance, but the appearance of whole new abilities—sometimes seemingly out of nowhere.

---

## Code Example: Emergence in Action

Let’s see this for ourselves. We’ll give the same language task—reversing a sentence—to both a small and a larger model. This is an example of *compositional language* ability. Small models often fail. Larger models tend to succeed.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def prompt_model(model_name, prompt):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=30)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

prompt = "Reverse the sentence: The cat sat on the mat."

# Test with a small model
small_model = "gpt2"  # 124M parameters
print("Small model output:")
print(prompt_model(small_model, prompt))

# Test with a larger model
large_model = "gpt2-large"  # 774M parameters
print("\nLarge model output:")
print(prompt_model(large_model, prompt))
```

You’ll likely see the small model give a nonsensical or unrelated answer. The larger model, while not perfect, is much more likely to follow instructions and actually reverse the sentence.

This simple example captures what researchers notice at scale. Once a model is big enough, new abilities—like following complex instructions—appear out of nowhere.

Emergence is not just a buzzword. It’s a real, observed phenomenon in LLMs. Understanding it is essential for working with these models, and for anticipating their rapid and unexpected progress.

---

## Key Takeaways

- Emergent abilities are skills that appear in large models without explicit programming.
- These abilities arise abruptly when the model crosses certain size thresholds.
- Small models cannot perform tasks like arithmetic or complex instructions, while larger ones can.
- Model scaling unlocks not just better performance, but entirely new capabilities.
- The appearance of emergent abilities is unpredictable and challenges our understanding of AI.

## Try It Yourself

Pick a simple language task requiring reasoning—for instance, translating a sentence or solving a basic math problem. Try this prompt with both a large language model (like GPT-3 or GPT-4) and a much smaller model or an autocomplete tool (such as your phone's keyboard prediction). Compare the outputs and observe whether the larger model demonstrates abilities that the smaller one lacks.

## Further Resources

- 🎥 [Stanford CS25: V2 I Emergent Abilities and Scaling in LLMs](https://www.youtube.com/watch?v=tVtOevLrt5U)
- 📘 [Characterizing Emergent Phenomena in Large Language Models (Google Research blog)](https://research.google/blog/characterizing-emergent-phenomena-in-large-language-models/)
- 📄 [Emergent Abilities of Large Language Models (TMLR paper, arXiv)](https://arxiv.org/abs/2206.07682)
- 📄 [Emergent Abilities in Large Language Models: A Survey (arXiv)](https://arxiv.org/abs/2503.05788)

---

**Coming up on Day 46:** RLHF: Reinforcement Learning from Human Feedback