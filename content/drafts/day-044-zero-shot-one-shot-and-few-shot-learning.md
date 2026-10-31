---
day: 44
generated_at: '2026-10-26T15:34:43.997354+00:00'
phase: Phase 6 — The Rise of LLMs
recap_summary: Explained zero-shot, one-shot, and few-shot learning in LLMs, showing
  how the number and quality of prompt examples affect task adaptation and results.
status: pending_review
title: 'Day 44: Zero-Shot, One-Shot, and Few-Shot Learning'
topic_title: Zero-Shot, One-Shot, and Few-Shot Learning
---

**Previously, on Day 43:** Explained in-context learning, where language models adapt to new tasks by imitating examples in the input prompt rather than updating their internal parameters, and contrasted this approach with traditional fine-tuning.

---

## What Is Zero-Shot, One-Shot, and Few-Shot Learning?

In machine learning, a "shot" means an example.  
**Zero-shot learning**: The model is asked to solve a task with no examples to guide it.  
**One-shot learning**: The model gets a single example before tackling the task.  
**Few-shot learning**: The model sees a small handful—usually 2 to 10—examples before trying the task itself.

Think of learning a new card game:

- **Zero-shot**: You get the rulebook, but no demonstration. You’re expected to play based only on the written instructions.
- **One-shot**: You see one round played. Then you join the game.
- **Few-shot**: You watch a few rounds, noticing variations and quirks, then it’s your turn to play.

### Why Is This Useful?

Large language models (LLMs) such as GPT-3 and GPT-4 can pick up a task just by seeing how you format examples in the prompt. They do not need retraining for every task. This ability is called **in-context learning**—that is, the model uses context in the prompt to adapt on the fly. The number of examples you provide (zero, one, or a few) often makes the difference between a correct and totally wrong result.

---

## How LLMs Use Prompting for Task Adaptation

When you provide examples in a prompt, an LLM does not adjust its internal parameters. All its knowledge stays the same. Instead, it uses patterns you show as instructions for the examples that follow.

- With **zero-shot** prompting, the model guesses the expected answer using only your request and its background knowledge.
- With **one-shot** or **few-shot** prompting, the examples act as a template. The model picks up your format, answer style, and even little quirks.

The prompt acts as a mini lesson for the model, setting expectations for how to respond.

For example:  
- Zero-shot: "Classify this review as SPAM or NOT SPAM."  
- One-shot: "Here’s a review labeled SPAM. Please label this new review."  
- Few-shot: "Here are three reviews with labels. Now classify the next one."

LLMs combine knowledge from their training with the immediate examples you provide.

---

## Why Does Shot Count Matter?

**More examples help—up to a point.** Each example makes the pattern clearer. They help show the structure of the task, exact formatting, and even edge cases. However:

- On a new or ambiguous task, **zero-shot often fails**. The model can only guess from its prior experience.
- A single clean example helps. But if the example is unclear or very specific, it can quickly mislead the model.
- A handful of well-chosen examples lets the model generalize more robustly. But too many examples can overwhelm the prompt, or introduce conflicting signals.

**Diminishing returns:** The first extra example often has the biggest positive effect. By the fifth or tenth example, benefits are small—or even negative if the examples are inconsistent or noisy.

**Pitfalls to watch for:**
- Bad or strange examples can anchor the model to the wrong answer.
- Showing only rare or weird cases can make the model think all inputs must be handled this way.
- On tasks the model "knows" very well—like English sentiment—zero-shot can sometimes outperform one-shot if your example is weak.

Choosing clear, representative examples and getting the shot count right is as important as the prompt itself.

---

## Real Prompts: Zero-, One-, and Few-Shot Side by Side

Suppose you want the model to classify messages as **SPAM** or **NOT SPAM**. Below are three prompt styles:

**Zero-shot prompt:**
```
Classify the following message as SPAM or NOT SPAM.

Message: Congratulations! You have won a free iPhone. Click here to claim your prize.
Label:
```

**One-shot prompt:**
```
Classify each message as SPAM or NOT SPAM.

Message: Urgent! Your bank account is at risk. Login now to secure your funds.
Label: SPAM

Message: Congratulations! You have won a free iPhone. Click here to claim your prize.
Label:
```

**Few-shot prompt:**
```
Classify each message as SPAM or NOT SPAM.

Message: Reminder: Your subscription is about to expire. Renew now to continue service.
Label: NOT SPAM

Message: Urgent! Your bank account is at risk. Login now to secure your funds.
Label: SPAM

Message: We noticed a login from a new device. If this was you, ignore this message.
Label: NOT SPAM

Message: Congratulations! You have won a free iPhone. Click here to claim your prize.
Label:
```

These are context, not training data. The examples shape what the model expects, including answer format, label spelling, and the line between spam and not-spam.

---

## Limitations and Surprising Results

LLMs can perform extremely well in zero-shot mode for tasks they’ve seen in training—things like spam detection or basic summarization. For specialized or ambiguous tasks, they often need better guidance.

**Common ways things go wrong:**
- A misleading example may anchor the model to a wrong answer or odd format.
- If all your examples are SPAM, the model may bias toward labeling everything as SPAM.
- Too many examples can cause the model to run out of memory (“attention”), forgetting the task or mixing up details.

Sometimes, **few-shot is not better than zero-shot**:
- For simple, familiar tasks, the model’s own background knowledge can carry it—unless your examples are especially on-point.
- For trickier, custom tasks (like sarcasm detection or custom categories), a few strong examples are helpful—but only if they’re well-chosen.

The LLM does not "train" on your prompt. But if you include strange or contradictory cases, it can "overfit"—clinging to those examples and giving unpredictable results.

---

## Code Example: Comparing Zero-, One-, and Few-Shot with OpenAI API

Here’s a minimal Python script using the OpenAI API. This classifies a message as SPAM or NOT SPAM using zero-, one-, and few-shot prompts.

```python
import openai

openai.api_key = "your-api-key"

message_to_classify = "Congratulations! You have won a free iPhone. Click here to claim your prize."

# Zero-shot prompt
zero_shot_prompt = f"Classify the following message as SPAM or NOT SPAM.\n\nMessage: {message_to_classify}\nLabel:"

# One-shot prompt
one_shot_prompt = (
    "Classify each message as SPAM or NOT SPAM.\n\n"
    "Message: Urgent! Your bank account is at risk. Login now to secure your funds.\nLabel: SPAM\n\n"
    f"Message: {message_to_classify}\nLabel:"
)

# Few-shot prompt
few_shot_prompt = (
    "Classify each message as SPAM or NOT SPAM.\n\n"
    "Message: Reminder: Your subscription is about to expire. Renew now to continue service.\nLabel: NOT SPAM\n\n"
    "Message: Urgent! Your bank account is at risk. Login now to secure your funds.\nLabel: SPAM\n\n"
    "Message: We noticed a login from a new device. If this was you, ignore this message.\nLabel: NOT SPAM\n\n"
    f"Message: {message_to_classify}\nLabel:"
)

def classify_with_prompt(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=1,
        temperature=0
    )
    return response.choices[0].text.strip()

print("Zero-shot:", classify_with_prompt(zero_shot_prompt))
print("One-shot:", classify_with_prompt(one_shot_prompt))
print("Few-shot:", classify_with_prompt(few_shot_prompt))
```

Try changing the message or the sample examples. Watch how the output varies with each “shot” style. You’ll see how even a single example—or a change in format—can influence what the model predicts. Sometimes the difference is dramatic. Sometimes it’s subtle. But the patterns you give always matter.

---

## Key Takeaways

- Zero-shot, one-shot, and few-shot refer to the number of task examples given in a prompt.
- LLMs use in-context learning to adapt based only on provided examples, not by retraining.
- More examples often improve performance but have diminishing returns and can introduce confusion.
- Example clarity and relevance critically influence model performance and output style.
- Providing poor or inconsistent examples can mislead the model and reduce accuracy.

## Try It Yourself

Choose a task you know well, like assigning movie genres. Write zero-shot, one-shot, and few-shot prompts for it. Test each prompt with an LLM (via API, Playground, or another tool) and compare the outputs. Reflect on how each prompt style and your chosen examples influence the responses.

## Further Resources

- 🎥 [Few‑Shot & Zero‑Shot Learning: Teach AI with Minimal Data (Like a Human!)](https://www.youtube.com/watch?v=FqCCmLdjjos)
- 🎥 [Master AI Prompting: Zero‑Shot, Few‑Shot & Chain of Thought Explained](https://www.youtube.com/watch?v=sZIV7em3JA8)
- 📄 [Zero‑ and Few‑Shot NLP with Pretrained Language Models (ACL 2022 Tutorial)](https://aclanthology.org/2022.acl-tutorials.6.pdf)
- 📄 [Few‑Shot & Zero‑Shot Learning — Tutorial (ModelRefs)](https://modelrefs.com/tutorials/few-shot-learning/)

---

**Coming up on Day 45:** Emergent Abilities in LLMs