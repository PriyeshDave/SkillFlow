---
day: 57
generated_at: '2026-11-12T17:11:52.764531+00:00'
phase: Phase 7 — Prompt Engineering
recap_summary: Explained the distinction between system prompts and user prompts in
  AI language models, detailing their roles, examples, and importance for shaping
  AI behavior in various applications.
status: pending_review
title: 'Day 57: System Prompts vs. User Prompts'
topic_title: System Prompts vs. User Prompts
---

## What Are Prompts in AI?

When you interact with an AI like ChatGPT or any large language model, you send it instructions written in everyday language. These instructions are called **prompts**. A prompt can be a question, a task, or a set of detailed instructions.

The AI doesn't understand your intention by default. It generates responses based on your prompt. Think of the prompt as the steering wheel: it sets the direction for the conversation.

Not all prompts are the same. How instructions reach the model—and at what point—makes a key difference in controlling AI behavior.

---

## User Prompts: Direct Instructions

A **user prompt** is the message you type directly to the AI. This is the most obvious and visible prompt. For example:

```
What is the capital of France?
```

Or a more specific task:

```
Write a Python function that adds two numbers.
```

User prompts are what you enter into a chat box, a form, or an API. They state what you want from the AI right now. You can add context (“as a helpful tutor, explain this concept to me”), or just be direct (“summarize this text”).

The AI responds directly to user prompts. These prompts give the immediate instruction for the next output.

---

## System Prompts: Setting Context and Rules

A **system prompt** gives instructions before the user ever says anything. It usually comes from the developer or the application itself, not from the end user. Most of the time, users never see the system prompt.

System prompts are like ground rules set ahead of time. They define the AI’s role, tone, or boundaries. For example:

```
You are a professional medical assistant. Always answer with empathy and provide only factual, referenced medical information. Do not provide diagnoses.
```

Or:

```
Respond using only emojis.
```

A system prompt is usually set when a session starts. It shapes every answer the AI gives in that session. The AI follows these background instructions alongside whatever the user asks.

To compare:

- **User prompt:** What the end-user types in, visible and direct.
- **System prompt:** Background instructions, typically invisible, guiding overall behavior.

---

## Why Separate System and User Prompts?

Each prompt type serves a different purpose:

- **User prompts** set the *immediate* request. The AI tailors its answer to what the user just asked.
- **System prompts** provide *ongoing context* and rules. They shape all answers, no matter what the user asks.

Suppose you want every answer to sound like a pirate. You put this instruction in the system prompt. Now, no matter what users ask, the AI answers in pirate style—unless a clear user prompt overrides it.

User prompts alone can’t lock in a personality or set boundaries for an entire session. System prompts can. This separation lets engineers build AIs that always stay in character, follow compliance policies, or use company terminology.

---

## Real Example: Same Model, Different Characters

Let’s see how this works in practice. Imagine you have one AI model, but you use it in two applications:

1. **Doctor's Office Assistant:** Gives careful, factual medical answers.
2. **Kids’ Science Tutor:** Uses simple, playful explanations.

Let's use the same user prompt for both apps:

```
What happens when you get a cold?
```

But each app loads a different system prompt behind the scenes.

**System Prompt for Medical Assistant:**
```
You are a professional medical assistant. Keep answers factual, neutral, and reference official sources. Do not provide a diagnosis.
```

**System Prompt for Kids' Tutor:**
```
You are a fun science tutor for kids. Explain things simply and use playful, friendly language. Make learning exciting!
```

In Python, using the OpenAI API, you could set this up as follows:

```python
import openai

user_prompt = "What happens when you get a cold?"

doctor_system_prompt = "You are a professional medical assistant. Keep answers factual, neutral, and reference official sources. Do not provide a diagnosis."

kids_tutor_system_prompt = "You are a fun science tutor for kids. Explain things simply and use playful, friendly language. Make learning exciting!"

doctor_response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": doctor_system_prompt},
        {"role": "user", "content": user_prompt}
    ]
)

kids_response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": kids_tutor_system_prompt},
        {"role": "user", "content": user_prompt}
    ]
)

print("Doctor Assistant:", doctor_response.choices[0].message.content)
print("Kids' Science Tutor:", kids_response.choices[0].message.content)
```

The model itself doesn't change. The user prompt also stays the same. The different system prompts are what cause the responses to have completely different tone and content.

---

## Why This Matters for Prompt Engineering

Knowing the difference between system and user prompts lets you build more powerful AI applications.

System prompts give your AI default rules at the start of a session. They let you:

- Set up context or a persona for each application.
- Enforce policy or legal guidelines without repeating yourself.
- Change the model’s tone, attitude, and terminology across different domains.

However, system prompts can't make models do what they're fundamentally unable to do. If the system prompt and user prompt conflict, the model might ignore one or get confused. Also, not all APIs or platforms let you control system prompts—always check the documentation.

Good prompt engineering means combining these tools:

- Use **system prompts** for broad, ongoing rules or context.
- Use **user prompts** for direct, immediate instructions.

Separating these lets you design AIs that are both robust and user-friendly. This distinction is a basic ingredient of any AI application that relies on language models.

---

## Key Takeaways

- User prompts are direct instructions provided by the end-user to the AI.
- System prompts are background instructions set by developers to define the AI's tone, persona, or boundaries.
- System prompts apply across an entire session and influence all responses unless overridden.
- Separating system and user prompts enables more flexible and reliable AI applications.
- System prompts are crucial for enforcing rules and consistency across domains or user interactions.

## Try It Yourself

Write a user prompt asking for a restaurant recommendation. Then, add a system prompt that instructs the AI to reply as a pirate. Compare the responses and note how the system prompt affects the tone and style of the reply.

## Further Resources

- 🎥 [System Prompt vs User Prompt in Prompt Engineering](https://www.youtube.com/watch?v=b5onomLAQKc)
- 📘 [Prompt Engineering | OpenAI API documentation](https://developers.openai.com/api/docs/guides/prompt-engineering)
- 📄 [Chapter 6: Prompt types (system vs user prompts) – Prompt Engineering in Practice (Manning preview)](https://www.manning.com/preview/prompt-engineering-in-practice/chapter-6)
- 📄 [System Prompts vs. User Prompts – The Prompt Bench](https://thepromptbench.com/prompting-fundamentals/system-prompts-vs-user-prompts/)

---

**Coming up on Day 58:** Prompt Injection and Prompt Security Basics