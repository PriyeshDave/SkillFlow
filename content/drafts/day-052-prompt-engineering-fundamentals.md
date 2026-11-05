---
day: 52
generated_at: '2026-11-05T16:54:56.552116+00:00'
phase: Phase 7 — Prompt Engineering
recap_summary: Explained the fundamentals of prompt engineering, including key prompt
  components, practical techniques for controlling LLM outputs, and common limitations
  and sensitivities of prompting.
status: pending_review
title: 'Day 52: Prompt Engineering Fundamentals'
topic_title: Prompt Engineering Fundamentals
---

### What Is Prompt Engineering?

Prompt engineering is the skill of crafting the instructions—called *prompts*—that you give a large language model (LLM), like GPT-4 or similar AI systems. Prompts are plain language input: questions, instructions, or requests. The way you phrase a prompt can dramatically change the AI’s response.

Think of prompt engineering like directing a very capable but literal-minded assistant. If your directions aren’t clear, the results will be imprecise or off-target. The prompt is how you control the model’s behavior, output, and style.

Prompt engineering is central to working productively with LLMs. Any time you use a language model for tasks like writing help, code generation, question answering, research, or data analysis, you use prompt engineering. If you want consistent and reliable results, this is an essential skill.

### Why Do Prompts Matter?

LLMs don’t read your mind—they process only the words you input. Changing a few words or adjusting structure can lead to very different answers. This isn’t a side effect; it’s fundamental to how these models operate.

For example:

- **Prompt 1:** `Summarize the following text:`
- **Prompt 2:** `Write a bullet-point summary of the following text:`

Given the same block of text, the model will likely return one paragraph for the first and a list of bullet points for the second. By specifying “bullet-point summary,” you control the answer’s structure and style.

Even small wording changes matter:

- **Prompt 3:** `Explain to a five-year-old: Why is the sky blue?`
- **Prompt 4:** `Explain in detail: Why is the sky blue?`

Prompt 3 signals the model to use simple language. Prompt 4 invites a more technical answer. The model is following your lead.

### Key Elements of a Prompt

Good prompts often have four key ingredients: clarity, context, specificity, and intent. Here’s what each means, with examples:

**Clarity**: The model needs to understand exactly what you want. Avoid vague requests.

- Unclear: `Tell me about cells.`
- Clear: `Give a one-sentence definition of a biological cell.`

**Context**: Provide background information when needed so the model doesn’t have to guess.

- No context: `Translate this.`
- With context: `Translate the following text from English to French:`

**Specificity**: The more details you give, the more precise the output.

- Unspecific: `Summarize this article.`
- Specific: `Summarize this article in three sentences, focusing only on the main economic impacts discussed.`

**Intent**: State your goal for the information or describe the format you want.

- No intent: `What is quantum computing?`
- With intent: `Explain quantum computing in simple terms for a non-technical audience.`

A strong prompt combines these ideas. For example:

```
Act as a project manager. Based on the following notes, write a list of actionable next steps for the engineering team, focusing only on tasks due this week.
```

You set the role (project manager), the format (list), the recipient (engineering team), and a constraint (this week’s tasks).

### Common Prompt Engineering Techniques

A few simple techniques can make a big difference in LLM results.

**1. Format Specification**

Tell the model exactly how to format the answer.

- `Return the result as a JSON object with fields "name" and "age".`
- `Answer in a table with columns for pros and cons.`

**2. Few-Shot Prompting (Providing Examples)**

Give the model sample inputs and desired outputs. This is called “few-shot” because you show a few examples and then provide a new task.

Example:

```
Convert the following sentences from passive voice to active voice.
Example: The ball was thrown by John. -> John threw the ball.
Sentence: The cake was eaten by the children.
```

The model uses the pattern you set in your example.

**3. Specifying Roles or Constraints**

You can bias the tone, detail, or viewpoint by assigning a role or setting a constraint.

- `You are a helpful research assistant. Summarize the paper's main findings in three sentences.`
- `Answer as a sarcastic movie critic:`

**4. Step-by-Step Reasoning**

Prompt the model to lay out its reasoning step by step.

- `Explain your answer step by step.`
- `List all the assumptions you are making in detail.`

Each technique gives you more control. Combined, they help you get closer to the output you want.

### Limitations and Challenges

Prompt engineering is powerful, but not all-powerful.

**Unpredictability and Sensitivity**: Small changes in wording can lead to unexpected swings in the answer. The model can react in ways that aren’t always obvious.

**Knowledge Boundaries**: A prompt can’t give the model access to new or future facts it wasn’t trained on. If you ask for something it doesn’t know, it may make up (“hallucinate”) plausible-sounding— but incorrect—answers.

**Context Limits**: Every LLM has a maximum amount of text it can consider at once, called a “context window.” Very long inputs can get cut off or ignored.

**No Absolute Precision**: The same prompt won’t always produce exactly the same reply. LLM outputs can vary on each run, depending on settings or randomness.

Prompt engineering helps you *guide* the model. It’s more like writing clear instructions for a smart human than programming a strict computer script.

### Code Example: Changing Prompt, Changing Output

Here's a code example using the OpenAI API. It shows how a small change in the prompt yields a noticeably different result.

```python
import openai

openai.api_key = "your-api-key"

text_to_summarize = (
    "OpenAI is an AI research and deployment company. "
    "Our mission is to ensure that artificial general intelligence benefits all of humanity. "
    "We’re committed to ensuring that artificial general intelligence—AI systems that are generally smarter than humans—benefits everyone."
)

prompts = [
    "Summarize the following text:",
    "Write a bullet-point summary of the following text:"
]

for prompt in prompts:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": f"{prompt}\n\n{text_to_summarize}"}],
        max_tokens=100
    )
    print(f"\nPrompt: {prompt}\nOutput:\n{response['choices'][0]['message']['content']}")
```

Run this code. For the first prompt, the model usually gives a paragraph. For the second, it returns bullet points. Modifying your instructions—not the model—makes the difference. This illustrates how crucial prompt specificity is.

---

## Key Takeaways

- Prompt wording directly shapes the output and behavior of large language models.
- Effective prompts use clarity, context, specificity, and clear intent.
- Common techniques include format specification, few-shot prompting, role assignment, and step-by-step reasoning.
- Even minor changes in a prompt can result in dramatically different responses.
- Prompt engineering cannot overcome model knowledge limits or ensure perfect consistency in outputs.

## Try It Yourself

Pick a simple factual topic, such as Newton’s Laws of Motion. Write two prompts for an LLM: one requesting a general explanation, and another asking for the explanation as a poem or a story. Submit both prompts to the LLM and compare the responses. Note what aspects of the outputs changed and how your prompts influenced them.

## Further Resources

- 🎥 [Prompt Engineering Fundamentals for ChatGPT, Gemini, and Claude (YouTube video by Jono Catliff)](https://www.classcentral.com/course/youtube-99-of-people-still-don-t-know-the-basics-of-prompting-chatgpt-gemini-claude-489755)
- 📄 [Prompt Engineering Fundamentals (IBM Developer article)](https://developer.ibm.com/articles/awb-prompt-engineering-fundamentals/)
- 📘 [Prompt Engineering Guide (GitLab Docs)](https://docs.gitlab.com/development/ai_features/prompt_engineering/)
- 📄 [A Tutorial on Prompt Engineering: From Messy Thoughts to AI Workflows (arXiv, Sept 2026)](https://arxiv.org/abs/2609.22249)

---

**Coming up on Day 53:** Zero-Shot vs. Few-Shot Prompting