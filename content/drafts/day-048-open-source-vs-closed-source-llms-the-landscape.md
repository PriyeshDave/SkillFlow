---
day: 48
generated_at: '2026-10-30T15:24:23.888831+00:00'
phase: Phase 6 — The Rise of LLMs
recap_summary: Explained the fundamentals of Large Language Models (LLMs), the distinction
  between open-source and closed-source LLMs, their licensing implications for engineering
  and product use, and compared examples of each type along with their practical tradeoffs.
status: pending_review
title: 'Day 48: Open-Source vs. Closed-Source LLMs: The Landscape'
topic_title: 'Open-Source vs. Closed-Source LLMs: The Landscape'
---

## What Are LLMs? A Brief Refresher

A **Large Language Model (LLM)** is a type of artificial intelligence designed to process, generate, and understand natural language—plain human language, such as English sentences. LLMs learn by studying huge collections of text and discovering patterns in how words and sentences fit together. The “large” refers to the number of parameters, which are adjustable values inside the neural network that the model uses to learn and make predictions.

When you give an LLM a prompt—for example, a question, sentence starter, or a block of text—it tries to predict what comes next, based on its training. This is how chatbots, code-writing tools, and AI summarizers work. For instance, if you type "Once upon a time," the LLM generates a story continuation, drawing from what it's learned about stories from its training data.

## The Meaning of Open Source and Closed Source

**Open source** means the code (or model files, or training data) is publicly available. Anyone can inspect it, copy it, change it, and share their changes. Well-known open source software includes the Linux operating system and the Python language. For LLMs, open source typically means you can download the trained model weights—the values that make the model smart—and sometimes the code and data used to build them.

Think of open source like sharing not just your secret family recipe, but also allowing anyone to use it, tweak it, and even sell their adapted version. **Closed source**, on the other hand, is like selling cookies while keeping the recipe secret—even the ingredients may be hidden.

**Closed source** means the code or model is not shared with the public. You might be able to use the product—like chatting with an AI on a website—but you can’t see how it works, examine its calculations, or change it. The company decides who can use it, under what conditions, and often charges for access.

## Why Do LLM Licenses Matter?

The difference between open and closed source affects day-to-day work for engineers:

- **Usage**: Open models can run on your own computer or server. Closed models are usually accessible only by sending queries to a company's server through an API (a controlled gateway for your requests).
- **Modification**: You can retrain, fine-tune, or change open models (if you have the hardware and know-how). Closed models can't be changed—you have to wait for the company to add features.
- **Deployment**: Open-source LLMs let you decide where to run them—crucial for privacy, security, and legal reasons. Closed models usually require sending your data offsite, outside your organization’s control.
- **Research and Innovation**: Open models allow researchers to test, modify, and study their behavior. Closed models limit your ability to understand or analyze why a model made a specific prediction.
- **Product Development**: Building a product around a closed API ties you to the vendor’s pricing, terms, and reliability. Open-source LLMs mean greater independence, but you’re also responsible for maintenance and scaling.

## Examples of Open-Source LLMs

- **GPT-2, GPT-Neo, GPT-J, GPT-NeoX**: Open models inspired by OpenAI’s GPT series. Smaller than the most advanced GPTs, but still able to write texts, code, and answer questions.
- **LLaMA, LLaMA 2 (Meta)**: Released by Meta, these models are available for both research and (with LLaMA 2) commercial use. They’re efficient enough to run on many standard systems.
- **MPT (MosaicML)** and **Falcon**: Actively developed with broad community support.
- **Vicuna** and **Alpaca**: Fine-tuned chat-focused models built by academic and research teams on top of open bases.

To use an open-source LLM, engineers often rely on libraries like Hugging Face Transformers, which make loading and generating text from models straightforward.

**Example:** Loading and generating text from an open-source LLM with Hugging Face Transformers.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "distilgpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

prompt = "The future of AI is"
inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(**inputs, max_length=20)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(result)
```
This example runs entirely on your machine. It doesn't need to connect to an outside server.

## Examples of Closed-Source LLMs

- **GPT-3, GPT-4, ChatGPT (OpenAI)**: Available only via API or web chat. You can’t access the training data, code, or model weights.
- **Claude (Anthropic)**: Only accessible through Anthropic’s service. You send text and get responses, but the underlying model is hidden.
- **Gemini (Google/DeepMind)**: Used in Google’s products or via API, not available for download.
- **Cohere, Jasper, AI21 Studio**: Commercial providers offering access via paid API but not releasing the models.

**Example:** Sending a prompt to a closed-source LLM using OpenAI’s GPT-3.

```python
import openai

openai.api_key = "your-api-key"

response = openai.Completion.create(
    model="text-davinci-003",
    prompt="The future of AI is",
    max_tokens=20,
)

print(response.choices[0].text.strip())
```
Here, your prompt is sent to OpenAI’s servers, and you get back a generated response. An API key is required, and your usage is visible to OpenAI.

## Comparing the Two: Pros and Cons

Direct comparison of open and closed LLMs:

- **Flexibility**: Open models can be modified, retrained, or run anywhere. Closed models work only within the provider’s limits.
- **Reliability and Support**: Closed models usually offer high uptime, strong support, and fewer surprises if you want things to "just work." Open models depend on your own or community support, which can vary.
- **Customization**: Open models can be fine-tuned to your data—useful for unique industries or jargon. Most closed models can't be customized directly; some offer custom training as a paid service.
- **Security and Privacy**: Open models keep your data in-house. Closed models require you to send prompts and possibly sensitive data over the internet.
- **Community and Ecosystem**: Open models attract researchers and startups. Development can be fast but may lack polish. Closed models offer more mature tools, better documentation, and customer support but less transparency.
- **Performance**: As of mid-2024, leading closed models (like GPT-4 or Claude 3) tend to outperform open models, especially in reasoning and safety. Open models improve rapidly, but closed models still hold the lead in important areas.
- **Cost**: Running open models means paying for hardware upfront, but operating costs can be lower. Closed models charge per use, which adds up with heavy use.

There’s no single best option for everyone. Choose based on what matters for your project: customization, cost, privacy, support, and performance. Knowing the strengths and tradeoffs of each approach helps you make the right decision as an engineer or builder.

---

## Key Takeaways

- LLMs are AI models trained on large text datasets to generate and understand human language.
- Open-source LLMs allow full access to model code and weights for modification, research, and deployment.
- Closed-source LLMs restrict access to APIs only, with underlying code and data kept secret.
- Key tradeoffs include flexibility, privacy, support, performance, and cost when choosing between open and closed LLMs.
- Engineers must consider licensing and usage terms when integrating LLMs into products.

## Try It Yourself

Pick an LLM (open or closed source) that you’ve heard about. Look up its official model card or homepage, determine whether it is open-source or closed-source, and write down one specific way its licensing status would impact your ability to use or modify it in a project.

## Further Resources

- 🎥 [Open Source vs Closed AI: LLMs, Agents & the AI Stack Explained](https://www.youtube.com/watch?v=_QfxGZGITGw)
- 📄 [How far behind are open models? (Epoch AI report)](https://epoch.ai/publications/open-models-report)
- 📄 [Bridging the LLM Accessibility Divide? Performance, Fairness, and Cost of Closed versus Open LLMs for Automated Essay Scoring](https://aclanthology.org/2025.gem-1.60/)

---

**Coming up on Day 49:** Understanding Context Windows