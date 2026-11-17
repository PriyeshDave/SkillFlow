---
day: 60
generated_at: '2026-11-17T17:58:11.558370+00:00'
phase: Phase 8 — Working With LLM APIs
recap_summary: Explained the concept of LLM APIs, the reasons for using hosted APIs,
  how typical API calls work, main providers in the space, common parameters, and
  important security and cost considerations when connecting apps to LLM APIs.
status: pending_review
title: 'Day 60: Introduction to LLM APIs (OpenAI, Anthropic, and Others)'
topic_title: Introduction to LLM APIs (OpenAI, Anthropic, and Others)
---

### What is an LLM API?

An API (Application Programming Interface) is a set of rules that lets two computer programs exchange information, no matter how they're built or where they run. Think of it as a shared language that software uses to talk to other software.

A large language model (LLM) is a type of artificial intelligence trained to understand and generate human language. Examples include OpenAI’s GPT-4 and Anthropic’s Claude. These models are called "large" because they have billions of adjustable settings—called parameters—which makes them powerful, but also requires a lot of computing resources.

Running a modern LLM on your own laptop isn't practical. The model files are enormous. You’d need special hardware (like graphics processing units, or GPUs), a lot of memory, and fast networking—things most laptops don’t have. That’s where an *LLM API* comes in. Instead of loading the model yourself, you send your text to a company over the internet. That company runs the model on their own servers and sends you back a response.

With an API, you don’t manage hardware or software setup. You just send a request and receive an answer.

---

### Why Use Hosted LLM APIs?

Engineers usually choose hosted LLM APIs instead of running models themselves for four main reasons:

1. **Computational Complexity:** Modern LLMs are huge. For example, GPT-4 can need hundreds of gigabytes of memory to run. Companies like OpenAI spread this work across many high-end servers.

2. **Infrastructure Requirements:** Running an LLM isn’t just about powerful hardware. You need to keep machines updated, patch software, and manage networks. API providers handle all of this behind the scenes.

3. **Cost:** Paying per use through an API is usually cheaper than buying and maintaining your own GPU servers—unless you need millions of responses daily.

4. **Access to New Models:** Providers keep improving their models and rolling out better versions. By using an API, your app can use the latest models with zero extra setup.

For example, if a small startup wants to add a chatbot to their website, using an LLM API lets them launch in a day—no big hardware or infrastructure purchases required.

---

### A Sample LLM API Call

Interacting with an LLM API means making a network request, usually using HTTP. Most APIs use a POST request to a special web address, called an endpoint.

To use these APIs, you need an *API key*: a long code unique to you, which unlocks access and tracks how much you use.

Inputs and outputs are usually sent as JSON (JavaScript Object Notation), which is a simple, readable way to send data.

Here's a basic example using Python, for OpenAI’s chat completion API:

```python
import requests

API_KEY = "sk-..." # Put your actual API key here

url = "https://api.openai.com/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}
json_data = {
    "model": "gpt-3.5-turbo",
    "messages": [
        {"role": "user", "content": "What is the capital of France?"}
    ]
}

response = requests.post(url, headers=headers, json=json_data)
print(response.json()["choices"][0]["message"]["content"])
```

- The *endpoint* is the URL you send your request to.
- The *API key* goes in the headers to prove who you are.
- The *input* is the `messages` list—a series of chat turns or prompts.
- The *output* is the generated text, inside the JSON response.

Most LLM APIs use this same pattern: send JSON, get JSON back.

---

### Key Providers: OpenAI, Anthropic, and Others

Several companies run leading LLM APIs:

- **OpenAI:** Offers GPT-3.5 and GPT-4 models, popular for chat, code help, summarization, and more. Their API is well-documented and supports features like calling functions and giving system instructions.

- **Anthropic:** Makes the "Claude" models, focusing on safe, helpful responses. Anthropic’s API supports chat and document queries, and often gives longer replies than some competitors.

- **Google:** Provides "Gemini" and "PaLM" models via API, strong in handling multiple types of input (text, images) and following instructions. Google’s Vertex AI platform makes these models available for business users.

- **Cohere, AI21, Mistral:** Each brings something different. Cohere’s API is often used for document search and classification. AI21 makes Jurassic, another text generation family. Mistral stresses efficiency, and in some cases, open-source-style access.

All of these providers use similar ideas for how the APIs work.

---

### Common LLM API Parameters

Across providers, you’ll see a few parameters that let you fine-tune model behavior:

- **Model Name:** Picks the specific LLM to use, like `"gpt-4"` or `"claude-v2"`. Newer models handle more complex requests, but may cost more.

- **Temperature:** A setting (usually between 0 and 2) that controls randomness. Lower values (like 0) make responses more predictable and repeatable. Higher values (like 1.0) add more surprise and creativity. If you want steady answers, keep the temperature low. If you want creative writing, turn it up.

- **Max Tokens:** This controls the length of the generated response. A "token" is a small text chunk, on average about 4–5 characters for English. Fewer tokens means shorter, faster, cheaper outputs.

- **Streaming vs. Non-Streaming:** "Non-streaming" means you get the whole response at once. "Streaming" sends the result as it’s generated, bit by bit—useful for real-time interfaces. Not every API supports streaming, but it's becoming more common.

Once you know these options, you can shape LLM outputs to suit your app and control costs.

---

### Security and Cost Considerations

When you connect your app to an LLM API, be mindful of several safety and budgeting issues:

- **Privacy & Data Ownership:** Anything you send to an LLM API goes to someone else’s servers. That provider might keep logs for research or debugging. Check their privacy policy and data retention rules.

- **Sensitive Data:** Avoid sending confidential or regulated information (like user passwords or medical data) unless you have a clear, written guarantee and the right setup—usually, that means a special business agreement.

- **Rate Limits:** LLM APIs limit how many requests you can send per minute or day. Hitting the limit means you’ll get errors or delays.

- **Usage Cost:** You pay by the number of tokens processed (both input and output). Costs can climb quickly as your app grows. Always estimate expected usage, and set hard limits or automatic budget alerts.

Before you send any information to an LLM API, ask yourself: “Would this be safe if an engineer at the API company saw it?” If the answer is no, don’t send it unless you trust the provider and have strong protections in place.

---

---

## Key Takeaways

- LLM APIs let apps interact with powerful language models without running them locally.
- Hosted LLM APIs simplify infrastructure and provide access to up-to-date models.
- API calls typically use HTTP with JSON input and output, authenticated by an API key.
- Major providers include OpenAI, Anthropic, Google, Cohere, AI21, and Mistral.
- Controlling costs and safeguarding sensitive data are critical when using LLM APIs.

## Try It Yourself

Create a free developer account with OpenAI or Anthropic, generate an API key, and use Python or curl to make your first LLM API chat or completion request. Pay attention to how you include the API key, examine the JSON request and response, and note how the model returns answers.

## Further Resources

- 🎥 [Build an LLM Gateway in Python | Connect OpenAI and Anthropic with One Function](https://www.youtube.com/watch?v=yc4oDNKEfQQ)
- 🎥 [LLM Fundamentals Crash Course For Beginners 2026 (Tokens, Prompting, APIs & Tool Calls)](https://www.youtube.com/watch?v=YHNi88FaGug)
- 📄 [Chapter 11: Working with LLM APIs | Building Language AI](https://llmbook.icsgen-ai.org/part-3-working-with-llms/module-11-llm-apis/index.html)
- 📄 [LLM APIs: a first request to Claude and OpenAI](https://interlab.app/en/blog/llm-api-pervyj-zapros)
- 📄 [LLM App Development #1: Your First API Call and Environment Setup](https://www.schoolofweb.net/en/posts/llm-app-dev-1-first-api-call-and-setup/)

---

**Coming up on Day 61:** Understanding Tokens, Pricing, and Rate Limits