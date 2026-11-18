---
day: 61
generated_at: '2026-11-18T17:39:54.488383+00:00'
phase: Phase 8 — Working With LLM APIs
recap_summary: Explained how large language models process text as tokens, how tokenization
  differs from word separation, and how token counts impact API usage, pricing, and
  rate limits. Provided a code example to show how to count tokens in practice.
status: pending_review
title: 'Day 61: Understanding Tokens, Pricing, and Rate Limits'
topic_title: Understanding Tokens, Pricing, and Rate Limits
---

## What Are Tokens in Language Models?

Large language models (LLMs) process text in units called "tokens." Tokens are the smallest pieces of text the model understands.

A token is not just a word. It might be a whole word, part of a word, a space, a punctuation mark, or even a special symbol. The exact way text turns into tokens is set by a "tokenizer." A tokenizer is a set of rules (or an algorithm) that chops up text into chunks the model can handle.

For example:

- The sentence: **"We saw 100 elephants!"**
- Might be split into these tokens:
  - "We"
  - " saw"
  - " 100"
  - " elephants"
  - "!"

Notice how spaces often stick to the following word (" saw") and how the number (" 100") is a separate token. Different tokenizers may break up words, contractions, numbers, or punctuation in their own ways.

If you include an emoji (🙂), a URL, or programming code, tokenizers may split those even further. The key idea: tokens are not the same as words—they are the basic chunks models actually process.

## How Tokens Affect LLM API Usage

When you use an LLM API (like OpenAI’s GPT-4 or Anthropic's Claude), every message is counted in tokens. You are charged, limited, and managed based on tokens—not just by letter, word, or sentence.

There are two main types of token usage:
- **Input tokens:** All the tokens in the text you send.
- **Output tokens:** All the tokens in the text the model writes back.

Both matter. Providers count both input and output tokens for each request.

For example:
- You send: "Hello, how are you?" (input tokens)
- The model replies: "I'm fine, thank you." (output tokens)

You are billed* for the sum of both.

Why does this matter? Models store, compute, and generate everything using tokens. A longer prompt or a longer reply always uses more tokens, directly influencing cost and speed.

Most APIs have a hard cap on the total tokens allowed per request (for example, 4096 tokens per call). The total includes both your input and the model's reply.

## Pricing: How Token Counts Turn Into Costs

LLM APIs charge based on token count. Not by words, time, or number of calls. Prices can vary by model, and sometimes by input versus output tokens.

Here's a typical price example (as of mid-2024—always check the latest docs):

- OpenAI might charge **$0.0015 per 1,000 input tokens**
- And **$0.002 per 1,000 output tokens** for a certain model

Let's break it down step by step:
- You send a prompt: 50 tokens (input)
- The model responds: 100 tokens (output)
- Total tokens: 50 + 100 = 150

Prices are set "per 1000 tokens," so:
- 150 / 1000 = 0.15 (of a block)

Now the actual cost:
- Input: 50 / 1000 × $0.0015 = $0.000075
- Output: 100 / 1000 × $0.002 = $0.0002
- **Total**: $0.000275 for this exchange

Multiply this by thousands or millions of requests and the costs add up. Both prompt length and response length matter for pricing.

## Understanding Rate Limits

API providers set usage limits called "rate limits" to keep the system stable and fair for all. Rate limits control:
- **Requests per time window** (for example: no more than 60 calls per minute)
- **Tokens per time window** (for example: no more than 250,000 tokens per day)

These rules prevent one user or app from overloading the service.

Concrete examples:
- 100 requests per minute (request limit)
- 500,000 tokens per day (token limit)

Some providers also use "burst limits." These control how much you can do in very short periods (for example: no more than 20 calls in 10 seconds). This stops brief traffic spikes from overwhelming the system.

Your app must track both total tokens and number of calls. If you go over either limit, the API may reject requests or slow them down ("throttling").

## A Simple Token Counter in Code

To control costs and avoid hitting request limits, you need a way to estimate how many tokens your text will use—before you send it to the API.

OpenAI offers a Python library called `tiktoken`. This tool counts tokens the same way the model will.

Here is a minimal code example showing token counts for different kinds of text:

```python
import tiktoken

# Use the encoding for your target model, e.g., GPT-3.5 or GPT-4
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

examples = [
    "Hello, how are you?",
    "This is an email: john.doe@example.com",
    "Let's write some code: def foo(): return 42",
    "100 elephants walked over 1,000,000 miles!"
]

for text in examples:
    tokens = encoding.encode(text)
    print(f"Text: {repr(text)}")
    print(f"Token count: {len(tokens)}\n")
```

Different types of text can have very different token counts, even if they look similar in length. For example, numbers, code, or punctuation are sometimes split into more tokens than plain words.

Counting tokens before you send a prompt lets you predict cost and obey token limits. This is a key habit for controlling your application's expenses and reliability when working with LLMs.

---

## Key Takeaways

- Tokens are the units LLMs use to process text, not always matching words.
- Token counts directly affect API costs and speed.
- Both input and output tokens are billed separately in LLM APIs.
- API usage is limited by both number of requests and token count.
- You can estimate token counts and costs in advance using tools like tiktoken.

## Try It Yourself

Pick three short text samples: a greeting, a simple sentence, and a short paragraph. Use the provided Python code with tiktoken to count the tokens in each. Based on a typical API's per-token pricing (e.g., $0.0015 for 1,000 input tokens), calculate the estimated cost if you sent each to an LLM API.

## Further Resources

- 🎥 [What are LLM Tokens and API Prices? (Beginner Friendly)](https://www.youtube.com/watch?v=7zbuEULTHIs)
- 📘 [Rate limits · OpenAI API](https://developers.openai.com/api/docs/guides/rate-limits)
- 📄 [How to handle rate limits · OpenAI Cookbook](https://developers.openai.com/cookbook/examples/how_to_handle_rate_limits?translationFallback=zh-Hant)
- 📄 [Build a per‑run spending controller with the Responses API](https://developers.openai.com/cookbook/articles/per_run_spending_controller_responses_api)

---

**Coming up on Day 62:** Temperature, Top-P, and Sampling Parameters Explained