---
day: 49
generated_at: '2026-11-02T18:56:00.238840+00:00'
phase: Phase 6 — The Rise of LLMs
recap_summary: Explained the concept of a context window in large language models
  (LLMs), detailing how text is tokenized, why context window limits matter, and how
  to count and manage tokens for effective application usage.
status: pending_review
title: 'Day 49: Understanding Context Windows'
topic_title: Understanding Context Windows
---

## What Is a Context Window and Why Does It Matter?

A context window is the largest chunk of text a large language model (LLM) can “see” at once when generating new text. Imagine looking through a small moving window, or a camera lens: you only see what fits inside, not the whole scene. For an LLM, the context window limits how much of a conversation, story, or document is available at a time.

This matters because LLMs do not have ongoing memory. They predict the next word—or rather the next token—using only the text inside this window. Anything outside is invisible to the model. It is as if older content never existed. The size of the context window, measured in tokens, sets the hard limit on how much information the LLM can consider at once.

## Tokens: The Building Blocks of LLM Input

LLMs process text in units called tokens, not in words or letters. 

A token is a small chunk of text. Sometimes it is a single character (like ‘a’ or ‘!’). Sometimes it’s a whole word (like ‘cat’ or ‘banana’). Often, it’s part of a word (such as ‘walk’ and ‘ing’). Tokenization is handled by a tokenizer, a pre-processing algorithm that splits text based on rules learned from lots of real text.

Why not use full words? Words in English and other languages follow messy patterns—there are typos, slang, rare words, and punctuation. If models worked only with whole words, they could not handle unfamiliar or misspelled words. If they used only letters, they would have too many pieces to track meaningful context. Tokens strike a balance, capturing enough structure without exploding in length.

The context window is measured in tokens, not words or characters. So, when documentation says an LLM has a 2048-token context window, it means it can only process 2048 tokens at once, no matter how many words or characters that is.

## How Context Windows Work: Concrete Analogy

Picture reading a book with only a paper strip showing 50 words at a time. You move the strip as you read, but can only think about what's visible. Outside the strip, everything is hidden.

LLMs work this way. When the text supplied exceeds the context window, the model ignores the oldest content and sees only the most recent “window's worth” of tokens. The window slides forward as new text is added.

For example, suppose your LLM has a 2048-token window. If your input is 2200 tokens, it will only use the most recent 2048 tokens as context. The first 152 tokens are “out of view.”

## Effects: Loss, Forgetting, and Limits

These limits have important effects:

- **Losing History:** Once text scrolls out of the window, the model forgets it. The LLM doesn't remember prior turns or content unless you include them inside the current prompt window.
- **Dropping Details:** Any useful information outside the current window is lost. The model cannot reference it, and it cannot bring it “back” from memory.
- **Practical Consequences:**
    - Chatbots forget information from earlier in long conversations.
    - Summarizing or analyzing documents longer than the context window means splitting them into chunks small enough for the model to process.
    - Large source code files may need to be broken down, since code generation tools also face this context window limit.

Not every token is the same length in text. “Cat” might be a single token. “Walking” might be two. A long, rare word or a piece of code could be split into many tokens. Emoji and non-English text often break up differently. Because of this, fitting within a context window requires counting tokens, not simply counting words or characters.

## How to Find and Work With a Model’s Context Window

Different LLMs have different context window sizes. For example, GPT-3 “davinci” supports 2048 tokens. GPT-4 and newer models may support 8,192 tokens, 32,768 tokens, or even more. These numbers are in the model documentation. You can usually query the model or find this information through the API.

When building applications, check both prompt and expected output lengths in tokens. If your combined input and output exceed the context window, tokens are cut off or you may get an error. How this is handled depends on the API or platform.

Here's a code example showing how to tokenize text, count tokens, and check against a context window using HuggingFace’s `transformers` library. This uses the GPT-2 tokenizer, but most tokenizers work similarly:

```python
from transformers import GPT2Tokenizer

# Load tokenizer (example: GPT-2)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Set context window size (e.g., 2048 for older GPT models)
CONTEXT_WINDOW = 2048

# Example input text
text = ("In computer science, a context window refers to the amount of information "
        "an AI model can process at once when generating new outputs. It's measured "
        "in tokens—small chunks of text learned from data.")

# Tokenize and count tokens
tokens = tokenizer.encode(text)
num_tokens = len(tokens)

print(f"Number of tokens: {num_tokens}")
if num_tokens > CONTEXT_WINDOW:
    print("Warning: Text exceeds the context window and will be truncated.")
else:
    print("Text fits within the context window.")

# Show the first 10 tokens and their decoded text
print("First 10 tokens:", tokens[:10])
print("Decoded:", tokenizer.decode(tokens[:10]))
```

Replace `'gpt2'` with your chosen model’s identifier. HuggingFace supports many models; see their documentation for supported options. If you use OpenAI’s API, they offer a token counting tool as well.

Understanding the context window—and knowing how to count and manage tokens—makes sure your LLM applications work as expected. You'll avoid problems with “forgotten” content and make better decisions about how to structure prompts and handle longer text.

---

## Key Takeaways

- The context window defines how much text an LLM can process at one time, measured in tokens.
- Tokens may be entire words, characters, or parts of words, depending on the tokenizer.
- LLMs forget any content outside the current context window; no ongoing memory is retained.
- Managing prompt length in tokens is crucial to avoid truncated outputs or errors.
- Tokenizers like those in HuggingFace and OpenAI help count tokens and monitor window limits.

## Try It Yourself

Write a script in your preferred programming language that asks the user for any text input, tokenizes the text, counts the tokens, and compares the total to a specified context window size (like 2048 tokens). Test your script with a paragraph, a full page of book text, and a chat transcript to see how much fits within the window and observe when truncation would occur.

## Further Resources

- 🎥 [Understand LLM Context Windows by Tim Warner](https://www.youtube.com/watch?v=oOL0WY1b_x4)
- 📘 [Context windows – Claude Platform Documentation](https://platform.claude.com/docs/en/build-with-claude/context-windows)
- 📄 [Context Windows – Generative AI Atlas (AWS Labs)](https://awslabs.github.io/generative-ai-atlas/topics/2_0_technical_foundations_and_patterns/2_1_key_primitives/2_1_3_context_windows/2_1_3_context_windows.html)
- 🎥 [Stop Stuffing Your LLM's Context Window (AI‑fokus 2026 talk)](https://confpad.io/conferences/ai-fokus-2026/stop-stuffing-your-llm-s-context-window)
- 📄 [Parallel Context Windows for Large Language Models (ACL Anthology paper)](https://aclanthology.org/2023.acl-long.352.pdf)

---

**Coming up on Day 50:** Why LLMs Hallucinate