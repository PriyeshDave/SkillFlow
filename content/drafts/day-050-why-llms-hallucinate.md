---
day: 50
generated_at: '2026-11-03T16:41:50.014350+00:00'
phase: Phase 6 — The Rise of LLMs
recap_summary: Explained what hallucinations are in large language models, why they
  happen, how model design contributes to them, and current limits in mitigating hallucinations.
status: pending_review
title: 'Day 50: Why LLMs Hallucinate'
topic_title: Why LLMs Hallucinate
---

### What Is a Hallucination in Large Language Models?

In large language models (LLMs) such as GPT-3 or Llama, a "hallucination" means the model generates information that sounds reasonable but is untrue. The model isn’t seeing visions — instead, it’s making things up.

For example, if you ask, “Who was the first person to eat ice cream on the moon?” an LLM might give you a specific astronaut, a date, and even a mission, presented with confidence. In reality, nobody has ever eaten ice cream on the moon. This isn’t just a simple mistake—it’s the model inventing facts.

### Quick Refresher: How LLMs Generate Text

LLMs generate text by predicting one word at a time. They look at the words you give them (the prompt) and use patterns from their training data to guess the most likely next word. This process continues, word by word, until the output is finished.

Think of this like your phone’s predictive text. Type “I am going to the…” and you might get suggestions like “store,” “gym,” or “park.” LLMs do this at a much larger scale and with a deeper sense of context.

LLMs do not check their answers against any database or source of truth while generating text. They cannot browse, look things up, or fact-check. They rely entirely on the patterns they saw during training.

### Where Hallucinations Come From: Training Data Limits

LLMs only learn from the text in their training data. If no one ever wrote about an event, the model can’t possibly "know" about it.

Sometimes the true answer isn’t present in the data the model saw. Sometimes the data has mistakes, or conflicting statements. Many times, the model has no idea that information is even missing.

When asked a question with missing, conflicting, or ambiguous information, the model will still generate an answer. It often stitches together fragments from different topics, or simply invents details. The model does this with the same confidence as when it gives a correct answer—because its system doesn’t actually measure accuracy, only what’s statistically likely.

LLMs do not understand the world. They recognize statistical patterns in words, not reality. They don’t “know” that the moon is a barren place or that eating ice cream there never happened—they just predict text likely to follow a prompt, based on past data.

### How Model Design and Objectives Lead to Hallucinations

Most LLMs are trained to maximize the probability that the next word is correct, given all previous words. This is known as next-token prediction. Their main target is human-like, fluent language—not factual accuracy.

Suppose the prompt is “The largest ocean on Earth is.” The model will predict “the Pacific Ocean” because that fits its data. But if you ask about something where the real answer isn’t clear, it still tries to look helpful and fills in what sounds right—even if it has to invent it.

LLMs don’t truly feel uncertainty. A human would say “I don’t know” when unsure. An LLM usually does not unless specifically trained to do this. Instead, it generates the most plausible answer—even if it’s pure fiction.

Being trained on human conversations or question-answer pairs further reinforces this. The model learns to sound helpful, confident, and authoritative, no matter what information is actually available.

### Why Hallucinations Sound So Convincing

LLM hallucinations are convincing because:

- **Text fluency:** LLMs mimic human writing. Their sentences look and sound right.
- **Borrowed authority:** The model can drop in real names, dates, or places, which makes the response feel believable—even if the details are wrong or fabricated.
- **No visible hesitancy:** Most models don’t add hedges like “maybe” or “I think.” They present hallucinated details as hard facts.

For example:

Prompt: “Who invented the electric lightbulb and in what year?”  
Model output: “The electric lightbulb was invented by Thomas Edison in 1878.”

This answer leaves out key inventors like Humphry Davy and Joseph Swan. It appears definitive, but omits nuance and presents an oversimplified—and partly false—story. That’s typical LLM hallucination: confident, plausible, and not entirely true.

### Limits of Current Approaches: Can We Eliminate Hallucination?

Some techniques to reduce hallucination include:

- **Retrieval augmentation:** Connecting an LLM to external databases or search engines during text generation, so it can reference up-to-date facts.
- **Post-hoc fact checking:** Analyzing the model’s output with separate tools that highlight possible errors.
- **Instruction tuning:** Training the model to admit when it doesn’t know an answer or decline to respond when information is missing.

These help, but they don’t solve the problem completely. Even with access to external data or extra training, models still hallucinate—making up facts, joining information from separate contexts, or misunderstanding prompts.

Stopping hallucinations entirely isn’t possible with today’s LLM technology. Ultimately, LLMs rely on statistical patterns in language, not on any guaranteed connection to truth. Anyone using LLM outputs should verify the facts, especially if accuracy matters.

### Code Example: Seeing a Hallucination

Here’s how to observe a hallucination using Hugging Face’s Transformers library in Python. This code prompts a pre-trained LLM and shows its answer:

```python
from transformers import pipeline

# Load a pre-trained large language model ("gpt2" as an example)
generator = pipeline("text-generation", model="gpt2")

# Input: a factual question the model doesn't know
prompt = "Who was the first person to eat ice cream on the moon?"

# Generate a response
output = generator(prompt, max_new_tokens=50)
print(output[0]["generated_text"])
```

The model will usually return a confident answer—often mentioning an astronaut or a NASA mission. None of it is true. That’s a real example of an LLM hallucination, and a reminder to check the facts before assuming a model’s output is reliable.

---

## Key Takeaways

- LLM hallucinations are confident, plausible-sounding outputs that are untrue or fabricated.
- Models generate text based on statistical patterns, not real understanding or access to facts.
- Training data limitations and ambiguous prompts increase the likelihood of hallucinations.
- Techniques like retrieval augmentation and instruction tuning can reduce but not eliminate hallucinations.
- Human users must verify LLM outputs, especially when accuracy is important.

## Try It Yourself

Choose three factual questions you know the answers to and prompt an open-source LLM with each. Compare the model's responses to the actual answers, and note any hallucinations or inaccuracies. Try paraphrasing your prompts to see if the LLM's answers improve or change.

## Further Resources

- 🎥 [Why Language Models Hallucinate – Adam Kalai (IAS seminar video)](https://www.youtube.com/watch?v=0dRouBLcvMs)
- 📄 [Why Language Models Hallucinate – arXiv paper by Kalai et al.](https://arxiv.org/abs/2509.04664)
- 📄 [Understanding Why Language Models Hallucinate: Testing Reasoning Against Priors](https://arxiv.org/abs/2607.00447)
- 📄 [Why and How LLMs Hallucinate… Evidential Closure (EMNLP 2023)](https://aclanthology.org/2023.emnlp-main.192/)

---

**Coming up on Day 51:** Multimodal LLMs: Vision, Audio, and Beyond