---
day: 23
generated_at: '2026-09-18T13:13:51.320494+00:00'
phase: Phase 3 — Sequence Models & Deep Learning for NLP
recap_summary: Explained the encoder-decoder architecture in neural networks, detailing
  how inputs are encoded into a context vector and then decoded to generate outputs
  of possibly differing lengths and formats, with applications including translation
  and summarization.
status: pending_review
title: 'Day 23: The Encoder-Decoder Architecture'
topic_title: The Encoder-Decoder Architecture
---

## What Is the Encoder-Decoder Architecture?

The encoder-decoder architecture is a neural network pattern for handling tasks where input and output can have different lengths. This setup uses two separate neural networks.

The **encoder** reads the input, such as a full sentence. Its job is to absorb and compress all the information from the input into a single, fixed-size vector. This vector is called the **context vector** or **thought vector**. Think of it like reading a paragraph and distilling its whole meaning into a dense summary.

The **decoder** uses this context vector to generate the output. The output could be a translation, a summary, a reply, or any sequence that may be a different length or in a different format from the input.

This design allows neural networks to handle tasks where input and output aren’t the same length or even the same “type.”

## Why Do We Need Encoder-Decoder Models?

Most classic neural networks handle only fixed-size inputs and outputs. But many real tasks, especially in natural language processing (NLP), need more flexibility.

Machine translation is a good example. Translating “How are you?” (3 words) into French gives “Comment ça va ?” (4 words). The output doesn’t always match the input in length or word order.

Trying to map input to output one word at a time misses context and grammar. Some translations need the full sense of the sentence before choosing the right words. Encoder-decoder models solve this by letting the network process the whole input before producing any output.

The encoder-decoder architecture is made for:
- Handling variable-length inputs and outputs.
- Generating outputs that may rearrange or rephrase the input.
- Tasks like translation, summarization, dialogue, even generating captions for images (with the right kind of encoder).

## The Encoder: Turning Input into Context

The encoder starts with a sequence of input tokens, such as words in a sentence. It processes these tokens one by one, updating an internal state as it goes. This state is usually managed by a recurrent neural network (RNN), which “remembers” what it has seen so far.

When the encoder finishes reading the entire input sequence, the final value of its internal state becomes the **context vector**. This vector is a compressed representation of the whole input.

For example, encoding “hello world”:
- The encoder reads “hello” and updates its state.
- It reads “world” and updates again.
- The final state after “world” is the context vector summarizing “hello world.”

The encoder does not know yet what the output will be. Its job is only to produce the best possible summary of the input.

## The Decoder: Producing Meaningful Output

The decoder is another neural network, usually another RNN. It starts with the context vector from the encoder. At each step, it generates one output token.

The decoder uses its own internal state, plus the context from the encoder, and often the previous output it produced. With this information, it chooses the next output token.

For English-to-French translation, for example:
- The decoder starts with the context vector from the encoder.
- At the first step, it predicts the first French word.
- At each following step, it uses the context and previous outputs to predict the next word.
- It stops when it produces a special “end” token.

This process allows the decoder to produce outputs of any length, only stopping when the task is finished.

## Example: Translating a Simple Sentence

Let’s see how this works when translating “cat sleeps” into French: “le chat dort.”

1. **Encoding:**
   - Input sequence: [“cat”, “sleeps”].
   - The encoder processes “cat” (updates its internal state).
   - It processes “sleeps” (updates again).
   - The last encoder state is the context vector.

2. **Decoding:**
   - The decoder starts with the context vector.
   - It produces the first output, “le.”
   - Using the context and “le,” it outputs “chat.”
   - It outputs “dort.”
   - Finally, it produces the end-of-sentence token (“<eos>”).

At each step, the decoder’s output draws from both its own state and the original context. This helps ensure the translation stays grammatical and meaningful.

## Minimal Working Example in Code

Let's use a simple problem: reversing a sequence of numbers. For input `[1, 2, 3]`, the output should be `[3, 2, 1]`. This demonstrates every part of the encoder-decoder setup.

We’ll use PyTorch for this example. The code is minimal and focused on the idea.

```python
import torch
import torch.nn as nn

# Parameters
INPUT_SIZE = 10   # Numbers 0–9
HIDDEN_SIZE = 16
OUTPUT_SIZE = 10  # Same as input
SEQ_LEN = 3

# Encoder: RNN that reads the input
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(INPUT_SIZE, HIDDEN_SIZE)
        self.rnn = nn.GRU(HIDDEN_SIZE, HIDDEN_SIZE)
    def forward(self, input_seq):
        embedded = self.embedding(input_seq).unsqueeze(1)
        outputs, hidden = self.rnn(embedded)
        return hidden

# Decoder: RNN that writes the output, one token at a time
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(OUTPUT_SIZE, HIDDEN_SIZE)
        self.rnn = nn.GRU(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)
    def forward(self, input_token, hidden):
        embedded = self.embedding(input_token).unsqueeze(0)
        output, hidden = self.rnn(embedded, hidden)
        logits = self.fc(output.squeeze(0))
        return logits, hidden

# Example: run once (no training)
encoder = Encoder()
decoder = Decoder()

input_seq = torch.tensor([1, 2, 3], dtype=torch.long)
target_seq = torch.tensor([3, 2, 1], dtype=torch.long)

# Encode
context = encoder(input_seq)

# Decode step by step
decoded_tokens = []
decoder_input = torch.tensor([0])  # Start token (use 0 here)
hidden = context

for i in range(SEQ_LEN):
    logits, hidden = decoder(decoder_input, hidden)
    prediction = logits.argmax(1)
    decoded_tokens.append(prediction.item())
    decoder_input = prediction

print("Predicted output:", decoded_tokens)
```

This code uses two small RNNs: one encodes the input sequence into a context vector, and the other decodes that context to produce the output sequence. There’s no training here, so the outputs will be random, but it shows how information flows from encoder to decoder.

## Limitations and What's Next

The classic encoder-decoder model squeezes all the input into a single fixed-length context vector. For short or simple sentences, this works. But as the input grows longer or more complex, it’s easy for important details to vanish. This “bottleneck” limits the model’s performance.

Modern architectures address this by letting the decoder look back at the whole input using mechanisms like "attention." The encoder-decoder idea is the foundation. From here, we'll see how to overcome its biggest limits and handle even more challenging tasks.

---

## Key Takeaways

- Encoder-decoder architecture handles variable-length input and output sequences.
- The encoder compresses the input into a fixed-size context vector.
- The decoder generates the output sequence using the context vector and previous outputs.
- This pattern enables flexible tasks like translation, summarization, and dialogue.
- A core limitation is information loss when input sequences are long or complex.

## Try It Yourself

Write a function that takes a list of numbers (such as [3, 1, 4]) and uses a pre-defined encoder-decoder model to reverse the sequence. Run your function with several different input lists to observe how the encoder compresses the input and the decoder reconstructs the output. Reflect on how the architecture changes with different sequence lengths.

## Further Resources

- 🎥 [Encoder‑decoder architecture: Overview (Google Cloud Tech YouTube video)](https://www.youtube.com/watch?v=zbdong_h-x4)
- 🎥 [NLP ‑ 11: Encoder‑Decoder Model (AI & ML with Sanjay Chouhan YouTube video)](https://www.youtube.com/watch?v=VwwOuuFCqJM)
- 📄 [Dive into Deep Learning – 18.1 Encoder‑Decoder Models for Sequence Transduction](https://d2l.smola.org/chapter_natural-language-processing-pretraining/seq2seq.html)
- 📘 [Dive into Deep Learning documentation – 10.6 The Encoder‑Decoder Architecture](https://d2l.ai/chapter_recurrent-modern/encoder-decoder.html)
- 📄 [Hugging Face blog – Transformers‑based Encoder‑Decoder Models](https://huggingface.co/blog/encoder-decoder)

---

**Coming up on Day 24:** Attention Mechanism: The Core Intuition