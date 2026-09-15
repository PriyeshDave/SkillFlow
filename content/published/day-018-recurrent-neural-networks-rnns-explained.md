---
day: 18
generated_at: '2026-09-14T14:56:06.899064+00:00'
phase: Phase 3 — Sequence Models & Deep Learning for NLP
recap_summary: Explained why sequential dependencies are critical in language tasks
  and introduced Recurrent Neural Networks (RNNs), describing how they maintain and
  update a hidden state to capture sequential context. Demonstrated the RNN cell's
  core computation, illustrated step-by-step processing in code, and discussed the
  limitations of vanilla RNNs, motivating the development of improved architectures
  like LSTMs and GRUs.
status: published
title: 'Day 18: Recurrent Neural Networks (RNNs) Explained'
topic_title: Recurrent Neural Networks (RNNs) Explained
---

**Previously, on Day 17:** Explained the structure and function of neural networks, including neurons, layers, weights, biases, and activation functions, and how neural networks are trained to solve natural language processing tasks.

---

### The Challenge with Sequences in NLP

Human language is structured as a sequence. The meaning of a sentence depends on the order of its words:

- “I ate breakfast before running” is not the same as “I ran before eating breakfast.”

Feedforward neural networks, which we've seen so far, treat each input as independent from the others. These models can classify an image or predict a single label, but they have no way to remember earlier inputs. If you process a sentence word by word, a feedforward network processes each word as if it's unrelated to what came before.

This becomes a serious limitation for tasks like machine translation, speech recognition, or conversational context. In language, the order of words and the context from earlier words are essential. We need models that can remember what has already happened and use that memory to interpret each new word.

### Introducing Recurrent Neural Networks (RNNs)

A Recurrent Neural Network (RNN) is designed for sequential data. RNNs process data one element at a time—think one word at a time—while carrying a short-term memory, called the **hidden state**, from one step to the next.

Unlike feedforward networks, each step in an RNN passes information forward through this hidden state. The network can "remember" what it has seen and use that memory when processing new inputs.

This memory makes RNNs a much better fit for sequences like sentences, where the meaning of a word depends on what came before.

### How RNNs Work: Unrolling in Time

Imagine reading a sentence one word at a time from left to right. You don't reset your memory at each word; you remember what you’ve read so far. RNNs are built with this same idea.

To make training possible, we _unroll_ the RNN in time:
- For an input sequence of five words, imagine five copies of the RNN cell—one for each word in the sequence.
- Each cell shares the same parameters (weights), but receives its own input and the hidden state from the previous step.

At each step, the RNN cell:
1. Takes in the current input (e.g., the current word’s encoding).
2. Receives the summary, or **hidden state**, from the preceding step.
3. Updates its hidden state to reflect both the current input and past context.
4. Optionally, produces an output (such as a predicted word or tag).

This chained process lets information—the hidden state—move forward through the sequence and capture context up to the current position.

### The RNN Cell: What Happens at Each Step

Inside an RNN cell, the computation is similar to a layer in a regular neural network, but with one key difference: it takes two inputs at each timestep—
- The current input (such as the vector representing the current word),
- And the previous hidden state (the memory of everything seen so far).

The core update at each time step \( t \) is:

\[
h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)
\]

Where:
- \( x_t \) is the current input at time \( t \)
- \( h_{t-1} \) is the hidden state from the previous timestep
- \( W_{xh} \) is the weight matrix for input-to-hidden connections
- \( W_{hh} \) is the weight matrix for hidden-to-hidden (memory) connections
- \( b_h \) is a bias term
- \( \tanh \) is an activation function, producing outputs between -1 and 1

The new hidden state \( h_t \) carries information from both the new input and everything summarized up to the previous step.

### RNNs in Action: A Simple Code Example

Here's a minimal PyTorch example showing how an RNN processes a sequence. This code:
- Defines a single-layer RNN cell,
- Feeds in a sequence of numbers (1 to 5),
- Shows the hidden state after each input.

```python
import torch
import torch.nn as nn

# Example sequence: batch size 1, sequence length 5, input size 1
sequence = torch.tensor([[[1.0]], [[2.0]], [[3.0]], [[4.0]], [[5.0]]])

input_size = 1    # Input dimension at each step
hidden_size = 2   # Hidden state size

rnn = nn.RNN(input_size, hidden_size, batch_first=False)

# Initialize hidden state: (num_layers, batch_size, hidden_size)
h_t = torch.zeros(1, 1, hidden_size)

print("Step | Input | Hidden State")
print("-------------------------")

for i, x_t in enumerate(sequence):
    out, h_t = rnn(x_t.unsqueeze(0), h_t)  # Add batch/time axes
    print(f"  {i+1}  |  {x_t.item()}   |  {h_t.squeeze(0).detach().numpy()}")

```

At each step, the RNN's hidden state reflects not just the current number, but also what it's seen so far. This is what gives the RNN its memory of the sequence.

### RNN Limitations and the Path Forward

Basic, or "vanilla," RNNs can remember context for short sequences, but they struggle with longer ones.

The challenge comes from training. As the RNN updates its weights, important signals from early in the sequence can quickly shrink to near zero (vanish) or grow out of control (explode). This is called the **vanishing or exploding gradient problem**. In practice, vanilla RNNs often forget information from more than a few steps back.

To fix this, researchers developed enhanced RNNs, such as **LSTMs** (Long Short-Term Memory networks) and **GRUs** (Gated Recurrent Units). These are specially designed to remember important information for much longer, allowing models to capture context even when it appears many words back in a sentence or conversation.

Understanding basic RNNs gives you the foundation to appreciate why these more advanced architectures are needed—and how they power real-world NLP systems, where meaning can depend on something far earlier in the sequence.

---

## Key Takeaways

- Language understanding requires remembering word order and prior context.
- RNNs keep a hidden state to maintain memory across sequence steps.
- At each step, an RNN combines current input with previous hidden state.
- Vanilla RNNs struggle to remember information in long sequences due to vanishing/exploding gradients.
- Enhanced architectures like LSTMs and GRUs help RNNs capture longer-range dependencies.

## Try It Yourself

Choose a short word (such as 'cat') and represent each character as a simple number (e.g., c=1, a=2, t=3). Using the update rule h_t = tanh(0.5 * x_t + 0.8 * h_{t-1}) with h_0 = 0, manually compute and record the hidden state for each character in order. Write down the value of h after processing each character to observe how the hidden state accumulates information.

## Further Resources

- 🎥 [Recurrent Neural Networks (RNNs), Clearly Explained!!! – StatQuest with Josh Starmer](https://www.youtube.com/watch?v=AsNTP8Kwu80)
- 🎥 [Recurrent Neural Networks - EXPLAINED! – CodeEmporium](https://www.youtube.com/watch?v=yZv_yRgOvMg)
- 📘 [Sequence Models and Long Short‑Term Memory Networks — PyTorch Tutorials](https://docs.pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html)
- 📄 [Recurrent Neural Networks and Long Short‑Term Memory Networks: Tutorial and Survey](https://arxiv.org/abs/2304.11461)
- 📄 [LLMs Part 2: Building a Vanilla RNN](https://olliegreen.info/posts/7/llms-part-2-building-a-vanilla-rnn/)

---

**Coming up on Day 19:** The Vanishing Gradient Problem