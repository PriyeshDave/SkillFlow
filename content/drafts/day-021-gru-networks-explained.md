---
day: 21
generated_at: '2026-09-16T13:49:21.952747+00:00'
phase: Phase 3 — Sequence Models & Deep Learning for NLP
recap_summary: Explained the limitations of vanilla RNNs with sequence data due to
  the vanishing gradient problem, and introduced GRU (Gated Recurrent Unit) networks
  as an improved approach for handling memory in sequences by using update and reset
  gates to control information flow.
status: pending_review
title: 'Day 21: GRU Networks Explained'
topic_title: GRU Networks Explained
---

Sequence data shows up everywhere in natural language processing (NLP). Sentences are sequences of words. Time-stamped data—like sensor readings or stock prices—are sequences of numbers. To predict the next word in a sentence, translate languages, or detect sentiment, a model needs to “remember” what came earlier in the sequence. Ordinary neural networks, which treat each input independently, fail at this. They have no memory.

A typical fully connected (or *dense*) neural network expects input in a fixed shape—like a flattened image vector or a list of features. There’s no built-in idea of “earlier” or “later.” For sequences, this means losing the chain of context. In language, for example, “The cat sat on the...” could end with “mat,” “roof,” or “sofa.” The correct word depends on every word that came before.

## What is a Recurrent Neural Network (RNN)?

A recurrent neural network (RNN) handles sequences directly. Imagine reading a sentence one word at a time. With each new word, you update your understanding, carrying forward what you’ve seen so far. An RNN does the same by passing information forward through the sequence, one time step at a time.

At each step, an RNN takes the current input and mixes it with its “memory” from the previous step (called the *hidden state*). It then produces a new hidden state to pass to the next time step.

The hidden state works like a running notebook. Each new word or data point lets you update, erase, or reinforce what you’ve written. This lets RNNs, at least in theory, learn patterns that depend on previous elements of the sequence.

## The Vanishing Gradient Problem in Vanilla RNNs

Vanilla RNNs (the basic kind) have a big limitation: they struggle to learn relationships between distant parts of a sequence. If something from early in the data matters much later, the RNN often can't connect the dots.

This happens because of the *vanishing gradient problem*. When we train RNNs using a process called backpropagation through time, the “signal” that updates their parameters—the gradient—must pass backward through every time step. If the sequence is long, these gradients get multiplied by numbers slightly less than one, over and over. The effect: the gradient quickly shrinks toward zero, like a message fading in a long game of telephone. The network can’t learn connections that span many steps.

Picture a row of dominoes where each only gives a slight nudge. If you have a lot of dominoes, the force barely reaches the last one. Similarly, the signal in an RNN dies out, and the network “forgets” by the time it reaches a distant step.

## Gated Recurrent Unit (GRU): Smarter Memory with Gates

Researchers developed new RNN variants specifically to tackle these memory issues. The Gated Recurrent Unit (GRU) is a popular choice. It adds internal mechanisms, called *gates*, to control how information moves through the network.

Gates are like smart switches. At each time step, they decide which information to keep, which to update, and what to forget. This helps the RNN keep important facts from earlier and ignore irrelevant data.

With this setup, GRUs let models carry forward important information across many time steps. They work much better for tasks where context really matters.

## How a GRU Works: The Gates

A standard GRU cell uses two gates: the **update gate** and the **reset gate**.

- **Update gate**: Decides how much of the previous hidden state (memory) to keep. When the update gate is close to one, nearly all previous memory is kept. When it's close to zero, the cell mostly listens to the new input and forgets the past.
- **Reset gate**: Decides how much of the previous memory to ignore before combining with the new input. If the reset gate is zero, the cell forgets the past for that step. If it's one, it considers both the new input and the memory in full.

Here’s what happens inside a GRU cell at each time step:
1. The current input (like a word vector) and previous hidden state are fed into both gates.
2. Each gate uses a small neural net (with a sigmoid activation squashing outputs between 0 and 1) to decide its value.
3. The reset gate determines how much memory to blend with the new input to create a “candidate” new memory.
4. The update gate scales how much of that candidate memory replaces the old hidden state versus just passing the old memory forward.
5. The result is a new hidden state, carried to the next time step.

Gates give the network fine control: it can “remember” relevant dependencies for as many steps as it needs, and “forget” what's unimportant.

## Visualizing the GRU Cell

Picture a GRU cell as a box with two sliders—one for each gate.

- The previous hidden state and the current input come in.
- The reset gate filters the previous memory. This filtered memory and the input are used to propose a new “candidate” memory.
- The update gate's value sets how much of the old memory versus the candidate memory should form the next hidden state.
- The output hidden state is passed to the next time step.

If you sketched it, you’d see arrows for input and previous hidden state feeding into two “gate” blocks (update and reset), then combining to create a candidate hidden state, which merges with prior hidden state based on the update gate’s position.

Vanilla RNNs overwrite their memory at every step. The GRU’s careful switches keep crucial context alive and ignore distractions.

## GRU in Action: Code Example

Here's a minimal GRU in PyTorch. It runs a sequence of numbers through a single GRU cell so you can watch the hidden state evolve:

```python
import torch
import torch.nn as nn

# Make a GRU: input size 1, hidden size 1, single layer
gru = nn.GRU(input_size=1, hidden_size=1, batch_first=True)

# Sequence: batch size 1, sequence length 5, 1 feature per step
input_seq = torch.tensor([[[1.0], [2.0], [3.0], [4.0], [5.0]]])  # shape (1, 5, 1)

# Initial hidden state: zeros
h0 = torch.zeros(1, 1, 1)

# Forward pass
output_seq, hn = gru(input_seq, h0)

print("Output at each timestep:")
print(output_seq)
print("Final hidden state:")
print(hn)
```

Try changing the input sequence—or initial hidden state—and see how the hidden state tracks the “memory” of what’s happened so far.

## When and Why to Use GRUs

GRUs are a common choice for sequence problems where you need memory, but want something faster and simpler than an LSTM (another variant, covered next). In actual NLP tasks, GRUs excel at:
- Sentiment analysis (positive/negative sentence meaning)
- Language modeling (predicting the next word)
- Named entity recognition (finding names, places, dates)

GRUs have fewer gates than LSTMs, so they train faster and use fewer parameters. They’re strong when your sequences are of moderate length, or you want fast experimentation.

On the toughest tasks with really long-range dependencies, LSTMs can sometimes outperform GRUs. But for many practical problems, GRUs offer a great combination of speed and effectiveness. As a starting point for sequence models, they’re a strong default—much better than vanilla RNNs for real tasks.

You’ll learn about LSTMs next. For now, think of GRUs as a smart, efficient upgrade for learning from sequences—especially where basic RNNs would forget important context.

---

## Key Takeaways

- Ordinary neural networks cannot capture dependencies across sequence steps due to lack of memory.
- The vanishing gradient problem prevents vanilla RNNs from learning long-range dependencies.
- GRUs use update and reset gates to control what information to keep or forget from previous steps.
- GRUs are more efficient and train faster than LSTMs while effectively modeling moderate-length sequences.
- PyTorch provides an easy way to experiment with GRUs using simple code on numeric sequences.

## Try It Yourself

Using PyTorch, define a simple GRU layer with one input and hidden unit. Pass a sequence like [[0.1], [0.2], [0.3], [0.4]] through the GRU, printing the output at each timestep. Change one value in the input sequence and observe how the GRU's outputs change, illustrating how it retains memory of earlier inputs.

## Further Resources

- 🎥 [GRUs Explained Simply | Gated Recurrent Units in Deep Learning (AI for Beginners)](https://www.youtube.com/watch?v=t7Q5TvWhLls)
- 📘 [tf.keras.layers.GRU  |  TensorFlow v2.16.1](https://tensorflow.google.cn/api_docs/python/tf/keras/layers/GRU)
- 📄 [RNNs, LSTMs & GRUs – Complete NLP Series Part 7](https://www.wasilzafar.com/pages/series/nlp/nlp-rnn-lstm-gru.html)
- 📄 [Section 10.3: GRU Networks | Building Temporal AI](https://temporalbook.icsgen-ai.org/part-3-temporal-deep-learning/module-10-recurrent-neural-networks/section-10.3.html)
- 📄 [Gated Recurrent Units (GRU) – Stanford CS224n lecture slides](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1204/slides/cs224n-2020-lecture07-fancy-rnn.pdf)

---

**Coming up on Day 22:** Sequence-to-Sequence Models