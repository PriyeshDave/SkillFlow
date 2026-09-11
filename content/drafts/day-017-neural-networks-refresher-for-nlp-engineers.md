---
day: 17
generated_at: '2026-09-11T13:05:24.364601+00:00'
phase: Phase 3 — Sequence Models & Deep Learning for NLP
recap_summary: Explained the structure and function of neural networks, including
  neurons, layers, weights, biases, and activation functions, and how neural networks
  are trained to solve natural language processing tasks.
status: pending_review
title: 'Day 17: Neural Networks Refresher for NLP Engineers'
topic_title: Neural Networks Refresher for NLP Engineers
---

## What Are Neural Networks?

A neural network is a way to model data by stacking layers of interconnected mathematical units called “neurons.” Unlike traditional algorithms that follow hand-written rules, neural networks learn patterns directly from example data.

In traditional programming, you write out logic: “If someone writes ‘great job,’ increase their rating by 1.” A neural network, instead, takes many labeled examples (such as messages marked positive or negative) and figures out, on its own, which patterns signal each label. This ability to learn directly from examples is what makes neural networks powerful, especially in cases where strict rules don’t capture all the subtleties of real data.

This is especially important in language tasks. Human language is subtle and context-dependent. Neural networks are good at picking up these nuances—patterns that are hard to describe with fixed rules but often appear in real conversations, reviews, or articles.

## A Simple Analogy: Neurons and Layers

A neuron in a neural network is a small processing unit, inspired by how brain cells (also called neurons) send and receive signals. In biology, a neuron collects signals from others, processes them, and decides what to send next.

A neural network neuron works similarly. It takes input numbers, multiplies each by a “weight” (a number showing its importance), adds them up, tweaks the sum with a “bias,” then passes the result through a (usually nonlinear) mathematical function.

Neurons are arranged into layers. The first layer receives input data; each neuron in this layer processes the inputs in parallel. Their outputs form the inputs to the next layer, and so on, until the final layer produces the network’s output.

Think of each layer as a step on a factory assembly line. Raw materials (the input data) are transformed at each station, until you end up with a finished product (the network’s output).

## How Neural Networks Process Data

Here’s what happens when you feed data into a basic neural network:

- **Input layer:** This is where the data enters, usually as a vector (list) of numbers. For text, each number might represent a feature like word count.
- **Weighted sum:** Each input is multiplied by a weight. If there are three inputs and four neurons in this layer, each neuron uses three separate weights.
- **Add bias:** Each neuron adds its own bias—an extra number it can adjust separately from the inputs.
- **Activation function:** The neuron passes the result through an activation function. This function helps the network deal with complex patterns, not just simple sums. Common activation functions include ReLU (rectified linear unit) and sigmoid.
- **Outputs:** The neurons’ outputs become the inputs to the next layer or, if this is the last layer, the network’s final output.

At every layer, the data is mixed, transformed, and prepared for the next step. This repeated transformation lets neural networks learn very complex relationships.

## Key Terms: Weights, Biases, and Activation Functions

Let’s define some essential terms:

**Weights** are the adjustable numbers that decide how much each input affects a neuron’s output. A big weight means the input is important. A weight of zero means the input has no effect.

**Biases** are numbers added to the output of each neuron. They allow the neuron to shift its output independently of its inputs, making the model more flexible.

**Activation functions** are math formulas applied after the weighted sum and bias. They let neural networks model complicated relationships, not just straight lines. Without them, no matter how many layers you use, the network can only handle linear patterns.

Two common activation functions:
- **ReLU (Rectified Linear Unit):** If the input is less than zero, returns 0; otherwise, returns the input value. It helps the network model “if/then” patterns.
- **Sigmoid:** Squeezes numbers into a range from 0 to 1, making it useful when you want the output to represent a probability.

## Learning by Example: Training a Neural Network

Neural networks learn through a process called training. Here’s a high-level view:

- The network sees many example pairs: input data and the correct output label.
- For each example, it makes a guess.
- We measure how far off the guess is. This is called the **loss**. The worse the guess, the higher the loss.
- The network then tweaks its weights and biases to try and reduce the loss next time. The process for figuring out which tweaks will help most is called **backpropagation**.
- This loop of prediction, loss calculation, and parameter adjustment repeats many times (over many “epochs”—full passes through the training data). Over time, the network gets better at predicting the correct output.

Don’t worry about the math for now. What matters is that neural networks gradually learn patterns in the data by repeatedly adjusting themselves based on feedback about their mistakes.

## Where Neural Networks Shine in NLP

Neural networks have revolutionized natural language processing (commonly abbreviated as NLP):

- **Sentiment analysis:** Decide if a review or comment is positive, negative, or neutral.
- **Machine translation:** Translate text from one language to another automatically.
- **Text summarization:** Create brief, accurate versions of longer texts.
- **Named entity recognition:** Detect and tag names of people, places, or organizations in text.
- **Speech recognition:** Turn spoken audio into written text.
- **Question answering:** Read passages and answer questions about them.

Neural networks excel at all these tasks because they learn patterns directly from data—patterns too complex to be captured by simple rules.

---

Below is a minimal code example using PyTorch, a popular deep learning library. This code builds a one-layer (single linear layer) neural network that learns to map three input features to a single output. The example uses random data, so you can run it anywhere. This shows the core mechanics, not a production system.

```python
import torch
import torch.nn as nn

# Random data: 100 examples, each with 3 features
X = torch.randn(100, 3)
# Random "labels" (what the network should learn to predict)
y = torch.randn(100, 1)

# Define a simple single-layer neural network
model = nn.Linear(3, 1)

# Mean squared error loss function
loss_fn = nn.MSELoss()

# Stochastic Gradient Descent optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(100):
    # Forward pass: compute prediction
    y_pred = model(X)
    # Compute loss
    loss = loss_fn(y_pred, y)
    # Zero gradients, backward pass, and update weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch+1) % 20 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

# After training, model(X) gives outputs closer to y
```

This toy example walks through the training loop: predict, measure error, adjust, and repeat. For larger datasets and more complex models, the main idea stays the same.

---

## Key Takeaways

- Neural networks learn patterns from data rather than following fixed rules.
- Neurons process inputs using weights, biases, and activation functions to model complex relationships.
- Activation functions like ReLU and Sigmoid allow networks to capture nonlinear patterns.
- Neural networks improve through repeated prediction, error measurement, and parameter adjustment (training).
- Neural networks are highly effective for NLP tasks such as sentiment analysis and machine translation.

## Try It Yourself

Modify the provided PyTorch code to change the activation function in the network (e.g., use Sigmoid instead of no activation). Run the code, observe how the outputs differ before and after training, and write one sentence comparing the behaviors with the two activation functions.

## Further Resources

- 🎥 [Stanford CS224N Lecture 5 – Recurrent Neural Networks (RNNs)](https://www.youtube.com/watch?v=PLryWeHPcBs)
- 🎥 [Neural Nets for NLP 2021 – Recurrent Neural Networks (CMU, Graham Neubig)](https://www.classcentral.com/course/youtube-cmu-neural-nets-for-nlp-2021-recurrent-neural-networks-146004)
- 📘 [NLP From Scratch: Translation with a Sequence to Sequence Network and Attention (PyTorch Tutorial)](https://docs.pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)
- 📘 [Sequence Models – Dive into Deep Learning (Chapter 8)](https://d2l.smola.org/chapter_recurrent-neural-networks/)
- 📄 [Recurrent Neural Networks and Long Short‑Term Memory Networks: Tutorial and Survey](https://arxiv.org/abs/2304.11461)

---

**Coming up on Day 18:** Recurrent Neural Networks (RNNs) Explained