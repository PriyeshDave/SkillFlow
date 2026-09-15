---
day: 19
generated_at: '2026-09-14T15:20:31.506242+00:00'
phase: Phase 3 — Sequence Models & Deep Learning for NLP
recap_summary: Explained the vanishing gradient problem in deep neural networks, including
  how gradients shrink as they move backward through layers—especially with activation
  functions like sigmoid or tanh—and why this hampers learning in deep and sequence
  models such as RNNs.
status: pending_review
title: 'Day 19: The Vanishing Gradient Problem'
topic_title: The Vanishing Gradient Problem
---

## What Is the Vanishing Gradient Problem?

The vanishing gradient problem makes it difficult to train very deep neural networks. It happens when the “gradient”—the math signal that tells each layer how to improve—gets smaller and smaller as it moves backward through the network. This especially affects models for sequence data, like Recurrent Neural Networks (RNNs).

A **gradient** is a number measuring how much changing a model parameter (like a weight) will change the model’s error. Large gradients mean weights should change a lot. Small gradients mean they barely change at all.

When these gradients become very small, early layers in the network stop learning. The whole model struggles to improve.

## Why Gradients Matter in Deep Learning

Gradients are how neural networks learn. Training works by making predictions, measuring how wrong they are, and adjusting the weights to make things less wrong next time.

Practically, you run some data through the network, compute an error, and use calculus (via derivatives) to find the gradient. **Backpropagation** is the algorithm that efficiently computes these gradients backward through the network, from the output layer back to the first layer.

You can think of each weight in the network as a dial you can turn. The gradient tells you which way, and how much, to turn that dial to reduce error. If the gradient is almost zero, the dial barely moves—no learning happens.

## How Vanishing Gradients Happen

Deep networks chain together multiple layers. Each layer passes its outputs through an **activation function**. An activation function is just a small math operation (like sigmoid or tanh) that helps the network model complex relationships. Functions like sigmoid and tanh “squash” their inputs into a narrow range—sigmoid to (0, 1); tanh to (-1, 1).

During backpropagation, gradients flow backward through these activation functions. At each step, the gradient gets multiplied by the slope (derivative) of the activation function at that point. For sigmoid or tanh, this multiplying factor is always less than 1.

Imagine every layer in a network multiplies the gradient by 0.4 (a typical value for tanh or sigmoid). If you have 5 layers:

- After 1 layer: original × 0.4
- After 2 layers: original × 0.16
- After 5 layers: original × 0.4⁵ ≈ original × 0.01

By the time the gradient reaches the earliest layers, it's nearly zero.

**Analogy:** Imagine whispering a message down a long line of people. If each person hears only half as loud as the last, the message fades to almost nothing before reaching the end. This is how gradients can fade in deep networks.

## Consequences in Sequence Models

Sequence models like RNNs process data that stretches out in time—like words in a sentence or steps in a time series. RNNs use the same set of weights, over and over, for each time step.

With vanishing gradients, learning signals must travel backward through many time steps. The longer the sequence and the more steps, the greater the problem.

If your RNN needs to predict a word at the sentence end based on a word from the beginning (maybe 20 steps ago), the gradient must travel back through all those steps. If it shrinks at every step, it will be nearly zero by the time it reaches the start. As a result, the network almost can’t learn to connect early and late information. Early RNNs struggled to model long-range relationships for this reason.

## Visualizing the Effect

Here’s a concrete math example. Suppose you have 6 layers, each using tanh. Tanh’s maximum slope (derivative) is about 1, but typical values are smaller—roughly 0.4 to 0.5 for most inputs.

If, at each layer, the gradient is multiplied by 0.5:

- Start: 1.0
- After 1 layer: 0.5
- After 2 layers: 0.25
- After 3 layers: 0.125
- After 4 layers: 0.0625
- After 5 layers: 0.03125
- After 6 layers: 0.015625

At layer 6, less than 2% of the original gradient remains. The optimizer can barely adjust these early weights.

## PyTorch Example: Gradients Shrinking in Deep Tanh Networks

Below is a runnable code example. This builds a deep feedforward neural network with tanh activations. After doing one forward and backward pass on random data, it prints the average absolute gradient for each layer. You’ll see that earlier layers (closer to the input) have much smaller gradients than later layers (closer to the output).

```python
import torch
import torch.nn as nn

# Deep network: 7 hidden layers, all tanh
class DeepTanhNet(nn.Module):
    def __init__(self, dim, depth):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.append(nn.Linear(dim, dim))
            layers.append(nn.Tanh())
        self.seq = nn.Sequential(*layers)
        self.out = nn.Linear(dim, 1)
    def forward(self, x):
        return self.out(self.seq(x))

depth = 7
dim = 8
net = DeepTanhNet(dim, depth)

x = torch.randn(1, dim)
target = torch.tensor([[0.0]])

out = net(x)
loss = (out - target).pow(2).mean()
loss.backward()

# Print average absolute gradient for each Linear layer
for i, layer in enumerate(net.seq):
    if isinstance(layer, nn.Linear):
        grad = layer.weight.grad.abs().mean().item()
        print(f"Layer {i//2 + 1} mean abs gradient: {grad:.6f}")
```

The printed gradients will be much smaller for early layers than for those near the output. This is the vanishing gradient in action.

## Common Ways to Address the Vanishing Gradient Problem

Several well-established tools help with vanishing gradients:

- **ReLU Activations:** The Rectified Linear Unit (ReLU) activation doesn’t squash values for positive inputs, so gradients are less likely to vanish.
- **LSTM and GRU Units:** In RNNs, specially designed units like Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) let gradients flow more easily across many time steps.
- **Better Weight Initialization:** Initializing network weights in a smart way can keep gradients in a manageable range.
- **Batch Normalization:** Batch normalization helps keep values stable as they flow through the network, indirectly preventing gradients from shrinking too much.

Each of these approaches has trade-offs. All will be covered in more detail later in this series. For now, remember: vanishing gradients are a key challenge for deep and sequential neural networks, but there are proven solutions.

---

## Key Takeaways

- Gradients indicate how much model weights should change to reduce error.
- Vanishing gradients occur when gradients become extremely small in early layers of deep networks.
- Activation functions like sigmoid and tanh contribute to gradient shrinking through repeated multiplication by small derivatives.
- This issue makes it hard for models, especially RNNs, to learn long-range dependencies.
- Techniques such as ReLU activations, LSTM/GRU units, batch normalization, and better weight initialization help mitigate vanishing gradients.

## Try It Yourself

Build a simple multi-layer perceptron (MLP) with several hidden layers, using tanh activations, in your favorite deep learning framework. Pass some random data through the network and perform a backward pass. Print out the average absolute gradients for each layer and observe how the gradients are much smaller in earlier layers compared to later ones.

## Further Resources

- 🎥 [Stanford CS224N Lecture 7 – Vanishing Gradients, Fancy RNNs](https://www.youtube.com/watch?v=QEw0qEa0E50)
- 📄 [VizLearn – Vanishing Gradient Problem in RNN](https://vizlearn.in/natural_language_processing/vanishing_gradient_problem_in_rnn.html)
- 🎥 [MIT Introduction to Deep Learning – Sequence Modeling with Neural Networks](https://www.classcentral.com/course/youtube-mit-6-s191-2018-sequence-modeling-with-neural-networks-128115)
- 📄 [Michael Brenndoerfer – Vanishing Gradients: Why RNNs Fail on Long Sequences](https://mbrenndoerfer.com/writing/vanishing-gradients-rnn-long-range-dependencies)
- 📄 [Hu et al., Overcoming the vanishing gradient problem in plain recurrent networks (arXiv)](https://arxiv.org/abs/1801.06105)

---

**Coming up on Day 20:** LSTM Networks Explained