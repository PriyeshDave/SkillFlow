---
day: 27
generated_at: '2026-09-24T13:53:01.507688+00:00'
phase: Phase 4 — The Transformer Revolution
recap_summary: Explained the limitations of traditional sequence models and introduced
  self-attention as a solution for capturing long-range relationships in sequences.
  Detailed the self-attention mechanism using queries, keys, and values, and worked
  through an explicit step-by-step numerical example to demonstrate how self-attention
  creates context-aware word representations.
status: pending_review
title: 'Day 27: Self-Attention Explained From Scratch'
topic_title: Self-Attention Explained From Scratch
---

## What Problem Does Self-Attention Solve?

Understanding sequences, like sentences, depends on how words relate to each other. In language, a word's meaning can depend on words that come much earlier or later in the sentence.

Traditional sequence models, such as RNNs (Recurrent Neural Networks), process sequences one word at a time. Each step updates a memory that carries information forward. Passing information from the start of a sentence to the end requires many steps, which makes it easy to lose details—especially in long sentences.

CNNs (Convolutional Neural Networks) look at groups of nearby words to spot patterns. But connecting information between distant words, like the beginning and end of a sentence, takes stacking many CNN layers. This quickly becomes inefficient.

Both of these approaches struggle when long-range relationships matter. For instance, in the sentence “The animal, which the farmer chased, ran away,” the subject “animal” pairs with the verb “ran away,” but they're separated by several words. We need a model that gives every word direct access to all others, so it can use whatever context matters, even across long distances.

Self-attention solves this problem. It lets every word see and weigh all the other words in the sequence simultaneously.

## What Is Self-Attention (in Plain Terms)?

Self-attention is a process where each word looks at all other words in the sequence to decide which are important for its own representation. Instead of marching step by step, every word gets information from everywhere else, all at once.

Think of reading a sentence. For each word, you glance at the rest of the sentence to check what changes its meaning. Some words might barely matter, others might be essential.

Analogy: Imagine a group meeting. Each person (word) listens to everyone else before forming their opinion. You might care a lot about what one person says, and barely note another. The weight you give to each speaker shapes your final thought.

## The Ingredients of Self-Attention: Queries, Keys, and Values

Self-attention uses three parts for every input word: a **query**, a **key**, and a **value**. These are just different views of the same word, computed as simple linear projections.

- **Query:** What this word wants to find out about its context. It asks, “Which other words could change my meaning?”
- **Key:** What this word offers as information. It describes each word's own characteristics.
- **Value:** The actual content this word provides, if chosen as relevant.

Let’s use an example:  
*"The hungry cat chased the mouse."*

When focusing on “cat”:
- The query for “cat” is like asking, “What do I need to know from this sentence?”
- Each other word, including “cat” itself, has a key (“I’m a state word,” “I’m an animal,” “I’m an action,” etc.) and a value (the information it holds).
- “Cat" will attend strongly to “hungry” (its state) and “chased” (its action), based on how well its query matches their keys.
- “Chased" will attend to “cat” and “mouse.”

The query, key, and value for each word are just different mathematical transformations (matrix multiplications) of its word embedding—the basic numeric representation of a word.

## How Self-Attention Works Step by Step

Let's walk through a concrete example using tiny vectors, so you can see every math step.

Suppose our vocabulary consists of just three words: A, B, and C. Each word is represented as a 2-component vector.

Let’s define:
- A = [1, 0]
- B = [0, 1]
- C = [1, 1]

Our sequence is [A, B, C].

To keep things simple, we’ll set the query, key, and value for each word equal to the input vector. (Normally, you’d multiply by separate matrices. Here, we use the identity matrix, meaning nothing changes.)

### Step 1: Create Query, Key, and Value Matrices

```
Q = [[1, 0],   # A’s query
     [0, 1],   # B’s query
     [1, 1]]   # C’s query

K = [[1, 0],   # A’s key
     [0, 1],   # B’s key
     [1, 1]]   # C’s key

V = [[1, 0],   # A’s value
     [0, 1],   # B’s value
     [1, 1]]   # C’s value
```

### Step 2: Compute Attention Scores

For each word, calculate the dot product between its query and the keys of every word. This gives a score for how much each word might matter.

For A:
- A’s query · A’s key: [1,0]·[1,0] = 1
- A’s query · B’s key: [1,0]·[0,1] = 0
- A’s query · C’s key: [1,0]·[1,1] = 1

A’s attention scores: [1, 0, 1]

Repeat for B and C:
- B: [0,1]·[1,0]=0, [0,1]·[0,1]=1, [0,1]·[1,1]=1  → [0, 1, 1]
- C: [1,1]·[1,0]=1, [1,1]·[0,1]=1, [1,1]·[1,1]=2  → [1, 1, 2]

### Step 3: Normalize Scores with Softmax

Softmax is a math function that converts scores to positive numbers that sum to 1. This turns the raw scores into attention weights.

For A: softmax([1, 0, 1])

First, exponentiate: exp(1)=2.718, exp(0)=1  
Sum = 2.718 + 1 + 2.718 = 6.436

Weights:
- [2.718/6.436, 1/6.436, 2.718/6.436] = [0.42, 0.16, 0.42]

So A pays 42% attention to itself, 16% to B, 42% to C.

### Step 4: Weighted Sum of Values

Each word calculates its new representation as a weighted sum of all value vectors (one for each word), using its attention weights.

For A:
- Output = 0.42 × [1, 0]  +  0.16 × [0, 1]  +  0.42 × [1, 1]
-         = [0.42+0.42, 0.16+0.42]
-         = [0.84, 0.58]

Repeat this for B and C to get their new context-aware representations.

Now, every output vector contains a blend of information from the whole sequence, mixed according to what each word considers important.

### Minimal Numpy Example

```python
import numpy as np

# Step 1: inputs (3 words, each as 2D vector)
X = np.array([
    [1., 0.],  # Word A
    [0., 1.],  # Word B
    [1., 1.]   # Word C
])

# Step 2: queries, keys, values (use identity - no change)
Q = X
K = X
V = X

# Step 3: attention scores (dot product Q and K.T)
scores = Q @ K.T

# Step 4: softmax (row-wise)
def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

weights = softmax(scores)

# Step 5: attention output (weights @ V)
output = weights @ V

print("Attention output vectors for A, B, C:")
print(output)
```

This code prints the new, context-enriched vector for each word after self-attention. Each result blends information from every word and shows how the output shifts based on what each word attended to.

## Why Self-Attention Is Powerful

Self-attention lets every word immediately access and weigh all other words in the sequence. It doesn’t need to pass memory along step by step, like RNNs. It doesn’t have to rely on local windows, like CNNs.

This has big effects:
- **Parallelization:** All word-to-word calculations can run at once, making training fast with modern GPUs and TPUs.
- **Long-Range Context:** Words at the start and end of a sequence can influence each other directly, in a single operation.
- **Task-Specific Weights:** For each word, the model learns which other words matter most for what it needs to do. There’s no fixed rule or window size.

These qualities made self-attention—especially as used in the Transformer model—a foundation for modern language and sequence models. It enables models to capture relationships in data more flexibly and effectively than previous approaches.

---

## Key Takeaways

- Traditional RNNs and CNNs struggle to connect distant words in sequences.
- Self-attention allows every word to directly consider all others in a sentence or sequence.
- Self-attention uses queries, keys, and values derived from word embeddings.
- The mechanism involves matching queries to keys, normalizing with softmax, and combining values accordingly.
- Self-attention enables efficient parallelization and captures long-range dependencies critical for models like Transformers.

## Try It Yourself

Try implementing a simple self-attention calculation for three word embeddings: [1,0], [0,1], and [1,1]. Assign each as the query, key, and value for the corresponding word. For each word, compute dot products with all keys to get similarity scores, apply softmax to these scores, and then use the resulting weights to take a weighted average of the value vectors. Do this by hand or in code, and compare the computed outputs for each word.

## Further Resources

- 🎥 [Attention in transformers, step‑by‑step | 3Blue1Brown Deep Learning Chapter 6](https://www.youtube.com/watch?v=eMlx5fFNoYc)
- 📄 [Self‑Attention from Scratch — Tai Bui](https://taibui.dev/phases/07-transformers-deep-dive/02-self-attention-from-scratch)
- 📄 [Self‑attention from scratch (The Loss Curve)](https://thelosscurve.com/guides/attention-from-scratch)
- 📄 [Transformers from Scratch | Sovesh Mohapatra](https://soveshmohapatra.com/projects/transformers/)

---

**Coming up on Day 28:** Multi-Head Attention