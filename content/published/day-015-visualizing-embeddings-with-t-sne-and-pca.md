---
day: 15
generated_at: '2026-09-09T13:13:26.558724+00:00'
phase: Phase 2 — Word Representations & Embeddings
recap_summary: Explained why visualizing word embeddings is useful, introduced the
  challenges of high-dimensional data, and compared two popular dimensionality reduction
  techniques—PCA and t-SNE—for plotting embeddings, with code examples for practical
  visualization.
status: published
title: 'Day 15: Visualizing Embeddings with t-SNE and PCA'
topic_title: Visualizing Embeddings with t-SNE and PCA
---

**Previously, on Day 14:** Explained how word embeddings represent words as vectors, the limitations of classic approaches like one-hot encoding, and how FastText uses subword units to create robust embeddings that handle rare, new, or misspelled words.

---

### Why Visualize Embeddings?

Word embeddings are dense vectors that represent words as points in space. Imagine each word having a "position" in a space with maybe 100 or 300 directions—far beyond our normal three. Each number in the vector says how much the word lines up with one of these directions.

Words with similar meanings often end up close together in this space. For example, you might find "cat" and "dog" in roughly the same region, but far away from "car." By plotting these positions, you get a direct look at what the embedding has learned—what it thinks is similar, what feels distant, and which words cluster together.

Visualizing embeddings isn't just for curiosity. It can help spot patterns, catch mistakes, diagnose biases, and debug your model.

### The Challenge of High Dimensionality

Most people are comfortable thinking in two or three dimensions. Word embeddings use dozens or hundreds. If you try to plot 100 axes, you won’t see anything useful.

To make these embeddings visible, we use dimensionality reduction. This means taking high-dimensional data and squeezing it down to two or three dimensions in a way that tries to keep the important relationships between points.

### Principal Component Analysis (PCA): The Basics

Principal Component Analysis (PCA) is a classic way to reduce dimensions. It helps find the axes (directions) along which your data varies most.

Picture a swarm of points floating in 3D. You want to shine a light so that this cloud casts the biggest, flattest shadow onto a wall. PCA picks the best direction for that light—finding the axes with the most spread. In higher dimensions, it does the same, but finds the "shadow" in 2D or 3D that keeps as much of the structure as possible.

For embeddings, PCA projects all those dimensions down to just 2 or 3. It’s a linear technique, so it captures big movements or trends, but misses subtle, nonlinear patterns.

### t-SNE: Visualizing Similarities

t-Distributed Stochastic Neighbor Embedding (t-SNE) is another way to reduce dimensions. While PCA focuses on big-picture spread, t-SNE tries to make sure points that were close together in the original space stay close together in 2D.

Imagine t-SNE as reshuffling the points so that neighborhoods stay strong. If "cat" and "dog" were tight friends in the embedding, they’ll show up beside each other in the plot. t-SNE does this by matching up how similar pairs of points are in the high-dimensional space versus the lower-dimensional plot.

The tradeoff: t-SNE makes local groups clear, but you can’t trust the exact distances between distant groups. The pattern inside a cluster is meaningful; the gap between two clusters often isn’t.

### PCA vs. t-SNE: A Quick Comparison

- **PCA**: Fast, linear, good for spotting broad trends and outliers. If you want to compare distances and see big-picture structure, start with PCA.
- **t-SNE**: Finds tight clusters and preserves local relationships. Great for spotting groups, analogies, and patterns you might miss with PCA. But don’t use t-SNE to judge how far apart two separate clusters are.

Many people use both—PCA for the overview, t-SNE for finding groups inside.

### Visualizing Embeddings in Python: Step by Step

Let's use a tiny, hardcoded example. Real-world code would load hundreds or thousands of real vectors (say, from Word2Vec or GloVe), but it's easier to see what's happening with something simple.

Here’s code to shrink five 6-dimensional word embeddings down to 2D using both PCA and t-SNE, then plot and label the results:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Hardcoded 6-dimensional "embeddings" for 5 words
words = ["cat", "dog", "car", "bus", "apple"]
embeddings = np.array([
    [0.4, 0.2, 0.5, 0.1, 0.4, 0.2],  # cat
    [0.5, 0.1, 0.4, 0.2, 0.3, 0.2],  # dog
    [0.1, 0.6, 0.2, 0.8, 0.2, 0.3],  # car
    [0.2, 0.7, 0.1, 0.9, 0.1, 0.3],  # bus
    [0.8, 0.2, 0.9, 0.1, 0.7, 0.2],  # apple
])

# PCA to 2D
pca = PCA(n_components=2)
reduced_pca = pca.fit_transform(embeddings)

# t-SNE to 2D (small perplexity for small dataset)
tsne = TSNE(n_components=2, random_state=42, perplexity=2)
reduced_tsne = tsne.fit_transform(embeddings)

# Plotting
def plot_embeddings(X, title):
    plt.figure(figsize=(5, 4))
    for i, word in enumerate(words):
        x, y = X[i]
        plt.scatter(x, y)
        plt.text(x+0.01, y+0.01, word, fontsize=12)
    plt.title(title)
    plt.axis('off')
    plt.show()

plot_embeddings(reduced_pca, "PCA Visualization")
plot_embeddings(reduced_tsne, "t-SNE Visualization")
```

This plots each word, showing how PCA and t-SNE arrange them. Even with just five points, you’ll see "cat" and "dog" land near each other, and "car" and "bus" group up. "Apple," less related, stays apart. The two methods usually agree on clusters, but t-SNE's groupings look sharper.

If you use a larger dataset, t-SNE’s patterns will move around a bit between runs (unless you fix the random seed). For deeper experiments, try swapping in real pretrained embeddings—libraries like Gensim make this easy.

Visualizations like this are a first step to understanding your model’s internal map of words. For more precision, try probing embeddings with arithmetic or nearest-neighbor searches. But as a tool for sanity checks, bias hunting, or demoing what word embeddings capture, PCA and t-SNE are invaluable.

---

## Key Takeaways

- Word embeddings can be visualized to reveal similarities, clusters, and patterns.
- PCA reduces dimensions by capturing broad trends and is linear and fast.
- t-SNE preserves local relationships and reveals tight clusters but may distort global distances.
- Visualization helps debug, interpret, and check for biases in embeddings.
- Python code using sklearn makes it easy to plot word embeddings with PCA and t-SNE.

## Try It Yourself

Pick five words—some related, some unrelated (e.g., 'king', 'queen', 'man', 'woman', 'apple'). Use the provided code to plot their embeddings with both PCA and t-SNE. Examine the plots and write one sentence describing what you notice about how the words are grouped or separated.

## Further Resources

- 🎥 [4.9. Embedding Techniques Visual Comparison: Meaning Map with PCA and t‑SNE](https://www.youtube.com/watch?v=xpmVLDrCvc0)
- 📄 [Introduction to t‑SNE: Nonlinear Dimensionality Reduction and Data Visualization](https://www.datacamp.com/tutorial/introduction-t-sne)

---

**Coming up on Day 16:** Cosine Similarity and Semantic Search Basics