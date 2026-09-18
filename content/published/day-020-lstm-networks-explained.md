---
day: 20
generated_at: '2026-09-15T13:56:07.473877+00:00'
phase: Phase 3 — Sequence Models & Deep Learning for NLP
recap_summary: Explained the limitations of simple RNNs with long-term dependencies
  in sequence data, introduced LSTM cells and their gating mechanisms, and demonstrated
  how LSTMs solve the memory retention issue in tasks like text prediction.
status: published
title: 'Day 20: LSTM Networks Explained'
topic_title: LSTM Networks Explained
---

### Sequences in Language, and Why Simple RNNs Struggle

Language is made of sequences. Every sentence is a list of words. Each word is a sequence of characters. Each item in the sequence depends on what came before it—and sometimes, what comes after.

Take this sentence:

> “I grew up in France, so I speak fluent ___.”

To predict the missing word (“French”), you need to remember “France” earlier in the sentence. This is called a **long-term dependency**: you have to use information from much earlier in the sequence to make a correct decision later.

**Recurrent neural networks** (RNNs) were invented to handle sequence data. An RNN takes one input at a time, updating its memory at each step. In theory, this lets it “remember” what happened earlier in the sequence.

But in practice, simple RNNs have trouble holding on to information as the sequence gets longer. Earlier information fades away. The technical name for this is the **vanishing gradient problem**: when training, error signals sent backwards through the RNN shrink rapidly, so the model forgets things from many steps ago. Plain RNNs work well only for short-term dependencies.

### The LSTM Cell: A New Type of RNN Building Block

**Long Short-Term Memory networks** (LSTMs) were designed to fix this memory problem.

An LSTM is a special kind of RNN block. Inside each LSTM “cell” are components called **gates**. You can think of these gates as switches or valves. They decide what information to keep, what to update, and what to forget as the network reads a sequence.

Each LSTM cell keeps its own memory, called the **cell state**. This is like a conveyor belt running through the network, carrying important data from one step to the next.

Each LSTM cell has four main parts:

- **Cell state:** The main memory, able to carry information forward for many steps.
- **Input gate:** Decides how much of the current input to write into memory.
- **Forget gate:** Decides how much of the previous memory to erase.
- **Output gate:** Decides how much of the memory to use to make an output.

Each gate is a small neural network: some weights and biases, trained to do the job. The gates look at the current input and what the cell remembers from the previous step (the hidden state), and then decide what to do.

### How LSTMs Handle Long-Range Dependencies

Picture the cell state as a highway running through the whole sequence. The gates are like traffic lights. The highway lets important information travel far. The lights decide who gets to stay, who merges in, and who exits.

- **Forget gate:** Looks at what's happening now and what the model remembers. It decides which parts of the memory are no longer useful and removes them from the cell state.
- **Input gate:** Lets new, relevant information onto the highway, updating the cell state with what's important right now.
- **Output gate:** Selects what information leaves the memory and becomes visible as the “output” for this step.

Because the cell state can carry information untouched through many sequence steps, and the gates only make precise, targeted changes, LSTMs can remember key facts far back in the sequence—even hundreds of steps ago.

This makes LSTMs good at problems where you need long memory, like predicting the next word in a sentence, machine translation, or anything with important patterns spread out over time.

### A Minimal LSTM Example on Text

Here’s a toy example. We'll use an LSTM to predict the next character in a sequence using Keras (a popular Python deep learning library).

We’ll use the sequence “hel” and try to predict the next letter, “l”.

```python
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Map characters to integers: {'h':0, 'e':1, 'l':2, 'o':3}
char_to_int = {'h':0, 'e':1, 'l':2, 'o':3}
int_to_char = {i: c for c, i in char_to_int.items()}

# Input sequence: "h", "e", "l"
X = np.array([[ [0], [1], [2] ]])  # shape (samples, timesteps, features)
# Next char to predict: "l"
y = np.array([2])
# One-hot encode output
y = np.eye(4)[y]

# Build model: 3 timesteps, 1 feature per step
model = Sequential([
    LSTM(8, input_shape=(3, 1)),
    Dense(4, activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Train briefly (just to show it runs)
model.fit(X, y, epochs=100, verbose=0)

# Predict what comes after "hel"
pred = model.predict(X)
print("Predicted next char:", int_to_char[np.argmax(pred[0])])
```

**Inputs:** The input is a sequence of three characters, each mapped to an integer, then turned into a column vector.  
**Outputs:** The model picks the most likely next character out of four possible characters.

**What's happening:**  
- At each step (“h”, then “e”, then “l”), the LSTM updates its memory with its gates.
- After the last letter, the LSTM predicts the next character.

This example is small for teaching. In practice, you’d use much larger inputs and vocabularies. But the pattern is the same: a sequence goes in, the LSTM processes it step by step, and then predicts the next item.

### When to Use LSTMs (and When Not To)

LSTMs are best when your data comes in order and things far apart in the sequence matter. Examples:

- Predicting the next word or character in text
- Translating one language to another, sequence by sequence
- Speech recognition, music, or any task where order and timing are important

LSTMs solved the memory loss in plain RNNs and pushed the field forward. But they are not the top choice for every sequence problem today. For much longer sequences, or for tasks that need to be trained very quickly and in parallel, the **Transformer** model works better. Transformers use **attention** instead of recurrence. They remember details from anywhere in the sequence and train faster.

For sequences up to a few hundred steps, LSTMs remain practical and easy to use with existing tools. For much longer text or very large context windows, newer models are a better fit.

---

## Key Takeaways

- Language involves sequences where later predictions depend on earlier context.
- Simple RNNs suffer from the vanishing gradient problem and cannot remember long-term dependencies well.
- LSTM networks use gates to control memory, allowing them to retain important information across many sequence steps.
- Gates in LSTMs include the input gate, forget gate, output gate, and the cell state.
- LSTMs outperform plain RNNs on tasks requiring long-context memory, such as language modeling and translation.

## Try It Yourself

Run the provided minimal LSTM example using Keras to predict the next character in a short sequence (e.g., predicting 'l' after 'hel'). Then, change the input sequence to something else (like 'heo') and observe how the output prediction changes. Reflect on how the model responds to different input contexts.

## Further Resources

- 🎥 [LSTM Networks: Explained Step by Step!](https://www.youtube.com/watch?v=P_TZN8kRObQ)
- 🎥 [Long Short-Term Memory - LSTM Models with TensorFlow (MLCon conference talk)](https://www.classcentral.com/course/youtube-long-short-term-memory-lstm-models-with-tensorflow-ml-conference-session-sahil-dua-241059)
- 📄 [Understanding LSTM Networks (Christopher Olah’s blog)](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- 📄 [LSTM Models: A Complete Guide to Long Short‑Term Memory Networks (DataCamp)](https://www.datacamp.com/tutorial/lstm-models)

---

**Coming up on Day 21:** GRU Networks Explained