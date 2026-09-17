---
day: 22
generated_at: '2026-09-17T13:53:11.137556+00:00'
phase: Phase 3 — Sequence Models & Deep Learning for NLP
recap_summary: Explained the structure and purpose of sequence-to-sequence (seq2seq)
  models, detailing how encoder-decoder architectures process and generate variable-length
  input and output sequences, with a hands-on example using character sequence reversal.
status: pending_review
title: 'Day 22: Sequence-to-Sequence Models'
topic_title: Sequence-to-Sequence Models
---

## What Are Sequence-to-Sequence Models?

A sequence-to-sequence model (often called a seq2seq model) is a type of neural network that transforms one sequence into another. In natural language processing (NLP), a "sequence" is usually a sentence, a paragraph, or a list of words. Seq2seq models excel when both the input and the output can be different lengths.

These models are central to tasks like machine translation (turning an English sentence into a French sentence), text summarization (condensing a paragraph into a short summary), and building conversational agents (turning a question into a relevant reply). Unlike simple classification, which produces just a single label per input, these tasks require outputs that are also sequences, sometimes longer or shorter than the input.

Seq2seq models are designed specifically for cases where you need to map entire sequences of any length to corresponding output sequences, which can also vary in length.

## Why Do We Need Sequence Models?

Compare two tasks:

In sentiment analysis, you read a whole review and output a single answer like “positive” or “negative.” Inputs might be different lengths, but the output is always a single label. Feed-forward or convolutional neural networks can usually handle this by padding or truncating inputs so they’re always a fixed size, then producing a fixed-size output vector.

Now look at machine translation. Both the input (“I like cats”) and output (“J’aime les chats”) are sequences, and their lengths are unrelated. A fixed-size network can’t naturally output a sentence of arbitrary length. Padding outputs so they’re all the same length isn’t practical—you may end up wasting space, or worse, cutting sentences short.

Tasks like translation, summarization, dialog, or code generation require systems that:
- Process inputs of any length
- Generate outputs of any length
- Preserve relationships across both sequences

Seq2seq models solve these problems by letting networks handle full input and output sequences, without fixed sizes, while learning how each part of the input relates to the output.

## How Do Seq2Seq Models Work?

The core seq2seq model has two main parts: an **encoder** and a **decoder**.

- **Encoder**: The encoder reads the input sequence one token at a time (a token could be a character, word, or subword unit). Each token is first mapped to a numeric vector (an embedding). As tokens are read, the encoder maintains a **hidden state**—a summary of what’s been seen so far. After the last token, the encoder outputs a set of numbers called the **context vector**. In LSTMs, this means the last hidden state and the last cell state. This context is a summary of the entire input.

- **Decoder**: The decoder receives the context vector and generates an output sequence, one token at a time. At each step, it uses the context from the encoder *and* what it has already generated. This allows it to decide the next token based on both what it knows about the input and what it has produced so far.

Here's the basic process, step by step:
1. Each input token is converted to an embedding (a vector).
2. The encoder processes these vectors one by one, updating its hidden state after every input.
3. When finished, the encoder passes its final hidden (and cell) state to the decoder—that’s the context vector.
4. The decoder starts with this initial state and a special start-of-sequence token (`<sos>`). It predicts the next output token. After each prediction, that token becomes input for the next step, until an end-of-sequence token (`<eos>`) is produced or a maximum length is reached.

This architecture means the model can flexibly handle sequences of any reasonable length. While modern models often use "attention" to further improve performance, the fundamental encode-then-decode pattern remains the same.

## Core Components: Encoder and Decoder

**Encoder:**
- Reads each token and converts it to a numeric vector.
- Maintains a hidden state—a set of numbers summarizing all tokens seen so far.
- After the last token, produces a final set of hidden (and cell, in LSTMs) states as the **context vector**. This context summarizes the input as compactly as possible.

**Decoder:**
- Receives the context vector from the encoder.
- Generates the output sequence one token at a time.
- Takes as input the last predicted token (starting with `<sos>`) and its own previous hidden and cell state.
- Stops generation when it produces the `<eos>` token.

A simple example:
1. Input: "ABC"
2. The encoder converts 'A', 'B', and 'C' to vectors and updates its state with each.
3. After 'C', its final state is a summary of "ABC".
4. The decoder uses that summary to generate 'C', then 'B', then 'A', taking its own previous output as input for the next prediction. It continues until it generates `<eos>`.

At every decoder step, both the summary context and the tokens generated so far influence the next output. This stepwise process from a fixed summary is the foundation of seq2seq modeling.

## Minimal Seq2Seq Example: Reversing Sequences

Let’s build a small seq2seq model using TensorFlow and Keras. This toy model reverses sequences of characters: given `['A', 'B', 'C']`, it outputs `['C', 'B', 'A']`. This illustrates the seq2seq pattern with minimal code.

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

# Vocabulary and helpers
vocab = ['<pad>', '<sos>', '<eos>'] + list("ABCDE")
vocab_size = len(vocab)
char_to_idx = {ch: idx for idx, ch in enumerate(vocab)}
idx_to_char = {idx: ch for ch, idx in char_to_idx.items()}

def encode_seq(seq):
    # Add start (<sos>) and end (<eos>) tokens
    return [char_to_idx['<sos>']] + [char_to_idx[c] for c in seq] + [char_to_idx['<eos>']]

def pad_seq(seq, max_len):
    # Pad with '<pad>' (index 0) to uniform length
    return seq + [char_to_idx['<pad>']] * (max_len - len(seq))

inputs = ['A', 'AB', 'ABC', 'ABCD', 'ABCDE']
input_seqs = [encode_seq(seq) for seq in inputs]
target_seqs = [encode_seq(seq[::-1]) for seq in inputs]  # reversed

max_len = max(len(seq) for seq in input_seqs + target_seqs)
X = np.array([pad_seq(seq, max_len) for seq in input_seqs])
Y = np.array([pad_seq(seq, max_len) for seq in target_seqs])

# Model parameters
embed_dim = 8
hidden_dim = 16

# Encoder
encoder_inputs = layers.Input(shape=(max_len,))
encoder_embedding = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, mask_zero=True)
x = encoder_embedding(encoder_inputs)
encoder_lstm = layers.LSTM(hidden_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(x)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = layers.Input(shape=(max_len,))
decoder_embedding = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, mask_zero=True)
x_dec = decoder_embedding(decoder_inputs)
decoder_lstm = layers.LSTM(hidden_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(x_dec, initial_state=encoder_states)
decoder_dense = layers.TimeDistributed(layers.Dense(vocab_size, activation="softmax"))
decoder_outputs = decoder_dense(decoder_outputs)

# Assemble model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Teacher forcing setup
Y_decoder_input = np.array([pad_seq([char_to_idx['<sos>']] + seq[1:], max_len) for seq in target_seqs])
Y_decoder_target = np.expand_dims(Y, -1)  # shape (batch, seq, 1)

# Train
model.fit([X, Y_decoder_input], Y_decoder_target, epochs=300, verbose=0)

# Inference setup: encoder model
encoder_model = Model(encoder_inputs, encoder_states)

# Inference decoder: predict one token at a time
decoder_state_input_h = layers.Input(shape=(hidden_dim,))
decoder_state_input_c = layers.Input(shape=(hidden_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
dec_emb_inf_input = layers.Input(shape=(1,))
dec_emb_inf = decoder_embedding(dec_emb_inf_input)
dec_outputs_inf, state_h_inf, state_c_inf = decoder_lstm(dec_emb_inf, initial_state=decoder_states_inputs)
dec_outputs_inf = decoder_dense(dec_outputs_inf)
decoder_model = Model([dec_emb_inf_input] + decoder_states_inputs, [dec_outputs_inf, state_h_inf, state_c_inf])

def decode_sequence(input_seq):
    # Encode the input as state vectors
    states_value = encoder_model.predict(input_seq, verbose=0)
    # Initial target sequence is '<sos>'
    target_seq = np.array([[char_to_idx['<sos>']]])
    decoded = []
    for _ in range(max_len):
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value, verbose=0)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = idx_to_char[sampled_token_index]
        if sampled_char == '<eos>' or len(decoded) > max_len:
            break
        decoded.append(sampled_char)
        target_seq = np.array([[sampled_token_index]])
        states_value = [h, c]
    return ''.join(decoded)

# Run a few test sequences
for test_seq in ['A', 'AB', 'ABC']:
    inp = np.array([pad_seq(encode_seq(test_seq), max_len)])
    print(f"Input: {test_seq} -> Predicted: {decode_sequence(inp)}")
```

**How this works:**
- The encoder ingests a padded, tokenized input (with padding values masked so they’re ignored during computation).
- The decoder is trained with **teacher forcing**: during training, it is fed the actual previous target token at each step, not its own previous guess. This accelerates learning by giving it the true recent history.
- During inference, the decoder has to generate each token itself: its last output becomes the input for the next prediction, repeating until `<eos>` or a maximum length is reached.
- The **LSTM hidden state** (`state_h`) is its current "memory"—what it remembers about the sequence so far. The **cell state** (`state_c`) carries longer-term information to help deal with long sequences.
- Embedding layers translate tokens (like 'A') into small numeric vectors that the LSTM can process more effectively.
- Padding allows us to batch sequences of different lengths in a way libraries like Keras can handle, while masking prevents the model from learning to “predict the padding.”

This code demonstrates the core seq2seq workflow:
- The encoder summarizes an arbitrary-length input into a fixed set of numbers.
- The decoder generates an arbitrary-length output, one token at a time, based on that context and what’s been produced so far.

Most real-world NLP systems build on these same principles, adding layers like attention to handle longer and more complex data. The fundamental workflow—encoding a flexible input, then decoding a flexible output, one step at a time—remains the backbone of sequence-to-sequence modeling.

---

## Key Takeaways

- Seq2seq models map input sequences to output sequences of arbitrary lengths.
- Encoder networks summarize input sequences into context vectors.
- Decoder networks generate output sequences step by step using context and prior outputs.
- Teacher forcing during training accelerates learning by providing true previous tokens.
- Variable-length batching is handled with padding and masking in practical implementations.

## Try It Yourself

Write a function that takes a list of word strings (e.g., ['cat', 'dog', 'rat']) and returns a new list where each word’s characters are reversed (['tac', 'god', 'tar']). Optionally, adapt the seq2seq example code to use your own custom token pairs for both input and target data. This simulates preparing data for a simple character-level sequence modeling task.

## Further Resources

- 🎥 [Encoder-Decoder Architecture Explained: Seq2Seq Models for Beginners](https://www.youtube.com/watch?v=ye02_9vFr_k)
- 📘 [NLP From Scratch: Translation with a Sequence to Sequence Network and Attention — PyTorch Tutorials](https://docs.pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)
- 📄 [An introduction to sequence-to-sequence learning](https://lorenlugosch.github.io/posts/2019/02/seq2seq/)
- 📄 [18.1 Encoder‑Decoder Models for Sequence Transduction – Dive into Deep Learning](https://d2l.smola.org/chapter_natural-language-processing-pretraining/seq2seq.html)
- 📄 [Neural Machine Translation and Sequence‑to‑sequence Models: A Tutorial](https://arxiv.org/abs/1703.01619)

---

**Coming up on Day 23:** The Encoder-Decoder Architecture