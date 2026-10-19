---
day: 40
generated_at: '2026-10-19T15:57:45.888168+00:00'
phase: Phase 5 — Pretraining & Language Models
recap_summary: Explained the importance of tokenization in NLP, comparing word, subword,
  and character tokenization, and introduced the key algorithms BPE, WordPiece, and
  SentencePiece, highlighting their differences and practical applications.
status: pending_review
title: 'Day 40: Tokenization Deep Dive: BPE, WordPiece, SentencePiece'
topic_title: 'Tokenization Deep Dive: BPE, WordPiece, SentencePiece'
---

Why Tokenization Matters

Computers do not understand language the way humans do. For natural language processing (NLP), the first step is turning text into something algorithms can work with: numbers. Tokenization is the process of breaking raw text into pieces called tokens. A token can be a word, a piece of a word (subword), or a single character.

The choice of tokenization matters because it defines the units an NLP model will learn from. If you tokenize at the character level, the model must learn how characters form words from scratch. If you tokenize at the word level, the model cannot recognize words it has not seen during training. A good tokenization strategy lets models handle new words, understand a wide variety of languages, and represent symbols efficiently.

From Words to Subwords

Older NLP systems often used word-level tokenization. This means splitting text by spaces and sometimes punctuation. Word-level tokenization has some clear drawbacks:

- Large vocabulary: Every different word becomes a token. This quickly grows into a huge list the model must learn to handle.
- Out-of-vocabulary (OOV) problem: Words not seen in training become "unknown" tokens. This is especially bad for rare words, names, or languages with many word forms.
- Poor handling of compound words: Many languages create new words by combining smaller pieces. Word tokenizers cannot generalize to new compounds or inflections.

Subword tokenization solves these issues. It breaks words into smaller, meaning-preserving pieces. When the model sees a new word, it can represent it as a series of familiar subwords or, at worst, individual characters.

What Is Byte-Pair Encoding (BPE)?

Byte-Pair Encoding (BPE) is a subword tokenization algorithm. It starts with the input text split into individual characters (including a symbol to mark the start of a new word, like '_'). BPE then repeatedly finds the most frequent pair of symbols and merges them into a new symbol, updating the text every time. These merged symbols can be character pairs, triplets, or longer pieces.

Here's a concrete example. Suppose our text is:

```
low lower newest widest
```

BPE starts by treating each character as a symbol (including the '_' word boundary). The algorithm works like this:

1. Count all pairs of adjacent symbols.
2. Find the most frequent pair (for example, 'e' and 's'), then merge 'e''s' into the new symbol 'es' everywhere.
3. Repeat: find the next most frequent pair and merge, updating your vocabulary and the text after each step.

After a fixed number of merges, BPE produces a vocabulary that includes entire words it saw often, plus subword units for rarer patterns ('est', 'wid', etc.). Now, if you encounter a new word, you can break it into these subwords. BPE is simple and helps the model generalize to unseen words.

WordPiece: An Alternative Approach

WordPiece is another popular subword tokenization method. It was first used for speech recognition and is best known for BERT and similar models.

WordPiece also builds subword units from characters, but its merge process is different. Instead of merging the most frequent symbol pair, WordPiece merges pairs to maximize the probability of the training data under a language model (a statistical model of word sequences). This often leads to splits that line up better with language structure, especially for rare words.

WordPiece always picks the longest known subword from left to right. You'll notice this when breaking down rare or ambiguous words: it may produce different, and sometimes more meaningful, splits compared to BPE.

SentencePiece: Language-Agnostic Tokenization

SentencePiece is a general-purpose tokenization library. Unlike BPE and WordPiece, it does not require text to be split by spaces first. This makes it well-suited to languages without spaces (like Japanese or Chinese) and to data such as URLs, code, or mixed-language content.

SentencePiece treats the text as a continuous stream of raw bytes or Unicode characters, including spaces. It supports both BPE and a method called Unigram Language Model (Unigram LM) tokenization. The key difference: SentencePiece builds subwords from raw text, without language-specific preprocessing. Any substring is allowed as a potential subword, not just those that line up with what we think of as "words."

This approach offers several advantages:

- No need for custom rules for each language.
- Easily handles edge cases like typos, concatenated words, or unusual symbols.
- Works well for multilingual or mixed-data scenarios.

Comparing BPE, WordPiece, and SentencePiece

All three methods turn text into subwords, but differ in the details:

|               | Input assumption   | Merge criterion                 | Needs whitespace? | Where used           |
|---------------|--------------------|---------------------------------|-------------------|----------------------|
| BPE           | Characters/bytes   | Most common symbol pair         | Typically, yes    | GPT, RoBERTa         |
| WordPiece     | Characters/bytes   | Language model likelihood       | Typically, yes    | BERT, ALBERT         |
| SentencePiece | Raw text/bytes     | BPE or Unigram LM               | No                | T5, mT5, XLNet       |

**Pros and cons:**
- BPE: Fast and simple, but sometimes splits words strangely.
- WordPiece: Smarter splits, but can be less transparent.
- SentencePiece: Most flexible; handles any language or data; tokens can be unpredictable.

Practical Tokenization: Try It Yourself

Let's see how to use SentencePiece to train a simple tokenizer, then encode and decode text with it. Install SentencePiece with:

```
pip install sentencepiece
```

We'll use a few example sentences as our training data.

```python
import sentencepiece as spm

# Write training data to a file
with open('train.txt', 'w') as f:
    f.write("low lower newest widest")

# Train a SentencePiece model with a small vocabulary
spm.SentencePieceTrainer.Train('--input=train.txt --model_prefix=demo --vocab_size=20')

# Load the trained model
sp = spm.SentencePieceProcessor()
sp.load('demo.model')

# Encode a new word into token IDs
ids = sp.encode("lowered", out_type=int)
print("Token IDs:", ids)

# Decode the token IDs back to text
decoded = sp.decode(ids)
print("Decoded text:", decoded)

# See the subword tokens
tokens = sp.encode("lowered", out_type=str)
print("Subword tokens:", tokens)
```

Sample output:

```
Token IDs: [5, 8, 9, 3]
Decoded text: lowered
Subword tokens: ['▁low', 'er', 'e', 'd']
```

The special character `▁` marks the start of a new word in SentencePiece encoding. Even though "lowered" was not in the training data, SentencePiece breaks it into "low," "er," "e," "d"—subwords it learned.

This shows how subword tokenization allows NLP models to handle new words gracefully, even if they never appeared in training. Subword tokenization underpins reliable, flexible performance in modern natural language models.

---

## Key Takeaways

- Tokenization converts text into tokens for algorithms to process in NLP.
- Word-level tokenization struggles with unseen words and large vocabularies.
- Subword tokenization, via BPE or WordPiece, helps generalize to new or rare words.
- SentencePiece tokenizes raw text without language-specific preprocessing, supporting non-whitespace languages.
- Practical tools like SentencePiece allow custom tokenization models to be easily trained and used.

## Try It Yourself

Download a short text sample (e.g., a Wikipedia paragraph) and save it to a file. Train a simple SentencePiece tokenizer on your sample, then encode the text into tokens. Examine the resulting tokens for subwords and try decoding the token sequence back to the original text to observe how subword modeling works in practice.

## Further Resources

- 🎥 [LLM Tokenizers Explained: BPE Encoding, WordPiece and SentencePiece](https://www.youtube.com/watch?v=hL4ZnAWSyuU)
- 📘 [WordPiece Tokenization · Google's Subword Algorithm — SentencePiece Guide](https://sentencepiece.org/concept/wordpiece-tokenization/)
- 📄 [Tokenization Deep Dive: BPE, WordPiece & SentencePiece](https://mo-abdalkader.github.io/Portfolio/blog/tokenization-deep-dive.html)
- 📄 [Tokenization Deep Dive: BPE, WordPiece, and SentencePiece](https://enricopiovano.com/blog/tokenization-deep-dive/)

---

**Coming up on Day 41:** Scaling Laws: Why Bigger Models Work Better