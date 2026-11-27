---
day: 68
generated_at: '2026-11-27T18:32:20.206785+00:00'
phase: Phase 9 — Embeddings & Retrieval / RAG Foundations
recap_summary: Explained the importance of chunking in document processing, detailed
  fixed-size, sliding window, and semantic chunking strategies, and discussed tradeoffs
  in chunk size, overlap, and context for effective retrieval with language models.
status: pending_review
title: 'Day 68: Chunking Strategies for Documents'
topic_title: Chunking Strategies for Documents
---

---
### What Is Chunking in Document Processing?

Chunking means breaking a large document into smaller, manageable pieces called "chunks." Imagine taking a book and dividing it into small booklets by pulling out groups of pages. Each booklet is a chunk of the original book.

Modern language models like GPT, and embedding models, perform better when processing smaller text pieces.

- An **embedding** is a set of numbers that represents the meaning of a piece of text so a computer can work with it.
- An **embedding model** turns text into an embedding, usually as a list of numbers called a **vector**.
- A **vector** is simply an ordered list of numbers, like `[2.5, -1.6, 0.87]`. Vectors let computers compare the similarity between texts.

You can't ask a language model to process an entire book at once. These models have a hard limit—the **context window**—on how much text they can handle in a single request. Chunking lets models analyze, retrieve, and answer questions from large documents more effectively.

### Why Chunk Documents? The Problem With Long Texts

Say you want a model to answer a question about a lengthy technical manual. Most language models can only process a few thousand words at once (current limits usually max out around 100 pages). If you feed too much text, the model either ignores the overflow or refuses to process it.

This is like searching for a sentence in a long novel without chapters. You’d have to skim the entire book each time—slow, costly, and easy to make mistakes.

Chunking also helps with searching and retrieval. Looking for information in small parts is faster and more accurate than searching the whole document.

For example:
- Embedding models turn text into vectors, but these only capture the meaning of the chunk they’re built from. If the chunk is too big, the embedding may miss details. If it’s too small, it might lose context.
- Retrieval-Augmented Generation (RAG) systems retrieve relevant chunks—not entire documents—for answering questions.

Dividing text into manageable chunks is essential for scalable, accurate question-answering and document search.

### Common Chunking Strategies: Fixed, Sliding, and Semantic

#### Fixed-Size Chunking

This is the simplest strategy. Split the text into equal-sized pieces, like every 200 words or every 1,000 characters. Any leftover text forms a smaller final chunk.

**Example:**  
Text: `"This is a sample sentence. Here is another one. And another."`  
Chunk by 3 words:  
- `"This is a"`
- `"sample sentence. Here"`
- `"is another one."`
- `"And another."`

#### Sliding Window Chunking

Sliding window chunking makes overlapping chunks. Instead of non-overlapping sections, you "slide" a window across the text, producing chunks that overlap. This matters when important information crosses chunk boundaries.

**Example:**  
Chunk size: 4 words, overlap: 2 words  
Chunks:  
- `"This is a sample"`
- `"a sample sentence. Here"`
- `"sentence. Here is"`
- `"Here is another one."`
- `"is another one. And"`
- `"another one. And another."`

With a sliding window, you process all words. The final chunk might be shorter but includes all text. Overlap helps preserve meaning that may span between chunks.

#### Semantic (Content-Aware) Chunking

Semantic chunking splits text based on meaning, not just size. For example, it might chunk at paragraph or section boundaries, or use algorithms to keep related sentences together.

**Example:**  
For a technical article, instead of chopping every 200 words, you split by section headings or paragraphs, grouping by topic.

This usually requires natural language processing or metadata to find meaningful cut points. Semantic chunking means each chunk is a self-contained paragraph or topic.

### Tradeoffs in Chunking: Size, Overlap, and Semantics

**Chunk Size:** Too large, and each chunk risks exceeding the model’s input limit, or the embedding may mix unrelated ideas. Too small, and you lose context, making each chunk harder to understand.

**Chunk Overlap:** Overlap helps catch ideas that cross boundaries, but more overlap means more storage and slower searches because of duplicated content.

**Semantic Boundaries:** Size-based chunking can split sentences or paragraphs awkwardly. Semantic chunking, which avoids this, takes more computation and sometimes needs additional information.

Common problems:
- Splitting in the middle of a sentence makes chunks hard to interpret or search.
- No overlap means answers spanning boundaries can be missed.
- Too much overlap causes redundant data and slows retrieval.
- Chunks too big or small can result in embeddings that fail to capture meaning.

### Choosing a Chunking Strategy for Your Application

Several factors influence chunking:

**Document Type:**  
- Legal or technical manuals: Chunk by section or subsection to keep context.
- Conversation logs: Chunk by speaker turn or topic change.
- News articles: Paragraph-based chunking often works best.

**Retrieval System:**  
- For keyword search, smaller chunks improve accuracy.
- For dense retrieval using embeddings, medium, meaning-preserving chunks are best.

**Language Model Limits:**  
- If your model has a 2048-token context limit, aim for chunks under 1500 tokens to allow room for questions and formatting.

**User Queries:**  
- For summary requests, larger chunks help.
- For fact lookup, small, focused chunks are better.

Most systems try a few chunk sizes and overlaps, then measure retrieval performance (does the top result contain the right answer?) and adjust as needed.

### Example Code: Fixed-Size and Sliding Window Chunking

Minimal Python examples for fixed-size and sliding window chunking:

```python
def fixed_size_chunks(text, chunk_size):
    words = text.split()
    return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def sliding_window_chunks(text, chunk_size, overlap):
    words = text.split()
    step = chunk_size - overlap
    chunks = []
    for i in range(0, len(words), step):
        chunk = words[i:i+chunk_size]
        if chunk:
            chunks.append(' '.join(chunk))
    return chunks

sample_text = "Chunking strategies are crucial for document retrieval tasks, especially when working with large language models that have limited context windows. Effective chunking helps preserve meaning and enhances search performance."

print("Fixed-size chunks (5 words):")
for chunk in fixed_size_chunks(sample_text, 5):
    print("-", chunk)

print("\nSliding window chunks (5 words, 2-word overlap):")
for chunk in sliding_window_chunks(sample_text, 5, 2):
    print("-", chunk)
```

**Output:**
```
Fixed-size chunks (5 words):
- Chunking strategies are crucial for
- document retrieval tasks, especially when
- working with large language models that
- have limited context windows. Effective
- chunking helps preserve meaning and
- enhances search performance.

Sliding window chunks (5 words, 2-word overlap):
- Chunking strategies are crucial for
- are crucial for document retrieval
- for document retrieval tasks, especially
- retrieval tasks, especially when working
- especially when working with large
- working with large language models
- large language models that have
- models that have limited context
- have limited context windows. Effective
- windows. Effective chunking helps
- chunking helps preserve meaning and
- preserve meaning and enhances search
- and enhances search performance.
```

This script shows:
- How to split text into fixed-size word chunks.
- How to make overlapping chunks with a sliding window, while ensuring all content is included.

Try different chunk sizes and overlaps with your own text to see the effects. Every use case has its own optimal chunking—choose and tune thoughtfully.
---

---

## Key Takeaways

- Chunking breaks large documents into smaller, manageable text pieces called chunks.
- Language models and embedding systems require chunking to stay within their context window limits.
- Fixed-size, sliding window, and semantic chunking are the main chunking strategies.
- Choosing chunk size and overlap affects retrieval accuracy, speed, and meaning preservation.
- Selecting the right chunking approach depends on document type, retrieval needs, and language model limits.

## Try It Yourself

Find a short article or blog post you like and copy its text into a document. Manually split it into fixed-size chunks (e.g., every 20 words), then try creating overlapping chunks with a sliding window (e.g., 20-word chunks overlapping by 10 words). Compare the results and reflect on how each method changes the way information is grouped and understood.

## Further Resources

- 📘 [Chunk Documents for Retrieval · LangChain Docs](https://xiaoqianran.github.io/langchain-docs/oss/python/langchain/chunking.html)
- 📘 [Chunk large documents for RAG and vector search in Azure AI Search](https://learn.microsoft.com/en-us/azure/search/vector-search-how-to-chunk-documents)
- 📄 [Chunking Strategies for RAG – Breaking Down Documents for Better Retrieval](https://sifal.social/posts/Chunking-Strategies-for-RAG-Breaking-Down-Documents-for-Better-Retrieval/)
- 🎥 [Chunking for beginners: 3 simple techniques in RAG systems (YouTube)](https://www.youtube.com/watch?v=HJHSNVqQBJI)
- 🎥 [Lecture-11: RAG for Beginners: Text Chunking Explained | Split Documents for Better AI Retrieval (YouTube)](https://www.youtube.com/watch?v=C5q-xMJwBQM)

---

**Coming up on Day 69:** Introduction to Retrieval-Augmented Generation (RAG)