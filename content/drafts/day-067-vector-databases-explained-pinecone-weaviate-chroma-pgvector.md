---
day: 67
generated_at: '2026-11-26T18:42:00.311181+00:00'
phase: Phase 9 — Embeddings & Retrieval / RAG Foundations
recap_summary: Explained the purpose and function of vector databases, emphasizing
  how they store and search embeddings to enable similarity-based queries in AI applications.
  Compared vector databases with traditional databases and introduced popular systems,
  along with a hands-on code example using Chroma.
status: pending_review
title: 'Day 67: Vector Databases Explained (Pinecone, Weaviate, Chroma, pgvector)'
topic_title: Vector Databases Explained (Pinecone, Weaviate, Chroma, pgvector)
---

### Why Do We Need Vector Databases?

Language models and search systems create *embeddings*: long lists of decimal numbers that capture the meaning of texts, images, or other complex data. These lists—vectors—usually have hundreds or thousands of values. To find content that's similar (not just identical) to a search query, you need to compare these vectors and see which are "closest."

For simple or structured data—like user IDs or timestamps—conventional SQL databases work well. You can filter rows with clear rules like `WHERE name="Alice"` or `WHERE age>30`. But what if you want to ask, "Find me texts most similar in meaning to this question?" Standard databases aren't designed to handle hundreds of floating-point values per record or to compare them based on similarity.

Vector databases solve this. They're built for searching high-dimensional data—especially for cases where "similarity" matters more than exact matches.

---

### What Is a Vector Database?

A vector database stores and searches *vectors*—which you can picture as long coordinates that mark positions in a mathematical space. In AI, those "positions" are typically the embeddings produced by language models.

A traditional database is like a neatly organized warehouse. Each box is labeled with clear names and numbers. You can easily ask, "Which boxes are labeled 'Alice'?" and get a quick answer.

A vector database, instead, is like a huge high-dimensional map. Each data item—like a piece of text—gets a spot on this map. When you search, you don't look for a matching label. Instead, you ask, "Which items are closest to this point?" The database efficiently finds your nearest neighbors—even among millions of items and hundreds of dimensions.

---

### Embeddings: The Core Data

Reminder from earlier: An *embedding* is a fixed-length list of numbers capturing the key features of data, such as the meaning of a sentence or the style of an image. For example, a language model might convert "cat" into `[0.1, -0.05, 0.27, ...]`.

Embeddings are *dense*: similar content ends up as similar lists. So, if you want to find data like "cat," you search for vectors close to `[0.1, -0.05, 0.27, ...]`. But because these are long lists instead of single labels or numbers, the storage and search methods must be different.

---

### How Vector Search Works

The core function of a vector database is *similarity search* or *nearest neighbor search*.

In SQL, you might search for people `WHERE age=42`. It's an exact match. With embeddings, you usually want, "Show me what is most similar." For this, databases use a *distance function*—a math formula that says how "far apart" two vectors are. Popular choices are *cosine similarity* and *Euclidean distance*.

Imagine a globe covered with millions of tiny dots, each representing a vector. If you add a new dot (your search embedding), finding out "Which other dots are closest?" is the main job. For language tasks, this powers semantic search, question answering, recommendations, and more.

Efficient vector search in high dimensions is hard. Vector databases use smart data structures and search algorithms (like *approximate nearest neighbors*, or ANN) to keep searches fast even with huge datasets.

---

### Popular Vector Database Systems

Four common systems:

- **Pinecone**: Commercial, cloud-based, and fully managed. Fast, scalable, and supports metadata filtering. Commonly used for retrieval-augmented generation (RAG).

- **Weaviate**: Open-source and also available managed in the cloud. Lets you combine vector and keyword searches, attach metadata, and use built-in machine learning for some tasks.

- **Chroma**: Lightweight, open-source, and pure Python. Great for experiments and prototypes. Works in-memory or with simple persistent storage.

- **pgvector**: An extension for PostgreSQL. Adds vector search features to a traditional relational database. Useful if your system already relies on PostgreSQL.

Each system suits different needs. The best choice depends on your scale, infrastructure, and how you want to mix vector search with other data.

---

### Example: Searching Embeddings with Chroma

Chroma is a simple vector database that works well for small projects and learning. Let’s use it to store a few embeddings and search for the most similar entry.

Here, we'll use small, fake vectors to focus on what’s happening. In a real setup, embeddings are generated by a language model.

```python
# Install Chroma (if not already): pip install chromadb

import chromadb
from chromadb.utils import embedding_functions

# Instantiate an in-memory Chroma client
client = chromadb.Client()
collection = client.create_collection("my_collection")

# Example 3D embeddings for three sentences
embeddings = [
    [0.1, 0.2, 0.3],   # "Cats are playful animals."
    [0.2, 0.1, 0.4],   # "Dogs love to play."
    [0.9, 0.8, 0.7]    # "Economics is the study of value."
]
documents = [
    "Cats are playful animals.",
    "Dogs love to play.",
    "Economics is the study of value."
]
ids = ["doc1", "doc2", "doc3"]

# Insert embeddings, texts, and ids into the collection
collection.add(
    embeddings=embeddings,
    documents=documents,
    ids=ids
)

# Example query embedding (like "Playful pets")
query_embedding = [[0.15, 0.19, 0.31]]

# Search for the two most similar entries
results = collection.query(
    query_embeddings=query_embedding,
    n_results=2
)

print("Most similar texts to 'Playful pets':")
for doc in results['documents'][0]:
    print(doc)
```

This code stores three simple embeddings and asks, "Which are most like 'Playful pets'?" Chroma returns the closest matches, measuring similarity—not just keywords or exact matches.

With real language model embeddings, vector databases unlock powerful search and recommendation capabilities that traditional databases can’t match. They're the core of modern systems for RAG and deep semantic search.

---

## Key Takeaways

- Vector databases are designed to store and search high-dimensional embeddings efficiently.
- Embeddings represent the meaning or features of data as dense vectors.
- Similarity search in vector databases relies on distance functions like cosine similarity or Euclidean distance.
- Popular vector database systems include Pinecone, Weaviate, Chroma, and pgvector.
- Vector search enables semantic retrieval beyond exact keyword matching.

## Try It Yourself

Pick 3-5 short sentences of your own and store them in a local Chroma vector database using simple embeddings (e.g., small lists of numbers). Then, create a new 'query' sentence with its own embedding and search for the stored sentence most similar to it. Modify your query to observe how the results change.

## Further Resources

- 🎥 [Vector Databases Explained (2026) | Beginner's Guide to RAG](https://www.youtube.com/watch?v=4MgY6kUxOdI)
- 🎥 [Vector Databases Explained: The Complete Guide for 2026](https://www.youtube.com/watch?v=4pUYfY-b5CQ)
- 📄 [ChromaDB: Embeddings and Vector Databases in Python – Real Python](https://realpython.com/chromadb-vector-database/)
- 📄 [Vector Databases: Embeddings, ANN Indexes, and the Retrieval Layer for AI – The HLD Handbook](https://hld.handbook.academy/curriculum/data-systems/vector-databases/)
- 📄 [RAG and Vector Databases Tutorial – Kenhuang Academy](https://kenhuang.com/en/rag/index.html)

---

**Coming up on Day 68:** Chunking Strategies for Documents