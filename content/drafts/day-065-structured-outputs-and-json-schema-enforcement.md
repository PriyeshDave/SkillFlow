---
day: 65
generated_at: '2026-11-24T19:46:19.835038+00:00'
phase: Phase 8 — Working With LLM APIs
recap_summary: Explained the need for structured outputs from LLMs in real-world applications,
  introduced JSON as a standard data format, and demonstrated how JSON Schema and
  schema enforcement improve consistency and reliability in LLM-generated data.
status: pending_review
title: 'Day 65: Structured Outputs and JSON Schema Enforcement'
topic_title: Structured Outputs and JSON Schema Enforcement
---

## When LLMs Need to Output Structured Data

Large language models (LLMs), like GPT-4, are often used to answer questions, summarize text, or write stories. These tasks produce plain text, which is easy for humans to read.

But many real-world applications need more structure. For example, you might build:

- A customer support chatbot that must return details as specific fields, like `"user_name"`, `"issue_type"`, and `"priority"`.
- An automated tool that fills out database entries.
- A virtual assistant that schedules meetings based on information it extracts.

In these cases, you need more than just an answer in sentences. You want information in a predictable, machine-readable format. Without it, your code can’t reliably extract or use the results. Parsing text is brittle—any wording change can break things.

## What Is JSON? Why It's the Common Language for AI Outputs

JSON stands for JavaScript Object Notation. It's a way to represent structured data as nested key-value pairs, lists, and basic types, like strings, numbers, and booleans. Here’s an example, showing data about one person:

```json
{
  "name": "Alice",
  "age": 30
}
```

JSON is easy for code to generate and read. It’s straightforward for humans, too—no tricky syntax. Servers, web apps, databases, and APIs all use JSON because everyone agrees on what the data means.

When an LLM outputs JSON, your code can use the result directly. There’s no need to guess or clean up the answer. JSON acts as a bridge between what the AI generates and what your software needs.

## Example: What Can Go Wrong Without Structure

Suppose you want an LLM to give you a person's name and age.

You might prompt it like this:

```
Tell me the name and age of a person.
```

The model could reply:

```
The person's name is Alice, and she is 30 years old.
```

Now, to get `"Alice"` and `30` in your code, you'd have to write brittle text-parsing logic. If the wording changes—maybe you get "Alice (30) is her name and age"—your code could fail.

If you instead prompt the LLM:

```
Please reply in JSON format with 'name' and 'age' keys.
```

You *might* get:

```json
{
  "name": "Alice",
  "age": 30
}
```

But you might also get:

```json
{
  "Name": "Alice",
  "Age": "thirty"
}
```

Or:

```
Sure! Here is the person info:
name: Alice,
age: 30
```

The outputs aren’t consistent. This is the problem that JSON Schema aims to solve.

## Introducing JSON Schema: A Blueprint for Output Consistency

JSON Schema is a standard way to describe what valid JSON data should look like. You define the allowed structure: which fields are required, their types, and possible values. It's a contract between whoever creates the data and whoever uses it.

Here’s a simple schema for our “person” example:

```json
{
  "type": "object",
  "properties": {
    "name": { "type": "string" },
    "age": { "type": "integer" }
  },
  "required": ["name", "age"]
}
```

This schema says: output an object with a `"name"` field (a string) and an `"age"` field (an integer), both required. If you use a modern LLM API that supports schema-guided output, the API will try to ensure the model sticks to this structure.

## How LLM APIs Use JSON Schema Enforcement

When you provide a JSON Schema to APIs from OpenAI, Google, or Anthropic, the LLM uses that "blueprint" to shape its output. It tries to produce only the specified keys and values, and the API may validate or correct minor errors.

Schema enforcement does not guarantee correctness, but it sharply reduces formatting problems. It also means your application can trust the returned data structure much more.

## Concrete Example: Asking an LLM for Structured Output with JSON Schema

Here's a Python example with OpenAI's API, showing a free-form prompt versus a schema-enforced one.

```python
import openai

openai.api_key = "sk-..."  # Use your real API key

# Define the schema: a person with a string name and integer age
person_schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"}
    },
    "required": ["name", "age"]
}

# 1. Free-form prompt (no schema)
response_freeform = openai.ChatCompletion.create(
    model="gpt-4-0125-preview",
    messages=[{"role": "user", "content": "Tell me the name and age of a person"}],
    max_tokens=50,
    temperature=0
)
print("Free-form output:")
print(response_freeform["choices"][0]["message"]["content"])

# 2. Schema-enforced response
functions = [{
    "name": "get_person",
    "parameters": person_schema,
    "description": "Generate a person with a name and age."
}]
response_structured = openai.ChatCompletion.create(
    model="gpt-4-0125-preview",
    messages=[{"role": "user", "content": "Generate a person."}],
    functions=functions,
    function_call={"name": "get_person"},
    max_tokens=50,
    temperature=0
)
print("\nSchema-enforced output:")
print(response_structured["choices"][0]["message"]["function_call"]["arguments"])
```

The free-form output might be:

```
The person's name is Alice, and she is 30 years old.
```

With schema enforcement, you'll get JSON like:

```json
{"name": "Alice", "age": 30}
```

## What To Watch For: Limitations and Common Pitfalls

LLMs and schema enforcement are not perfect:

- **Partial Compliance:** Models may miss some schema rules, especially if the schema is complex.
- **Hallucinated Keys:** Sometimes the model invents keys, or uses the wrong name ("full_name" instead of "name").
- **Wrong Types:** You might get `"30"` as a string instead of the integer `30`, or `"thirty"` spelled out.
- **Nesting Issues:** For complex schemas, the model may flatten, misplace, or skip nested fields.
- **Almost Right:** The output might be very close, but still need some minor cleanup.

Schema enforcement catches many errors, but always validate LLM outputs before using them in production code. LLMs are helpful assistants, not strict rule followers.

By using JSON and JSON Schema, you make LLMs much more reliable for software tasks. Structured outputs let you build automation, agents, and interfaces with fewer bugs and more confidence.

---

## Key Takeaways

- Plain text LLM outputs are unreliable for automation due to inconsistent formatting.
- JSON is a widely used, machine-readable format for structured data exchange.
- JSON Schema defines required fields and types, serving as a blueprint for expected output.
- Schema enforcement in LLM APIs increases output consistency but does not guarantee perfection.
- LLM outputs should always be validated before use, even with schema enforcement.

## Try It Yourself

Write a Python function that uses the OpenAI API to request a movie record with 'title' and 'year' fields, enforcing a JSON Schema for the expected structure. Examine whether the returned output matches the schema exactly, noting any discrepancies.

## Further Resources

- 📘 [Structured model outputs | OpenAI API](https://developers.openai.com/api/docs/guides/structured-outputs)
- 🎥 [Python + AI: Structured outputs (Microsoft Reactor)](https://www.youtube.com/watch?v=OudnE_HOwJA)
- 📄 [LLM structured output: JSON Schema, validation and retries in production (Daniele Petrollo blog)](https://blog.danielepetrollo.com/en/llm-structured-output-json-schema)

---

**Coming up on Day 66:** Embeddings for Retrieval: Revisiting the Concept for LLMs