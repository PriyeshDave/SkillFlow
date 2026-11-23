---
day: 59
generated_at: '2026-11-16T19:00:28.689197+00:00'
phase: Phase 7 — Prompt Engineering
recap_summary: Explained the difference between freeform and structured output in
  language models, introduced JSON formatting for structured data extraction, covered
  best practices for structured output prompting, and demonstrated how function calling
  enables more reliable, machine-usable results.
status: pending_review
title: 'Day 59: Structured Output Prompting (JSON Mode, Function Calling)'
topic_title: Structured Output Prompting (JSON Mode, Function Calling)
---

## What Is Structured Output and Why Does It Matter?

Language models generate text that sounds natural to humans. This is called "freeform" output. Freeform text works well for people, but computers struggle to use it directly. For example, if you ask a model:

*"What are the most important features of a good bicycle?"*

You might get:

*"A good bicycle should be lightweight, have reliable brakes, and a comfortable seat."*

Humans can read and understand this. But a computer program can't easily pull out those features, because the information isn't structured for code.

**Structured output** means asking the model to organize information in a predictable format—a list, a table, or a set of labeled fields—so programs can use the data automatically. This matters wherever LLMs power tools, automate tasks, or feed information into data pipelines.

Example:

**Unstructured:**  
"The capital of France is Paris."

**Structured:**  
`{"country": "France", "capital": "Paris"}`

With this structure, the meaning is explicit and machine-readable. There's less ambiguity. Downstream software can work with the results, not just display them to humans.

## What’s a Prompt? (Quick Refresher)

A **prompt** is the instruction you give a language model. It's the input text that tells the model what you want.

Prompt: `"List three fruits."`  
Model output: `"Apple, banana, orange."`

Prompts determine the style, tone, length, and structure of the response. When you want structured, predictable outputs, crafting the prompt carefully becomes even more important. This is called **prompt engineering**.

## Structured Output Prompting: Getting Predictable Results

To automate tasks, you often need the model’s reply in a format like **JSON** (JavaScript Object Notation). JSON is a widely-used system for representing data as key-value pairs. With JSON, you know exactly what shape the data will have. That means you can write code to process results, instead of guessing their meaning.

Suppose you are building a customer support bot. You want to turn a user's complaint into data fields, like:

`{"issue_type": "billing", "urgency": "high", "customer_name": "Kim"}`

If you only get conversational text back, automation falls apart.  
**Structured output prompting** means telling the model, as clearly as possible: “Return the answer in this format.” This usually means listing required fields and specifying the format in the prompt.

Example prompt:

*"Extract from the text below the following fields as JSON:  
- product_name  
- issue  
- urgency  
Text: 'My internet is down and I need it fixed today. The router is flashing red.'  
Respond only with valid JSON."*

Effective prompts:

- Explicitly list the fields you want.
- Specify the desired format (like JSON).
- (Best) Include a sample output.

## JSON: The Foundation of Structured Output

**JSON** (JavaScript Object Notation) represents data as key-value pairs. Each piece of data has a label (the "key") and a value. JSON is easy for humans to read and for machines to parse.

- *Freeform text* is flexible and readable, but ambiguous.
- *JSON* is rigid and unambiguous.

Why require JSON?

- Almost any programming language can read ("parse") JSON.
- No guesswork. No ambiguity about "did the model return the country, or the capital, or both?"
- You can pass JSON directly between systems.

**How to prompt for JSON:**

1. List required fields.
2. Tell the model to reply with JSON only—no extra text.
3. (Optional) Specify data types.
4. (Best practice) Show a sample JSON output.

Example prompt:

```plaintext
Extract the following info as JSON:
- country (string)
- capital (string)
Text: "The capital of France is Paris."
Respond only with valid JSON.
Example:
{"country": "France", "capital": "Paris"}
```

The clearer and more concrete your prompt, the more reliable the output. Vague prompts or ambiguous input will still cause errors, even with good models.

## Function Calling: Next-Level Structured Output

Some new LLMs (like OpenAI's GPT-4 and certain open-source models) support **function calling**. Here, you define what "API function" is available, with its specific input parameters. The model receives this definition, plus your query. It answers by filling in the fields, not by generating a text reply.

This makes integration much more reliable and direct. Instead of parsing JSON out of text, you receive machine-ready arguments—no text processing needed.

Example:

Suppose you have a function:

```python
def answer_question(question: str, context: str) -> dict:
    ...
```

You tell the model about the function and its arguments. When prompted:

*"Based on the article below, answer the user's question."*

The model returns:

```json
{
  "question": "What is the capital of France?",
  "context": "France is a country in Europe. Its capital is Paris."
}
```

Your code can use the fields directly—no need to parse, validate, or squeeze meaning out of freeform text.

Function calling puts a strong "guardrail" on the output format and makes automation simpler.

## Pitfalls and Limitations

Structured output helps, but doesn’t solve everything.

**Common problems:**

- The model might leave out a field, especially if it can’t find that information in the input or the prompt is unclear.
- The model can invent (or "hallucinate") fields or values if it's not sure.
- Returned data types might be inconsistent (a string instead of a number, or vice versa).
- Sometimes the model tries to help and adds extra, unexpected fields.
- Invalid JSON is possible—missing quotes or other formatting errors, especially if the input is complex or the prompt is vague.

API-native "function calling" avoids many of these problems, because the API itself checks the schema. But even function-calling models can still get confused if the definitions are too broad or complex.

## Code Example: Prompting for Structured Output with OpenAI

These are minimal examples for extracting structured data from plain text using the OpenAI API.

### Prompted JSON Output

```python
import openai
import json

openai.api_key = "sk-..."  # Replace with your API key

prompt = """Extract the following information as JSON:
- country (string)
- capital (string)
Text: "The capital of Italy is Rome."
Respond only with valid JSON."""

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": prompt}]
)

output_text = response["choices"][0]["message"]["content"]
structured_output = json.loads(output_text)
print(structured_output)
```

### Using Function Calling (if your model supports it)

```python
import openai

openai.api_key = "sk-..."

functions = [
    {
        "name": "extract_country_capital",
        "parameters": {
            "type": "object",
            "properties": {
                "country": {"type": "string"},
                "capital": {"type": "string"}
            },
            "required": ["country", "capital"]
        }
    }
]

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0613",
    messages=[
        {"role": "user", "content": "The capital of Spain is Madrid."}
    ],
    functions=functions,
    function_call={"name": "extract_country_capital"}
)

# Output is already structured—no need to parse text
print(response["choices"][0]["message"]["function_call"]["arguments"])
```

With function calling, your application gets structured data every time, as defined in your schema.

---

Structured output prompting—especially when paired with function calling—transforms LLMs from chat partners into robust components for automation. This turns unstructured language into predictable, reliable data your code can use directly.

---

## Key Takeaways

- Structured output makes LLM-generated data predictable and machine-readable.
- Prompting for JSON allows reliable extraction of information from model responses.
- Effective prompts list required fields, specify formats, and can include sample outputs.
- Function calling lets models return data directly as structured arguments, not text.
- API-native function calling avoids many common issues like missing or extra fields.

## Try It Yourself

Write a prompt for a language model that requests a summary of a news article as a JSON object with the fields: 'headline', 'entities', and 'summary'. Try it in a playground or API. Check whether the response follows your structure and contains the requested fields.

## Further Resources

- 📄 [Function Calling — Complete Guide (Prompt Engineering Tutorial — Toolliyo Academy)](https://toolliyo.com/tutorials/prompt-engineering-tutorial-function-calling-complete-guide)
- 📘 [Structured model outputs | OpenAI API Guide](https://developers.openai.com/api/docs/guides/structured-outputs)
- 📄 [Structured Outputs and Function Calling (Agentic AI LLM Agents Academy)](https://szaher.github.io/agentic/lesson/2/03-structured-outputs/)

---

**Coming up on Day 60:** Introduction to LLM APIs (OpenAI, Anthropic, and Others)