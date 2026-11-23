---
day: 64
generated_at: '2026-11-23T20:41:01.488654+00:00'
phase: Phase 8 — Working With LLM APIs
recap_summary: Explained how function calling and tools extend large language models
  by enabling them to use external functions for tasks like calculations, live data
  access, or interaction with private databases, detailing the flow, schema definitions,
  error handling, and benefits over pure text generation.
status: pending_review
title: 'Day 64: Function Calling and Tool Use Fundamentals'
topic_title: Function Calling and Tool Use Fundamentals
---

## What Are Function Calls and Tools in LLMs?

Large language models (LLMs) like OpenAI's GPT, Anthropic's Claude, or Google's Gemini can generate impressive text, but they are limited to what they've "learned" during training. They can't access new information, do complex calculations, or use private databases on their own.

**Function calling** lets an LLM go beyond pure text. Instead of just answering, the model can propose to call a real function in your code, use an external tool, or query an API.

Imagine an expert assistant who knows when to pull out a calculator, check a website, or fetch data from a database to get the right answer. The LLM asks your software to do something it can't, then uses that result to answer the user.

Main terms:

- **Tool:** Any external resource or software function you connect to the LLM. Tools let the LLM fetch live information, trigger actions, or access private data it doesn't know itself.  
  Example: a function that checks the weather, or reads your company’s database.

- **Schema:** A formal description of a tool or function. The schema tells the LLM the function’s name, what it does, and what inputs (parameters) it expects. The schema acts as a contract for how the LLM should call the tool.

- **Parameter:** Each named piece of information a function needs to run. In a function that adds two numbers, the parameters might be "a" and "b".

- **Function call:** The model’s request for which function to use, and with which arguments. Example: `add_numbers(a=5, b=7)`.

## Why Can't LLMs Just Do Everything in Text?

LLMs are not calculators or search engines. Their strength is matching patterns in data they saw during training.

- **Math and Logic:** LLMs mimic human math and logic, but guesses can go wrong. Ask for "123456 × 789" and you might get an answer in the right format, but not the actual product.
- **Live Data:** LLMs cannot see real-time information. Today's weather, current events, or edits to a document after their last training are invisible.
- **Private Data:** Anything outside their training data—your database, your company emails, a secure server—is completely out of reach.

If you want accurate, dynamic answers in your application, relying on pure LLM text isn’t enough.

## How Function Calling Works: The Basic Flow

Here's how the collaboration happens between user, LLM, and backend code:

**Step 1 (User ➔ LLM):**
A user asks a question, such as “What is 257 plus 805?”

**Step 2 (Inside the LLM):**
The LLM sees which tools (functions) you’ve made available. It chooses a function and proposes a call, for example `add_numbers` with `a=257`, `b=805`. The LLM doesn't run the code—it just makes a structured proposal.

**Step 3 (LLM ➔ Backend):**
Your software receives this function call request.

**Step 4 (In your backend):**
Your code parses the proposal, extracts the arguments, and actually runs the function—here, it computes 257 + 805.

**Step 5 (Backend ➔ LLM):**
You feed the result back into the LLM so it can craft a conversational response based on the real answer.

**Step 6 (LLM ➔ User):**
The LLM tells the user: “The sum of 257 and 805 is 1062.”

Summary of data flow:

```
User → [LLM proposes function_call] → Backend (executes function) → LLM (uses result) → User
```

The LLM never runs real code itself—your application always controls what runs.

*Note: The exact API field names and response format may differ depending on your OpenAI SDK version or platform. Always check the documentation for your setup.*

## A Simple Example: LLM Calculation with Function Calling

Let’s walk through a minimal example, using OpenAI’s API.

We’ll make a simple add function, define the schema, let the LLM propose a call, and then run the function with the provided arguments. This code assumes you use Python and OpenAI's SDK.

```python
import openai
import json

# Define the tool: a function the LLM is allowed to call
def add_numbers(a, b):
    return a + b

# Define the schema that the model will see
functions = [
    {
        "name": "add_numbers",
        "description": "Add two numbers together.",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "First number"},
                "b": {"type": "number", "description": "Second number"}
            },
            "required": ["a", "b"]
        }
    }
]

# Simulate a user asking the model a question
messages = [
    {"role": "user", "content": "What is 257 plus 805?"}
]

# Send all this to the model, including function definition
response = openai.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=messages,
    functions=functions,
    function_call="auto"
)

# Function call proposals can vary by SDK version—always verify field access
function_call = response.choices[0].message.function_call

if function_call and function_call.name == "add_numbers":
    try:
        # Arguments usually arrive as a JSON string, so parse them
        args = json.loads(function_call.arguments)

        a = args.get("a")
        b = args.get("b")

        if a is None or b is None:
            raise ValueError("Missing required arguments 'a' or 'b'.")
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            raise TypeError("Arguments 'a' and 'b' must be numbers.")

        result = add_numbers(a, b)

        # Add the function call and its result to conversation history
        messages.append(response.choices[0].message)
        messages.append({
            "role": "function",
            "name": "add_numbers",
            "content": str(result)
        })

        # Let the LLM generate a user-friendly answer
        final_response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        print(final_response.choices[0].message.content)

    except (json.JSONDecodeError, TypeError, ValueError) as e:
        print(f"Error handling function call arguments: {e}")

else:
    print("The model did not try to use the provided function.")
```

Highlights:
- Functions and parameters are defined up front and shared with the LLM as a schema.
- The LLM proposes a function call with arguments. Your code parses these arguments from a JSON string.
- Your code runs the function securely, not the model.
- The function result is sent back through the LLM, enabling it to respond accurately in plain language.
- Basic error handling ensures that unexpected input does not crash your application.

## Why Function Calling Unlocks Powerful Applications

LLMs trained only on data are static. When you connect tools and let the LLM propose their use, the model can:

- Access **live information**—news headlines, weather, live stock prices, and more.
- Work **securely with your private or sensitive data**—on your terms, within your infrastructure.
- Trigger **actions and automations**—from sending emails to booking meetings or filing tickets.

Concrete examples:
- Scheduling a calendar event using your preferred calendar service.
- Looking up a country’s GDP from a trusted online database.
- Starting a password reset process in your internal system, when instructed.
- Searching document archives and returning or summarizing results for the user.

Function calling transforms LLMs from basic chatbots into assistant-like orchestrators, guiding complex tasks by proposing actions your code can run safely.

## Common Pitfalls and Mental Models

Keep these principles clear:

- **LLMs never execute code directly.** They make structured requests; all real code runs by your rules, in your environment.
- **Schema matters.** If you don’t clearly define the tool’s inputs and outputs, the LLM will guess—and sometimes guess wrong. Be explicit in your schema.
- **No memory between turns, unless you provide it.** The LLM doesn't “know” the result of a function call later unless you include previous context in the conversation.
- **Every tool must be wired up.** If it’s not actually implemented and connected in your code, the LLM can’t use it—even if it sees the schema.

Think of the LLM as a conversational orchestrator, not an all-knowing agent. Its real talent is deciding which tool to use, with what parameters, and wrapping up outputs in fluent text. You control the actual actions, keeping things safe and predictable.

Function calling opens up a partnership: the LLM handles communication and decision-making, your code does the real work. This is what enables safe, adaptable, and genuinely useful AI integrations.

---

## Key Takeaways

- LLMs use function calling to propose external actions beyond text generation.
- Schemas define available functions, expected parameters, and clarify tool usage for the model.
- The LLM never executes real code—your application interprets and runs function calls.
- Accurate results require properly wired and explicitly defined tools with clear error handling.
- Function calling allows LLMs to access live information, private data, and perform secure automations.

## Try It Yourself

Write a simple program where an LLM receives a math question and proposes a function call rather than answering directly in text. Let your code execute the proposed function (like multiplying two numbers), and then feed the result back into the model for a user-friendly reply. Experiment by adapting the function to perform a different computation or fetch information from another simple source.

## Further Resources

- 🎥 [How to Connect an LLM to Python Functions (Tool Calling Tutorial)](https://www.youtube.com/watch?v=liV0bfZ5Wu0)
- 🎥 [Function Calling Explained | How ChatGPT Uses APIs & External Tools | AI Agents Tutorial](https://www.youtube.com/watch?v=d3lSE34jDrg)
- 📘 [Function calling | OpenAI API](https://developers.openai.com/api/docs/guides/function-calling)
- 📄 [Working with LLM APIs (includes function calling & structured outputs)](https://skill-maps.com/skills/llms-and-rag/llm-apis)
- 📄 [Function Calling Basics (LLM Agents Academy)](https://szaher.github.io/agentic/lesson/3/02-function-calling-basics/)

---

**Coming up on Day 65:** Structured Outputs and JSON Schema Enforcement