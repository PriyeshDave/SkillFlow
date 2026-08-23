from __future__ import annotations

import json

from src.generation.voice import VOICE_GUIDE
from src.utils.claude_client import call_claude

OUTLINE_SYSTEM = f"""You are planning one day's lesson in a 105-day teaching
series that takes engineers from NLP fundamentals through GenAI to Agentic
AI, one topic per weekday.
{VOICE_GUIDE}

Respond ONLY with valid JSON, no markdown fences, in this shape:
{{
  "sections": [
    {{"heading": "...", "goal": "what this section must teach/prove"}}
  ],
  "planned_code_example": "description of the code example this lesson will \
include, or empty string if this topic has no natural code example (e.g. a \
history/concepts-only day)",
  "planned_exercise": "a short 'Try It Yourself' exercise idea the reader \
can do in 10-15 minutes to reinforce today's concept"
}}
"""


def generate_outline(today_entry: dict, previous_summary: str | None) -> dict:
    context_block = (
        f"Yesterday's lesson covered: {previous_summary}"
        if previous_summary
        else "This is Day 1 -- the very first lesson in the series, no prior context."
    )

    user_prompt = f"""Today's lesson: Day {today_entry['day']} -- {today_entry['title']}
Phase: {today_entry['phase']}

{context_block}

Plan the outline for today's lesson."""

    raw = call_claude(system=OUTLINE_SYSTEM, user=user_prompt, max_tokens=1000)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Outline stage returned non-JSON:\n{raw}") from e
