from __future__ import annotations

from src.generation.voice import VOICE_GUIDE
from src.utils.claude_client import call_claude_with_web_search

DRAFT_SYSTEM = f"""You are writing one day's lesson in a 105-day teaching
series on NLP, GenAI, and Agentic AI, for engineers learning it from
scratch.
{VOICE_GUIDE}

You have access to a web_search tool. Use it to verify any specific claim,
library API detail, or "current state of the field" statement before
writing it as fact. If you can't verify something, don't state it as a
specific fact.

Write ONLY the main teaching content in Markdown, following the outline's
sections. Do NOT include a title, a "Day N" header, a recap of the previous
day, a "key takeaways" section, an exercise section, or a "coming up next"
section -- all of those are added separately by the pipeline. Just write
the lesson body itself.

Include the planned code example as a real, correct, minimal code block if
one was planned."""


def generate_draft(today_entry: dict, outline: dict) -> str:
    sections_block = "\n".join(
        f"- {s['heading']}: {s['goal']}" for s in outline["sections"]
    )

    user_prompt = f"""Day {today_entry['day']}: {today_entry['title']}
Phase: {today_entry['phase']}

Section plan:
{sections_block}

Planned code example: {outline['planned_code_example'] or '(none for this topic)'}

Write the full lesson body now."""

    return call_claude_with_web_search(system=DRAFT_SYSTEM, user=user_prompt, max_tokens=5000)
