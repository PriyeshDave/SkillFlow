"""
Generates the structural pieces that make this a *series* rather than a
pile of independent posts: a short recap summary (stored in frontmatter for
tomorrow's pipeline to read back), key takeaways, the Try It Yourself
exercise text, and a teaser for the next day's topic.

Design note: this content is generated as structured JSON and then
assembled deterministically by pipeline.py, rather than trusting the LLM to
reliably include a "coming up next" section in free text. That approach
failed for BlogMind's LinkedIn posts (the model would drop required
sections despite explicit instructions) -- this project builds on that
lesson from day one instead of relearning it.
"""
from __future__ import annotations

import json

from src.utils.claude_client import call_claude

CONTINUITY_SYSTEM = """You extract the structural wrap-up pieces for one
day's lesson in a teaching series.

Respond ONLY with valid JSON, no markdown fences:
{
  "recap_summary": "1-2 sentences summarizing what THIS lesson taught, \
written so it can be used verbatim as 'Yesterday we covered...' in \
tomorrow's post. Be specific, not vague.",
  "key_takeaways": [
    "3-5 short, specific, standalone takeaways from this lesson, each \
under 20 words"
  ],
  "exercise_text": "A short, concrete 'Try It Yourself' exercise (2-4 \
sentences) the reader can do in 10-15 minutes to reinforce this lesson."
}"""


def generate_continuity(draft_markdown: str, planned_exercise: str) -> dict:
    user_prompt = f"""Today's lesson content:
---
{draft_markdown}
---

Planned exercise idea (use as a starting point, refine as needed): {planned_exercise}

Extract the recap summary, key takeaways, and exercise text as instructed."""

    raw = call_claude(system=CONTINUITY_SYSTEM, user=user_prompt, max_tokens=800)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Continuity stage returned non-JSON:\n{raw}") from e
