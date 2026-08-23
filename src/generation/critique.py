from __future__ import annotations

import json

from src.utils.claude_client import call_claude
from src.utils.settings import get_settings

CRITIQUE_SYSTEM = """You are a skeptical technical editor reviewing one
day's lesson in a teaching series before publication. You are looking for
failure modes specific to educational content:

1. Jargon used before it's defined, or assumed prior knowledge the reader
   wouldn't have yet at this point in the series
2. Technically inaccurate explanations, even if they sound confident
3. Code that looks plausible but wouldn't actually run, or that doesn't
   match what the surrounding text claims it does
4. A concept explained so abstractly that a beginner couldn't picture what's
   actually happening
5. Padding or repetition to hit a word count
6. Generic "AI-sounding" filler sentences with no teaching value

Respond ONLY with valid JSON, no markdown fences:
{
  "passes_quality_bar": true/false,
  "issues": ["specific issue 1", "specific issue 2"],
  "revision_instructions": "concrete instructions for fixing the issues, or empty string if it passes"
}
"""


def critique_draft(draft_markdown: str) -> dict:
    settings = get_settings()
    user_prompt = f"""Quality bar for this lesson:
- Word count between {settings['min_word_count']} and {settings['max_word_count']}
- Every technical term is defined before or at first use
- Any code included must be correct and match what the text says it does

Draft to review:
---
{draft_markdown}
---

Evaluate against the quality bar and the failure modes listed."""

    raw = call_claude(system=CRITIQUE_SYSTEM, user=user_prompt, max_tokens=1000)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Critique stage returned non-JSON:\n{raw}") from e


REVISION_SYSTEM = """You are revising one day's lesson in a teaching series
based on editor feedback. Apply the revision instructions precisely.
Preserve everything that already works well. Output ONLY the revised
Markdown lesson body, no preamble."""


def revise_draft(draft_markdown: str, revision_instructions: str) -> str:
    user_prompt = f"""Revision instructions:
{revision_instructions}

Original draft:
---
{draft_markdown}
---

Produce the revised lesson body."""
    return call_claude(system=REVISION_SYSTEM, user=user_prompt, max_tokens=5000)
