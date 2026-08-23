"""
Generates a short, LinkedIn-native post promoting one day's lesson.

Design notes carried over from BlogMind (both lessons learned the hard way,
applied here from day one instead of relearning them):

1. Structured JSON generation, deterministic Python assembly -- an LLM
   asked to produce fully-formatted free text with mandatory sections
   (hashtags, specific structure) proved unreliable; asking only for
   content and assembling formatting in code is not.

2. No raw URLs in the commentary body. LinkedIn's Posts API silently
   truncates the commentary text at the point a URL appears when that URL
   duplicates the one already attached via content.article -- confirmed by
   LinkedIn's own help docs (a shared link with no text after it gets
   hidden from the share). The dev.to link is attached separately as a
   proper preview card via publish_to_linkedin's content.article.source;
   it is never embedded as text here.
"""
from __future__ import annotations

import json

from src.utils.claude_client import call_claude

LINKEDIN_COPY_SYSTEM = """You extract the content for a LinkedIn post
promoting one day's lesson from "Zero to Agentic," a 105-day series taking
engineers from NLP fundamentals to production AI agents. You are NOT
responsible for final formatting -- just for picking the right content.

Respond ONLY with valid JSON, no markdown fences, in this shape:
{
  "headline": "a punchy, specific headline about TODAY's topic, under 12 \
words -- not the lesson title verbatim, a hook that makes the topic sound \
worth knowing",
  "context_lines": [
    "1-2 sentences explaining what today's lesson actually teaches and why \
it matters -- someone who hasn't clicked anything yet should understand \
the core idea from this alone"
  ],
  "insight_bullets": [
    "3-5 short, specific, standalone takeaways from today's lesson, each \
under 20 words, pulled from the lesson's actual content (including its Key \
Takeaways section if present)"
  ],
  "hashtags": ["#Exactly", "#Three", "#RelevantSpecificHashtags"]
}

Rules:
- Never just tease an isolated stat with no explanation -- the reader must
  understand the topic from headline + context alone.
- hashtags must be specific to today's actual topic, never generic filler
  like #AI or #innovation.
- Voice: educational and welcoming, not salesy. This is a teaching series,
  not an opinion blog -- confidence without hype."""


def _to_bold_unicode(text: str) -> str:
    """Unicode 'Mathematical Bold' mapping -- LinkedIn has no native bold."""
    result = []
    for ch in text:
        if "A" <= ch <= "Z":
            result.append(chr(0x1D400 + (ord(ch) - ord("A"))))
        elif "a" <= ch <= "z":
            result.append(chr(0x1D41A + (ord(ch) - ord("a"))))
        elif "0" <= ch <= "9":
            result.append(chr(0x1D7CE + (ord(ch) - ord("0"))))
        else:
            result.append(ch)
    return "".join(result)


def generate_linkedin_copy(
    post_markdown: str,
    day_number: int,
    total_days: int,
    series_name: str,
) -> str:
    """
    devto_url is deliberately NOT a parameter here -- it's attached to the
    LinkedIn post separately via content.article.source in
    publish_to_linkedin, never as raw text in the commentary body.
    """
    user_prompt = f"""Day {day_number} of {total_days} -- full lesson content:
---
{post_markdown}
---

Extract the headline, context, insight bullets, and hashtags as instructed."""

    raw = call_claude(system=LINKEDIN_COPY_SYSTEM, user=user_prompt, max_tokens=700)

    try:
        parts = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"LinkedIn copy stage returned non-JSON:\n{raw}") from e

    headline = _to_bold_unicode(f"Day {day_number}: {parts['headline']}")
    context = "\n".join(parts["context_lines"])
    bullets = "\n".join(f"- {b}" for b in parts["insight_bullets"])
    hashtags = " ".join(parts["hashtags"][:3])

    post = f"""{headline}

{context}

{bullets}

📚 Day {day_number} of {total_days} in "{series_name}" — a daily series taking engineers from NLP fundamentals to production AI agents.

{hashtags}"""

    return post
