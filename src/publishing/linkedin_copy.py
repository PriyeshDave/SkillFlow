"""
Generates a LinkedIn-native post promoting one day's lesson.

Deliberately fully deterministic -- no LLM call at all. Every piece of
content used here was already generated (and trusted) once by the main
pipeline: today's recap_summary and the previous day's recap_summary both
already exist, and the Key Takeaways bullets are already sitting in the
lesson's own Markdown body.

Note on formatting history (each step backed by direct evidence from live
posts, not guesswork):
1. Backslash-escaping reserved characters + a "{hashtag|#|Tag}" template
   syntax: wrong. Both showed up as literal, broken text on the live post.
2. Removing all of that: posts still truncated, cut off exactly before an
   unescaped "(" character -- twice, on two different posts, right before
   "(NLP)" and "(NER)". This is the actual trigger.
3. Backslash-escaping those characters DID stop the truncation (full post
   went through) but left ugly visible backslashes in the text, since
   LinkedIn parses the escape (preventing truncation) but doesn't strip it
   for display.
4. This version: replace reserved characters with visually near-identical
   Unicode "fullwidth" look-alikes (e.g. the fullwidth parenthesis "（"
   instead of ASCII "("). These are different codepoints entirely, so
   LinkedIn's parser never treats them as reserved trigger characters, and
   there's no visible escape artifact since they render as ordinary-
   looking (very slightly wider) punctuation.

Applied only to free-text fields (recap summaries, takeaways, topic
titles) -- never to the deliberately-constructed "#hashtag" line, since
those need to stay literal ASCII "#" for LinkedIn to auto-link them.
"""
from __future__ import annotations

import re

# ASCII reserved character -> visually near-identical Unicode fullwidth
# look-alike. Different codepoints entirely, so LinkedIn's parser doesn't
# treat them as special, but they read as normal punctuation to a human.
_SAFE_CHAR_MAP = {
    "(": "\uFF08", ")": "\uFF09",
    "[": "\uFF3B", "]": "\uFF3D",
    "{": "\uFF5B", "}": "\uFF5D",
    "@": "\uFF20", "*": "\uFF0A", "~": "\uFF5E",
    "<": "\uFF1C", ">": "\uFF1E",
    "_": "\uFF3F", "|": "\uFF5C", "\\": "\uFF3C",
    # Deliberately NOT mapping "#" here -- that character is only ever
    # used in the intentionally-built hashtag line, which must stay
    # literal ASCII "#" for LinkedIn to auto-link it. See _safe_text().
}


def _safe_text(text: str) -> str:
    """Swaps reserved characters for safe look-alikes in free-text fields."""
    return "".join(_SAFE_CHAR_MAP.get(ch, ch) for ch in text)


def _extract_key_takeaways(post_markdown: str) -> list[str]:
    """
    Pulls the bullet list out of the lesson's own '## Key Takeaways'
    section (always present -- assemble_final_post in pipeline.py
    guarantees it). Avoids re-deriving takeaways via a second LLM call
    when they already exist, correct, in the content itself.
    """
    lines = post_markdown.split("\n")
    bullets = []
    in_section = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("## Key Takeaways"):
            in_section = True
            continue
        if in_section:
            if stripped.startswith("##") or stripped == "---":
                break
            if stripped.startswith("- "):
                bullets.append(stripped[2:].strip())
    return bullets


def _phase_hashtag(phase: str) -> str | None:
    """'Phase 1 — NLP Foundations' -> '#NLPFoundations' (preserves acronyms)"""
    if "—" not in phase:
        return None
    name_part = phase.split("—", 1)[1].strip()
    skip_words = {"the", "a", "an", "of", "and", "to", "in", "for"}
    words = [w for w in re.split(r"[^A-Za-z0-9]+", name_part) if w and w.lower() not in skip_words]
    if not words:
        return None
    parts = [w if w.isupper() else w.capitalize() for w in words]
    return "#" + "".join(parts)


def generate_linkedin_copy(
    post_markdown: str,
    day_number: int,
    total_days: int,
    topic_title: str,
    phase: str,
    today_summary: str,
    previous_summary: str | None,
    devto_url: str,
    series_name: str = "Zero to Agentic In 105 Days",
    product_name: str = "SkillFlow",
) -> str:
    day_width = len(str(total_days))
    day_str = str(day_number).zfill(day_width)
    header = f"\U0001F4C5 Day {day_str}/{total_days}: {_safe_text(topic_title)}"

    lines = [header, ""]

    if previous_summary:
        lines.append(f"\U0001F4CC Yesterday's Recap: {_safe_text(previous_summary)}")
        lines.append("")

    lines.append(f"\U0001F4D6 Today's Lesson: {_safe_text(today_summary)}")
    lines.append("")

    takeaways = _extract_key_takeaways(post_markdown)
    if takeaways:
        lines.append("\U0001F4A1 Key Takeaways:")
        lines.extend(f"- {_safe_text(t)}" for t in takeaways)
        lines.append("")

    hashtags = ["#AIEngineering", "#100DaysOfCode"]
    phase_tag = _phase_hashtag(phase)
    if phase_tag:
        hashtags.append(phase_tag)
    lines.append(" ".join(hashtags))
    lines.append("")

    lines.append(
        _safe_text(
            f'\u2728 Curated by {product_name} \u2014 your AI-powered tutor automation '
            f'behind "{series_name}," turning a 105-day AI curriculum into one '
            f"lesson a day."
        )
    )
    lines.append("")
    lines.append("Happy Learning! \U0001F389")
    lines.append("")
    lines.append("Feel free to comment your doubts below \U0001F447")
    lines.append("")

    # URL placed as the absolute LAST line, deliberately. No content.article
    # card is attached (see linkedin_publish.py), so this is the only way
    # readers reach the article. Keeping it last means nothing is lost even
    # if any future platform quirk affects text after a URL.
    lines.append(f"\U0001F517 Read the full lesson: {devto_url}")

    return "\n".join(lines)