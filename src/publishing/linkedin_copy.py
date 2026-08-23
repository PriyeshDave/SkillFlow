"""
Generates a LinkedIn-native post promoting one day's lesson.

Deliberately fully deterministic -- no LLM call at all. Every piece of
content used here was already generated (and trusted) once by the main
pipeline: today's recap_summary and the previous day's recap_summary both
already exist, and the Key Takeaways bullets are already sitting in the
lesson's own Markdown body. Re-summarizing any of this via a second LLM
call would just introduce a new point of failure for no benefit -- the
data we need already exists and is already accurate.

This also carries forward the lesson learned the hard way in BlogMind:
never embed a raw URL in the LinkedIn commentary body that duplicates the
one already attached via content.article.source -- LinkedIn silently
truncates the post at that point. The dev.to link is attached separately
in publish_to_linkedin, never as text here.
"""
from __future__ import annotations

import re


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
    # Preserve words that are already all-caps (acronyms like NLP, AI, LLM,
    # RAG) instead of capitalize()'ing them, which would lowercase the rest
    # of the word ("NLP" -> "Nlp").
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
    series_name: str = "Zero to Agentic",
    product_name: str = "SkillFlow",
) -> str:
    day_width = len(str(total_days))
    day_str = str(day_number).zfill(day_width)
    header = _to_bold_unicode(f"Day {day_str}/{total_days}: {topic_title}")

    lines = [header, ""]

    if previous_summary:
        lines.append(f"📌 Previously (Day {str(day_number - 1).zfill(day_width)}): {previous_summary}")
        lines.append("")

    lines.append(f"📖 Today: {today_summary}")
    lines.append("")

    takeaways = _extract_key_takeaways(post_markdown)
    if takeaways:
        lines.append("Key takeaways:")
        lines.extend(f"- {t}" for t in takeaways)
        lines.append("")

    hashtags = ["#AIEngineering", "#100DaysOfCode"]
    phase_tag = _phase_hashtag(phase)
    if phase_tag:
        hashtags.append(phase_tag)
    lines.append(" ".join(hashtags))
    lines.append("")

    lines.append(
        f'🤖 Created by {product_name} — the AI-powered tutor automation behind "{series_name}."'
    )

    return "\n".join(lines)
