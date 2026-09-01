"""
Generates a LinkedIn-native post promoting one day's lesson.

Deliberately fully deterministic -- no LLM call at all. Every piece of
content used here was already generated (and trusted) once by the main
pipeline: today's recap_summary and the previous day's recap_summary both
already exist, and the Key Takeaways bullets are already sitting in the
lesson's own Markdown body.

Note on formatting history:
1. An earlier version backslash-escaped punctuation and used a
   "{hashtag|#|Tag}" template syntax, based on a developer forum report
   about LinkedIn's internal 'little' text format. That was wrong: both
   showed up as literal, broken text on the live post.
2. After removing those, posts still truncated -- but debug logs proved
   the full, correct text was being sent and accepted by the API (a valid
   post URN was returned every time). The one thing present in every
   single failing post: the bold-Unicode header (the "Mathematical Bold"
   character trick used to fake bold text, since LinkedIn has no native
   bold). That Unicode block is also a well-known signal used heavily by
   spam/growth-hacking accounts, and is a plausible trigger for LinkedIn's
   spam/quality filters to suppress the rest of a post's visible content
   even though the API accepted it. This version drops that trick
   entirely and uses plain text with an emoji marker for emphasis instead.
"""
from __future__ import annotations

import re


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
    # Plain text header, no Unicode bold trick -- see module docstring.
    header = f"\U0001F4C5 Day {day_str}/{total_days}: {topic_title}"

    lines = [header, ""]

    if previous_summary:
        lines.append(f"📌 Yesterday's Recap: {previous_summary}")
        lines.append("")

    lines.append(f"📖 Today's Lesson: {today_summary}")
    lines.append("")

    takeaways = _extract_key_takeaways(post_markdown)
    if takeaways:
        lines.append("💡 Key Takeaways:")
        lines.extend(f"- {t}" for t in takeaways)
        lines.append("")

    hashtags = ["#AIEngineering", "#100DaysOfCode"]
    phase_tag = _phase_hashtag(phase)
    if phase_tag:
        hashtags.append(phase_tag)
    lines.append(" ".join(hashtags))
    lines.append("")

    lines.append(
        f'✨ Curated by {product_name} — your AI-powered tutor automation '
        f'behind "{series_name}," turning a 105-day AI curriculum into one '
        f"lesson a day."
    )
    lines.append("")
    lines.append("Happy Learning! 🎉")
    lines.append("")
    lines.append("Feel free to comment your doubts below 👇")
    lines.append("")

    # URL placed as the absolute LAST line, deliberately. No content.article
    # card is attached (see linkedin_publish.py), so this is the only way
    # readers reach the article. Keeping it last means nothing is lost even
    # if any future platform quirk affects text after a URL.
    lines.append(f"🔗 Read the full lesson: {devto_url}")

    return "\n".join(lines)