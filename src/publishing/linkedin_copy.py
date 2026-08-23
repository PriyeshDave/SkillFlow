"""
Generates a LinkedIn-native post promoting one day's lesson.

Deliberately fully deterministic -- no LLM call at all. Every piece of
content used here was already generated (and trusted) once by the main
pipeline: today's recap_summary and the previous day's recap_summary both
already exist, and the Key Takeaways bullets are already sitting in the
lesson's own Markdown body.

CRITICAL -- LinkedIn's 'little' text escaping: the "commentary" field on
LinkedIn's Posts API is not plain text. It's a small markup language
called 'little', and a specific set of characters are reserved for its
syntax (mentions, hashtags, templates): _ | ( ) [ ] { } @ # * ~ < > \\
If any of these appear unescaped, LinkedIn's parser can silently stop
rendering content from that exact point onward -- this was confirmed as
the actual cause of posts truncating mid-sentence right before a literal
"(" character, with the API call still succeeding (a valid post URN
returned) despite the visible content being cut. Every reserved character
in generated text MUST be backslash-escaped before being sent. Hashtags
specifically should use LinkedIn's HashtagTemplate syntax
({hashtag|#|TagName}) rather than raw "#TagName" text, since "#" is
itself one of the reserved characters.
"""
from __future__ import annotations

import re

# Reserved characters for LinkedIn's 'little' text format. Must be
# backslash-escaped anywhere they appear as literal text in "commentary".
_LITTLE_RESERVED_CHARS = ["\\", "_", "|", "(", ")", "[", "]", "{", "}", "@", "#", "*", "~", "<", ">"]


def _escape_little_text(text: str) -> str:
    """Backslash-escape every LinkedIn 'little'-format reserved character."""
    return "".join(f"\\{ch}" if ch in _LITTLE_RESERVED_CHARS else ch for ch in text)


def _hashtag_template(tag: str) -> str:
    """
    Renders a hashtag using LinkedIn's HashtagTemplate syntax
    ({hashtag|#|TagName}) instead of raw "#TagName" text -- required
    because "#" is a reserved 'little'-format character. `tag` should be
    passed WITHOUT the leading "#".
    """
    return f"{{hashtag|#|{tag}}}"


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
    """'Phase 1 — NLP Foundations' -> 'NLPFoundations' (no leading #; preserves acronyms)"""
    if "—" not in phase:
        return None
    name_part = phase.split("—", 1)[1].strip()
    skip_words = {"the", "a", "an", "of", "and", "to", "in", "for"}
    words = [w for w in re.split(r"[^A-Za-z0-9]+", name_part) if w and w.lower() not in skip_words]
    if not words:
        return None
    parts = [w if w.isupper() else w.capitalize() for w in words]
    return "".join(parts)


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
    header = _to_bold_unicode(
        _escape_little_text(f"Day {day_str}/{total_days}: {topic_title}")
    )

    lines = [header, ""]

    if previous_summary:
        lines.append(f"📌 Yesterday's Recap: {_escape_little_text(previous_summary)}")
        lines.append("")

    lines.append(f"📖 Today's Lesson: {_escape_little_text(today_summary)}")
    lines.append("")

    takeaways = _extract_key_takeaways(post_markdown)
    if takeaways:
        lines.append("💡 Key Takeaways:")
        lines.extend(f"- {_escape_little_text(t)}" for t in takeaways)
        lines.append("")

    hashtag_names = ["AIEngineering", "100DaysOfCode"]
    phase_tag = _phase_hashtag(phase)
    if phase_tag:
        hashtag_names.append(phase_tag)
    lines.append(" ".join(_hashtag_template(t) for t in hashtag_names))
    lines.append("")

    lines.append(
        _escape_little_text(
            f'✨ Curated by {product_name} — your AI-powered tutor automation '
            f'behind "{series_name}," turning a 105-day AI curriculum into one '
            f"lesson a day."
        )
    )
    lines.append("")
    lines.append("Happy Learning! 🎉")
    lines.append("")
    lines.append("Feel free to comment your doubts below 👇")
    lines.append("")

    # URL placed as the absolute LAST line, deliberately. No content.article
    # card is attached anymore (see linkedin_publish.py docstring for why),
    # so this is the only way readers reach the article. Since we can't yet
    # rule out that a bare URL also truncates trailing text on a plain-text
    # post (only confirmed it does when paired with a duplicate content.article
    # card), keeping it last means nothing is lost even in the worst case.
    lines.append(f"🔗 Read the full lesson: {devto_url}")

    return "\n".join(lines)