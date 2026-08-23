from __future__ import annotations

from src.generation.voice import VOICE_GUIDE
from src.utils.claude_client import call_claude

STYLE_SYSTEM = f"""You are a copy editor doing a final voice pass on one
day's lesson in a teaching series. Do NOT change the technical content,
claims, or code. Only tighten prose to match this voice guide:
{VOICE_GUIDE}

Specifically: cut filler transitions, shorten dense paragraphs, make sure
jargon is genuinely defined in plain language on first use, not just
technically mentioned. Output ONLY the final Markdown lesson body, no
preamble."""


def style_pass(draft_markdown: str) -> str:
    return call_claude(system=STYLE_SYSTEM, user=draft_markdown, max_tokens=5000)
