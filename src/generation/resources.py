"""
Generates curated external resource links (videos, docs, articles) for one
day's lesson, so readers have a concrete next step for going deeper on the
topic -- a frequently-requested addition based on reader feedback.

Uses the model's web_search tool so links are grounded in real search
results rather than pulled from memory (which risks confidently wrong
URLs). Even so, LLM-returned URLs can still be stale, moved, or outright
hallucinated -- so every link is independently verified with a real HTTP
request before being included. Any link that doesn't resolve is dropped
rather than shipped into a published post. This section is treated as
optional: generation or validation failures degrade to no section at all,
never to a broken or fabricated link going live.
"""
from __future__ import annotations

import json

import requests

from src.utils.claude_client import call_claude_with_web_search

RESOURCES_SYSTEM = """You find high-quality external resources for a reader
who wants to go deeper on today's lesson topic, in a technical AI-education
series read by engineers. Use web search to find REAL, currently-live
resources -- do not invent URLs from memory, and do not guess at a URL's
existence.

Find 3-5 resources, prioritizing (in this order of value):
- A well-known, reputable YouTube video or conference talk explaining the
  topic well
- Official documentation (from the library/framework/paper's own site)
- One well-regarded blog post, tutorial, or the original paper if directly
  relevant

Respond ONLY with valid JSON, no markdown fences:
{
  "resources": [
    {"title": "exact title of the resource", "url": "https://...", "type": "video|docs|article|paper"}
  ]
}

Only include a resource if you found it via search and are confident the
URL is real and currently live. Fewer, verified resources are better than
more, uncertain ones. If you cannot confidently find good resources for
this specific topic, return an empty resources list rather than guessing."""


def _url_is_live(url: str, timeout: float = 6.0) -> bool:
    """
    Independently verifies a URL actually resolves, since LLM-returned
    links -- even web-search-grounded ones -- can still be stale, moved,
    or hallucinated. Tries a cheap HEAD request first; some servers reject
    HEAD outright, so a GET is used as a fallback before giving up.
    """
    headers = {"User-Agent": "Mozilla/5.0 (compatible; SkillFlowBot/1.0)"}
    try:
        resp = requests.head(url, allow_redirects=True, timeout=timeout, headers=headers)
        if resp.status_code < 400:
            return True
    except requests.RequestException:
        pass

    try:
        resp = requests.get(url, allow_redirects=True, timeout=timeout, headers=headers, stream=True)
        return resp.status_code < 400
    except requests.RequestException:
        return False


def generate_resources(today_entry: dict) -> list[dict]:
    """
    Returns a list of {"title", "url", "type"} dicts for verified-live
    resources only. Never raises -- returns an empty list if generation or
    validation leaves nothing usable. A missing "Further Resources" section
    is far better than a broken or fabricated link in a published lesson.
    """
    user_prompt = f"""Lesson topic: Day {today_entry['day']} -- {today_entry['title']}
Phase: {today_entry['phase']}

Find resources for this specific topic."""

    try:
        raw = call_claude_with_web_search(system=RESOURCES_SYSTEM, user=user_prompt, max_tokens=900)
        parsed = json.loads(raw)
        candidates = parsed.get("resources", [])
    except Exception as e:
        print(f"[resources] Generation failed, skipping resources section: {e}")
        return []

    verified = []
    for r in candidates:
        url = (r.get("url") or "").strip()
        title = (r.get("title") or "").strip()
        if not url or not title:
            continue
        if _url_is_live(url):
            verified.append(r)
            print(f"[resources] Verified: {title} ({url})")
        else:
            print(f"[resources] Dropping unreachable link: {title} ({url})")

    return verified[:5]
