from __future__ import annotations

import json
from pathlib import Path

import frontmatter

from src.utils.settings import get_roadmap, get_settings

REPO_ROOT = Path(__file__).resolve().parents[2]


def _state_path() -> Path:
    return REPO_ROOT / get_settings()["state_file"]


def load_state() -> dict:
    path = _state_path()
    if not path.exists():
        return {"next_day_index": 0}  # 0-indexed into the roadmap list
    with open(path) as f:
        return json.load(f)


def save_state(state: dict) -> None:
    path = _state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(state, f, indent=2)


def next_roadmap_entry(state: dict) -> tuple[dict | None, dict | None]:
    """
    Returns (today_entry, next_entry) based on state["next_day_index"].
    today_entry is None if the roadmap is exhausted (series complete).
    next_entry is None if today is the final day (no upcoming day to preview).
    Does NOT advance state or persist -- caller does that after a
    successful generation.
    """
    roadmap = get_roadmap()
    idx = state["next_day_index"]

    if idx >= len(roadmap):
        return None, None

    today_entry = roadmap[idx]
    next_entry = roadmap[idx + 1] if idx + 1 < len(roadmap) else None
    return today_entry, next_entry


def get_previous_post_summary(day_number: int) -> str | None:
    """
    Reads the recap_summary frontmatter field from the previous day's
    lesson (day_number - 1), used to build "Previously on Day N-1..." recap
    openings. Returns None for Day 1 (no previous post).

    Checks content/published/ first (the reviewed, final version), then
    falls back to content/drafts/ if the previous day hasn't been merged
    yet. Day-index advances at generation time regardless of review speed
    (see pipeline.py), so if PR review lags, the previous day's lesson may
    still be sitting in drafts/ rather than published/ -- continuity should
    still work off what was actually written, even pending review, rather
    than silently dropping the recap.
    """
    if day_number <= 1:
        return None

    settings = get_settings()
    prefix = f"day-{day_number - 1:03d}-"

    for dir_key in ("published_dir", "drafts_dir"):
        directory = REPO_ROOT / settings[dir_key]
        if not directory.exists():
            continue
        matches = list(directory.glob(f"{prefix}*.md"))
        if matches:
            with open(matches[0]) as f:
                post = frontmatter.load(f)
            return post.get("recap_summary")

    return None


def slugify(title: str) -> str:
    keep = [c.lower() if c.isalnum() else "-" for c in title]
    slug = "".join(keep)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-")[:80]


def draft_path(day_number: int, title: str) -> Path:
    slug = slugify(title)
    drafts_dir = REPO_ROOT / get_settings()["drafts_dir"]
    drafts_dir.mkdir(parents=True, exist_ok=True)
    return drafts_dir / f"day-{day_number:03d}-{slug}.md"
