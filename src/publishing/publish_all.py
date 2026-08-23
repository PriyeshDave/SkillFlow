"""
Runs the publish sequence for one approved lesson: moves it from
content/drafts/ to content/published/, then publishes to dev.to and
LinkedIn.

Usage:
    python -m src.publishing.publish_all content/drafts/day-001-example.md
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import frontmatter

from src.publishing.devto_publish import publish_to_devto
from src.publishing.linkedin_copy import generate_linkedin_copy
from src.publishing.linkedin_publish import publish_to_linkedin
from src.utils.settings import get_roadmap, get_settings
from src.utils.storage import REPO_ROOT, get_previous_post_summary


def move_to_published(draft_path: str) -> Path:
    settings = get_settings()
    published_dir = REPO_ROOT / settings["published_dir"]
    published_dir.mkdir(parents=True, exist_ok=True)

    src = Path(draft_path)
    dest = published_dir / src.name
    shutil.move(str(src), str(dest))

    with open(dest) as f:
        post = frontmatter.load(f)
    post["status"] = "published"
    with open(dest, "w") as f:
        frontmatter.dump(post, f)

    return dest


def publish_all(draft_path: str, dry_run: bool = False) -> None:
    published_path = move_to_published(draft_path)
    with open(published_path) as f:
        post = frontmatter.load(f)

    if dry_run:
        print(f"[publish_all] DRY RUN — would publish {published_path} to dev.to + LinkedIn.")
        return

    # 1. dev.to
    devto_result = publish_to_devto(str(published_path))
    devto_url = devto_result.get("url", "")

    # 2. LinkedIn
    settings = get_settings()
    day_number = post.get("day")
    total_days = len(get_roadmap())
    series_name = settings["publishing"]["devto"]["series_name"]
    previous_summary = get_previous_post_summary(day_number)

    linkedin_text = generate_linkedin_copy(
        post_markdown=post.content,
        day_number=day_number,
        total_days=total_days,
        topic_title=post.get("topic_title"),
        phase=post.get("phase"),
        today_summary=post.get("recap_summary"),
        previous_summary=previous_summary,
        series_name=series_name,
    )
    publish_to_linkedin(
        linkedin_text,
        devto_url,
        title=post.get("title"),
        description=f"Day {day_number} of {total_days} -- {series_name}",
    )

    print(f"[publish_all] Done. Live at: {devto_url}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("draft_path")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    publish_all(args.draft_path, dry_run=args.dry_run)
