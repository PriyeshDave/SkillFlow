"""
Publishes a Markdown lesson to dev.to via its public API.

Uses dev.to's native "series" field, which groups posts together and gives
readers automatic "Part X" navigation on the platform itself -- a good
complement to the recap/preview text baked into each post by the pipeline.
"""
from __future__ import annotations

import os

import frontmatter
import requests

from src.utils.settings import get_settings

DEVTO_API = "https://dev.to/api/articles"


def publish_to_devto(markdown_path: str) -> dict:
    api_key = os.environ.get("DEVTO_API_KEY")
    if not api_key:
        raise RuntimeError("DEVTO_API_KEY is not set.")

    with open(markdown_path) as f:
        post = frontmatter.load(f)

    settings = get_settings()["publishing"]["devto"]

    payload = {
        "article": {
            "title": post.get("title"),
            "published": settings["publish_status"] == "public",
            "body_markdown": post.content,
            "tags": settings["tags"],
            "series": settings["series_name"],
        }
    }

    resp = requests.post(
        DEVTO_API,
        headers={
            "api-key": api_key,
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=30,
    )

    if not resp.ok:
        print(f"[devto] Request failed ({resp.status_code}): {resp.text}")
    resp.raise_for_status()

    data = resp.json()
    print(f"[devto] Published: {data.get('url')}")
    return data


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python -m src.publishing.devto_publish <markdown_path>")
        sys.exit(1)
    publish_to_devto(sys.argv[1])
