"""
Publishes a text post to a personal LinkedIn profile using the official
LinkedIn Posts API (part of the Community Management API).

Requires: w_member_social scope, an access token, and the author's person
URN. See linkedin_oauth_helper.py for one-time setup.

Endpoint note: LinkedIn's current Posts API lives at /rest/posts. The older
/v2/posts and /v2/ugcPosts endpoints are from a previous API generation and
are not interchangeable with this one.

IMPORTANT -- no content.article attached: earlier versions attached the
dev.to link via content.article (a rich preview card). This reliably
truncated the visible post text after just 1-2 lines, regardless of
whether the URL also appeared as raw text in the commentary body -- so the
truncation is not specifically about duplicate URLs, it happens whenever
an article card is attached at all. LinkedIn's own docs note that article
posts should include a thumbnail, title, AND description; we were only
ever setting title/description, never thumbnail, which is a plausible
cause of the broken rendering. Rather than chase that further, this
version publishes plain text with no attached card at all, which
reliably renders in full. The tradeoff: no automatic link preview. If you
want the dev.to link clickable from the post, the standard LinkedIn
practice is to add it as the first comment immediately after publishing
(requires the separate "Community Management API" product approval on
your Developer App) -- ask if you want that built out.
"""
from __future__ import annotations

import os

import requests

LINKEDIN_POSTS_API = "https://api.linkedin.com/rest/posts"
LINKEDIN_VERSION = "202601"  # LinkedIn API version header (YYYYMM), bump periodically


def publish_to_linkedin(
    text: str,
    article_url: str | None = None,
    title: str | None = None,
    description: str | None = None,
    visibility: str = "PUBLIC",
) -> dict:
    access_token = os.environ.get("LINKEDIN_ACCESS_TOKEN")
    person_urn = os.environ.get("LINKEDIN_PERSON_URN")
    if not access_token or not person_urn:
        raise RuntimeError("LINKEDIN_ACCESS_TOKEN and LINKEDIN_PERSON_URN must be set.")

    payload = {
        "author": person_urn,
        "commentary": text,
        "visibility": visibility,
        "distribution": {
            "feedDistribution": "MAIN_FEED",
            "targetEntities": [],
            "thirdPartyDistributionChannels": [],
        },
        "lifecycleState": "PUBLISHED",
        "isReshareDisabledByAuthor": False,
    }
    # Deliberately no "content" key -- see module docstring. article_url,
    # title, description are accepted but unused for now, kept so callers
    # don't need to change while this is being worked out.

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "X-Restli-Protocol-Version": "2.0.0",
        "Linkedin-Version": LINKEDIN_VERSION,
    }

    resp = requests.post(LINKEDIN_POSTS_API, headers=headers, json=payload, timeout=30)

    if not resp.ok:
        print(f"[linkedin] Request failed ({resp.status_code}): {resp.text}")
    resp.raise_for_status()

    post_id = resp.headers.get("x-restli-id", "")
    print(f"[linkedin] Published: {post_id}")
    return {"post_id": post_id, "status_code": resp.status_code}


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python -m src.publishing.linkedin_publish <text_file> <article_url>")
        sys.exit(1)
    with open(sys.argv[1]) as f:
        body_text = f.read()
    publish_to_linkedin(body_text, sys.argv[2])