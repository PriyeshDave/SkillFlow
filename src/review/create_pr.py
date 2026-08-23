"""
Opens a GitHub PR containing a single generated lesson, so review/edit/
approve happens through normal PR review. Merging triggers publish.yml.

Usage:
    python -m src.review.create_pr <path-to-draft.md> <branch-name>
"""
from __future__ import annotations

import subprocess
import sys

import frontmatter


def build_pr_body(post: frontmatter.Post) -> str:
    return f"""## Lesson ready for review

**Day:** {post.get('day')}
**Title:** {post.get('title')}
**Phase:** {post.get('phase')}
**Generated at:** {post.get('generated_at')}

### Review checklist
- [ ] Technically accurate -- no confident-sounding but wrong explanations
- [ ] Jargon defined before first use
- [ ] Code (if present) is correct and matches what the text claims
- [ ] Recap of the previous day reads naturally (if present)
- [ ] "Coming up next" teaser is accurate to tomorrow's actual topic
- [ ] Matches the series' teaching voice (patient, not senior-engineer-only)

Merging this PR triggers `publish.yml`, which publishes to dev.to as part
of *Zero to Agentic* series. Edit the file directly in this PR
before merging if changes are needed.
"""


def create_pr(draft_file: str, branch_name: str) -> None:
    with open(draft_file) as f:
        post = frontmatter.load(f)

    body = build_pr_body(post)
    # post['title'] is already "Day N: <topic>" (set by pipeline.py) --
    # don't prepend "Day N:" again here, or it duplicates.
    title = post.get("title", draft_file)

    subprocess.run(
        [
            "gh", "pr", "create",
            "--title", title,
            "--body", body,
            "--head", branch_name,
            "--base", "main",
            "--label", "lesson-draft",
        ],
        check=True,
    )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python -m src.review.create_pr <draft.md> <branch-name>", file=sys.stderr)
        sys.exit(1)
    create_pr(sys.argv[1], sys.argv[2])