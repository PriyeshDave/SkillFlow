"""
Entry point for lesson generation. Run as:

    python -m src.generation.pipeline

Walks the roadmap sequentially (no topic scoring -- the roadmap IS the
source of truth), generates one day's lesson through
outline -> draft -> critique -> (optional revision) -> style pass, then
generates the recap/takeaways/exercise/next-preview content and assembles
the final Markdown file deterministically. Writes to content/drafts/.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone

import frontmatter

from src.generation.continuity import generate_continuity
from src.generation.critique import critique_draft, revise_draft
from src.generation.outline import generate_outline
from src.generation.draft import generate_draft
from src.generation.resources import generate_resources
from src.generation.style_pass import style_pass
from src.utils.storage import (
    draft_path,
    get_previous_post_summary,
    load_state,
    next_roadmap_entry,
    save_state,
)

MAX_REVISION_ROUNDS = 2


def assemble_final_post(
    today_entry: dict,
    next_entry: dict | None,
    previous_summary: str | None,
    body_markdown: str,
    continuity: dict,
    resources: list[dict],
) -> str:
    """
    Deterministically builds the final lesson structure. Day numbering,
    the recap block, and the next-day teaser are ALWAYS present regardless
    of what the LLM stages produced -- this is assembled in Python, not
    trusted to prompt compliance.
    """
    parts = []

    if previous_summary:
        parts.append(
            f"**Previously, on Day {today_entry['day'] - 1}:** {previous_summary}\n\n---"
        )

    parts.append(body_markdown.strip())

    takeaways_block = "\n".join(f"- {t}" for t in continuity["key_takeaways"])
    parts.append(f"---\n\n## Key Takeaways\n\n{takeaways_block}")

    parts.append(f"## Try It Yourself\n\n{continuity['exercise_text']}")

    if resources:
        icon_by_type = {"video": "\U0001F3A5", "docs": "\U0001F4D8", "article": "\U0001F4C4", "paper": "\U0001F4C4"}
        resource_lines = "\n".join(
            f"- {icon_by_type.get(r.get('type', ''), '\U0001F517')} [{r['title']}]({r['url']})"
            for r in resources
        )
        parts.append(f"## Further Resources\n\n{resource_lines}")

    if next_entry:
        parts.append(
            f"---\n\n**Coming up on Day {next_entry['day']}:** "
            f"{next_entry['title']}"
        )
    else:
        parts.append(
            "---\n\n**That's a wrap.** This was the final lesson in "
            "*Zero to Agentic*. Thank you for following along."
        )

    return "\n\n".join(parts)


def run_pipeline() -> str | None:
    state = load_state()
    today_entry, next_entry = next_roadmap_entry(state)

    if today_entry is None:
        print("[pipeline] Roadmap is complete -- no more days to generate.")
        return None

    day_number = today_entry["day"]
    print(f"[pipeline] Day {day_number}: {today_entry['title']}")

    previous_summary = get_previous_post_summary(day_number)
    if previous_summary:
        print(f"[pipeline] Continuity from Day {day_number - 1} found.")
    elif day_number > 1:
        print(f"[pipeline] WARNING: no previous-day summary found for Day {day_number}. "
              "Recap section will be omitted.")

    print("[pipeline] Generating outline...")
    outline = generate_outline(today_entry, previous_summary)

    print("[pipeline] Generating draft (with web search grounding)...")
    draft_md = generate_draft(today_entry, outline)

    for round_num in range(1, MAX_REVISION_ROUNDS + 1):
        print(f"[pipeline] Critique round {round_num}...")
        review = critique_draft(draft_md)
        if review["passes_quality_bar"]:
            print("[pipeline] Passed quality bar.")
            break
        print(f"[pipeline] Issues found: {review['issues']}")
        draft_md = revise_draft(draft_md, review["revision_instructions"])
    else:
        print("[pipeline] WARNING: did not pass quality bar after max revisions. "
              "Flagging for extra-careful human review.")

    print("[pipeline] Running style pass...")
    final_body = style_pass(draft_md)

    print("[pipeline] Generating recap/takeaways/exercise/preview content...")
    continuity = generate_continuity(final_body, outline["planned_exercise"])

    print("[pipeline] Finding and verifying further-reading resources...")
    resources = generate_resources(today_entry)
    print(f"[pipeline] {len(resources)} verified resource(s) will be included.")

    full_content = assemble_final_post(
        today_entry, next_entry, previous_summary, final_body, continuity, resources
    )

    post = frontmatter.Post(full_content)
    post["day"] = day_number
    post["phase"] = today_entry["phase"]
    post["topic_title"] = today_entry["title"]
    post["title"] = f"Day {day_number}: {today_entry['title']}"
    post["recap_summary"] = continuity["recap_summary"]
    post["generated_at"] = datetime.now(timezone.utc).isoformat()
    post["status"] = "pending_review"

    out_path = draft_path(day_number, today_entry["title"])
    with open(out_path, "w") as f:
        frontmatter.dump(post, f)

    # Advance state only after a successful write.
    state["next_day_index"] += 1
    save_state(state)

    print(f"[pipeline] Draft written to {out_path}")
    return str(out_path)


if __name__ == "__main__":
    try:
        path = run_pipeline()
        if path:
            print(path)
    except Exception as e:
        print(f"[pipeline] FAILED: {e}", file=sys.stderr)
        sys.exit(1)
