# Zero to Agentic: Your 105-Day AI Engineer Roadmap

A 105-day, weekday-cadence teaching series taking readers from NLP
fundamentals through GenAI to Agentic AI, one topic per day, published to
dev.to. Sibling project to BlogMind, reusing its proven publishing/review
architecture with a different generation strategy: this pipeline **walks a
fixed curriculum sequentially** instead of scoring dynamically-sourced
topics.

```
Roadmap (fixed, 105 entries) → Outline → Draft → Critique → Style Pass
   → Continuity (recap/takeaways/exercise/next-preview)
   → Human Review (GitHub PR) → Publish (dev.to, as part of a native Series)
```

## What's different from BlogMind

| | BlogMind | SkillFlow |
|---|---|---|
| Topic source | Dynamic (arxiv/HN/GitHub, scored) | Fixed roadmap, walked in order |
| Cadence | Every 2 days (self-throttled) | Weekdays only (native cron, no throttle needed) |
| Audience/voice | Senior engineers, no hand-holding | Beginners, teaches from fundamentals |
| Structure | Independent posts | Sequential: Day N recaps Day N-1, previews Day N+1 |
| Publishing | dev.to → Medium → LinkedIn | dev.to (as a native Series) + LinkedIn |

## Repo layout

```
config/
  roadmap.yaml       # the 105-day curriculum -- source of truth, ordered
  settings.yaml
src/
  generation/
    voice.py          # teaching-oriented voice guide (different from BlogMind's)
    outline.py
    draft.py
    critique.py         # pedagogical accuracy bar, not just "sounds confident"
    style_pass.py
    continuity.py         # recap/takeaways/exercise/next-preview as structured JSON
    pipeline.py             # walks roadmap.yaml sequentially, assembles final post
  publishing/
    devto_publish.py         # uses dev.to's native "series" field
    publish_all.py
  review/
    create_pr.py
  utils/
    claude_client.py          # OpenAI Responses API wrapper
    settings.py
    storage.py                  # day-index state + prev-day continuity lookup
content/
  drafts/, published/
.github/workflows/
  generate-draft.yml    # weekdays only, native cron -- no self-throttle hack needed
  publish.yml
```

## Setup

1. **Clone and install**
   ```bash
   pip install -r requirements.txt
   cp .env.example .env   # fill in secrets
   ```

2. **Required secrets** (GitHub Actions repo secrets, and locally in `.env`):
   | Secret | Used for |
   |---|---|
   | `OPENAI_API_KEY` | Generation pipeline + LinkedIn copy generation |
   | `DEVTO_API_KEY` | Publishing to dev.to |
   | `LINKEDIN_ACCESS_TOKEN` | Posting to LinkedIn |
   | `LINKEDIN_PERSON_URN` | Your LinkedIn member URN |
   | `GH_TOKEN` | Opening review PRs (repo + PR scopes) |

   `LINKEDIN_ACCESS_TOKEN` / `LINKEDIN_PERSON_URN` can be reused directly
   from BlogMind's setup if you already have them -- same LinkedIn
   Developer App works fine across both projects. Otherwise run
   `python -m src.publishing.linkedin_oauth_helper` once.

3. **Create the `lesson-draft` label** (the publish workflow checks for it):
   ```bash
   gh label create lesson-draft --description "Auto-generated lesson pending review" --color 0E8A16
   ```

## Running locally

```bash
# Generate the next day's lesson
python -m src.generation.pipeline

# Review the draft in content/drafts/, edit if needed

# Publish an approved lesson
python -m src.publishing.publish_all content/drafts/day-001-example.md
```

## How continuity works

Each lesson's frontmatter stores a `recap_summary` field. The next day's
generation reads that back (checking `content/published/` first, falling
back to `content/drafts/` if the previous PR hasn't been merged yet) to
build the "Previously, on Day N-1..." opening. The "Coming up on Day N+1"
teaser is built directly from `roadmap.yaml`, so it's always accurate to
what's actually next, regardless of what the LLM stages produced.

Day numbering, the recap block, the Key Takeaways section, and the next-day
teaser are **all assembled deterministically in `pipeline.py`**, not left
to LLM prompt compliance -- this is a lesson carried over from BlogMind,
where trusting the model to reliably include required sections in free text
(hashtags, formatting) proved unreliable.

## Known tradeoff: review lag and continuity

Day-index advances at generation time, not at merge time -- otherwise a
slow PR review would stall the entire remaining schedule. This means
`content/drafts/` can accumulate more than one pending lesson if review
falls behind. Continuity still works in this case (it reads from drafts as
a fallback), but it's worth reviewing PRs at a steady pace to avoid a large
backlog.
