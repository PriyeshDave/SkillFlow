# Interactive Explainers

Interactive HTML explainers for each published post in the **Zero to Agentic** series (dev.to).

Each file is a self-contained, animated, scroll-triggered explainer with an interactive playground — open directly in a browser, no build step or dependencies required.

## Usage

- **Local preview:** open the `.html` file directly in any browser.
- **Publish:** host on GitHub Pages / Netlify / Vercel / Cloudflare Pages, then link to it from the corresponding dev.to post (dev.to strips raw HTML/JS, so it can't be embedded directly).
- **Naming convention:** `nlp-{topic-slug}-explainer.html`, matching the dev.to post title.

## Notes

- Built with vanilla HTML/CSS/JS — no frameworks, no external dependencies except Google Fonts (IBM Plex Mono + Inter).
- Each includes an interactive "Try It Yourself" playground with simplified in-browser rules (not a real NLP library) — clearly labeled as a demo.
- `prefers-reduced-motion` is respected.
