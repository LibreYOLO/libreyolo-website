# Agent Instructions

- Keep `src/app/sitemap.js` synchronized with the public site. Every canonical, indexable public page must appear in the sitemap, including each localized URL whose main content is translated and independently indexable.
- Do not add redirects, 404 pages, `noindex` pages, duplicate URLs, or locale fallbacks that canonicalize to another URL to the sitemap.
- Whenever a route or article is added, removed, renamed, translated, or changes canonical behavior, update the sitemap in the same change and verify the generated production sitemap.
- `src/app/llms.txt/route.js` serves /llms.txt, an LLM-facing index of the site. Its article list is generated, but its page and docs-version lists are static: update them whenever routes or docs versions change (same trigger as the sitemap).

- Don't use em dashes "—" in this website. Exception: Chinese and Russian, where the dash is grammar rather than style (Chinese 破折号 "——", and the Russian copula in "X — это Y"). Removing it there produces broken sentences.

## Translated docs: NEVER SHIP ONE LANGUAGE ALONE

**If you change an English page under `content/docs/`, you MUST update every
existing `<slug>.<locale>.md` twin in the same change. NEVER UNDER ANY
CIRCUMSTANCE push a change that edits an English docs page and leaves its
translations behind.**

A stale translation is worse than a missing one: a missing twin falls back to
English and is correct, while a stale twin confidently documents a version of
LibreYOLO that no longer exists, in a language most reviewers cannot read.

Two scripts enforce this, and both must pass before a docs PR merges:

- `node scripts/translation/sync-check.mjs` — fails when an English page has
  changed since its twin was translated. After retranslating, re-record the
  hashes with `--stamp`. Stamping without retranslating defeats the check.
- `node scripts/translation/validate.mjs <locale>` — structural parity of a
  twin against its source: frontmatter keys, heading skeleton, code blocks
  (identical except comments) and link targets.

Adding a brand-new English page is fine without twins; it falls back to English
until translated. The rule is about *changing* a page that already has them.
Per-language conventions live in `scripts/translation/STYLE-<locale>.md`, and
`TRANSLATION_PLAN.md` holds the locale roster and rollout state.
