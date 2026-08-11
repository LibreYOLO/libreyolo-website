# Articles Translation Plan

Follow-on to `TRANSLATION_PLAN.md`, which covers `content/docs/`. This one covers
`content/articles/`. Start it only once the docs roster is finished, so the two
efforts never compete for the same quota.

## What the corpus actually is

10 English articles, **9,768 words total** — about **6% of the docs corpus**
(169k words). The whole articles set is smaller than a single docs section.

| Article | Words | zh twin |
|---|---|---|
| best-ultralytics-alternatives | 2,860 | yes |
| yolo-commercial-license | 1,439 | yes |
| rf100vl-benchmark | 906 | **no** |
| litert-vs-tensorflow-lite | 740 | yes |
| rtmdet-without-mmdetection | 760 | yes |
| simplest-way-to-run-depth-anything-v2 | 730 | yes |
| yolo-nas-with-libreyolo | 629 | yes |
| libreyolo-mentions | 626 | yes |
| yolox-with-libreyolo | 576 | yes |
| libreyolo-at-cvpr-2026 | 502 | yes |

Coverage today: **zh 9/10**, every other locale **0/10**.

## Why this is not just "docs, but smaller"

The mechanics are identical — `src/lib/articles.js` resolves
`<slug>.<locale>.md` exactly like the docs loader, and untranslated articles
fall back to English. But the *content* differs in ways that change the work:

1. **These are the traffic pages.** `yolo-commercial-license` and
   `best-ultralytics-alternatives` are the top two article entry points in
   search. A weak translation here costs more than a weak reference page.
2. **They are persuasive, not descriptive.** Docs pages describe an API; these
   argue a position about licensing and alternatives. Literal translation reads
   badly. Titles and headings often need rewriting rather than translating.
3. **They make legal claims.** The licensing articles state what AGPL-3.0
   requires and what MIT permits. A translator must not soften, strengthen or
   "clarify" any of it. This is the single biggest risk in the corpus.
4. **They carry dated, factual claims** — CVPR 2026 talks, benchmark numbers,
   quoted community posts. `libreyolo-mentions` quotes real people.
5. **One is special:** `rf100vl-benchmark` uses `layout: paper` and embeds
   custom components, and is the only article with no zh twin.

## Rules the docs guides do not cover

Add an `## Articles` section to each `scripts/translation/STYLE-<locale>.md`:

- **Never translate a quotation.** `libreyolo-mentions` and the home page
  testimonials quote real named people in English. Keep the quote verbatim and
  translate only the surrounding framing. Attribution handles stay as written.
- **Do not restate a licence.** Terms like AGPL-3.0, MIT, Apache-2.0,
  CC-BY-NC-4.0 keep their exact identifiers. Translate the explanation, never
  the obligation: if the English says a licence requires releasing source, the
  translation says exactly that, no stronger and no softer.
- **Titles are marketing copy, not labels.** Rewrite for the target market's
  search behaviour rather than translating word for word, the same latitude
  `keywords` already gets in the docs guides.
- **Keep `date`, `author`, `tags` byte-identical.** `tags` are routing keys, not
  prose. `date` is the publication date of the English original.
- **Numbers, model names and benchmark figures are facts.** Never localize a
  metric, never convert a unit, never round.

## Sequencing

1. **`rf100vl-benchmark` into Chinese first.** It is the one gap in the only
   locale with existing article translations, and its `layout: paper` plus
   embedded components make it the riskiest single file. Do it alone, by hand,
   and confirm the rendered page before scaling anything.
2. **Spanish, then the rest.** Spanish is reviewable in-house, so it doubles as
   the quality gate for the article-specific rules above before they reach
   locales nobody here reads.
3. **By value, not evenly.** Translate the two licensing articles into all
   locales before translating the long tail into any. They earn roughly ten
   times the traffic of `libreyolo-at-cvpr-2026`.
4. **Consider skipping two entirely.** `libreyolo-at-cvpr-2026` and
   `libreyolo-mentions` are time-stamped news, not evergreen reference. They
   age out, and a stale translation of a news post is worse than an English
   fallback. Recommend English-only, permanently.

If items 3 and 4 are accepted, the real corpus is **8 articles x 12 locales =
96 files, ~7,600 words each pass** — roughly one afternoon, not a campaign.

## Verification

`scripts/translation/validate.mjs` works unchanged: articles live under
`content/articles/`, so point it there or add the directory to its walk. Two
additions are worth making first:

- Extend the walk to `content/articles/` so twins there are checked at all.
- Add an assertion that `date`, `author` and `tags` are byte-identical to the
  English source, mirroring the existing `families` / `last_verified` check.

`sync-check.mjs` already generalizes: stamping article twins with `source_hash`
gives the same staleness protection, which matters more here because articles
get edited for freshness far more often than reference docs do.

## One-time chore

`src/app/articles/[file]/route.js` hard-codes `.zh.md` when generating the raw
markdown routes, so `/articles/<slug>.es.md` would 404 even once the file
exists. Derive that list from `routing.locales`, the same fix already applied to
`llms.txt`.
