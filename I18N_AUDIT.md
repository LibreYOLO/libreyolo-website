# Site-wide i18n audit

Measured 2026-08-11 against a running dev server, using Ukrainian as the probe
locale (all 174 docs twins exist, so anything still English is a code problem,
not a missing translation).

`scripts/translation/i18n-audit.mjs` produces the static half of this report;
the coverage percentages come from fetching each page and counting Cyrillic
versus Latin characters in the rendered text.

## Measured coverage per page

"ukr%" is the share of rendered letters that are Cyrillic. Code samples, model
names and CLI flags are legitimately Latin, so even a perfect page never hits
100%. Roughly 60-80% is what a fully translated page looks like here.

| Page | ukr% | Verdict |
|---|---|---|
| `/uk/commercial` | 82% | good |
| `/uk/datasets` | 81% | good |
| `/uk` (home) | 66% | good |
| `/uk/science` | 64% | good |
| `/uk/models` | 45% | model names and task keys are Latin by design |
| `/uk/docs/models/yolov9` | 45% | good, the rest is code |
| `/uk/articles` | 24% | articles not translated yet, expected |
| `/uk/benchmarks` | 18% | **mostly hardcoded** |
| `/uk/docs/models` | 13% | **section index is hardcoded** |
| `/uk/docs` | 6% | **docs landing page is hardcoded** |
| `/uk/articles/<slug>` | 6% | articles not translated yet, expected |
| `/uk/docs/librevlm` | 4% | **hardcoded** |
| `/uk/docs/experimental` | 3% | **hardcoded** |
| `/uk/docs/v1.4.0` | 0% | versioned snapshot, EN+ZH only by design |
| `/uk/<missing>` (404) | fine | uses `t('NotFound')`, renders Ukrainian |

The pattern: **markdown-driven pages are translated, hand-written JSX pages are
not.** The docs *content* is complete, but the docs *chrome* around it is not.

## Static audit

`node scripts/translation/i18n-audit.mjs`

**548 hardcoded user-visible strings across 49 files.**

Worst offenders that a reader actually lands on:

| File | Strings | Note |
|---|---|---|
| `docs/page.jsx` | 30 | the `/docs` landing page |
| `docs/[section]/page.jsx` | 16 | `/docs/models`, `/docs/tasks`, every section index |
| `docs/librevlm/page.jsx` | 20 | |
| `docs/experimental/page.jsx` | 19 | |
| `benchmarks/page.jsx` | 4 | plus its layout |
| `components/docs/DocsShell.jsx` | 3 | sidebar chrome on every docs page |
| `components/docs/DocsSearch.jsx` | 4 | search placeholder and empty states |
| `components/docs/ModelBlocks.jsx` | 4 | rendered on all 82 model pages |

Deliberately out of scope:

- `docs/v1.*.0/page.jsx` (197 + 71 + 40 + 20 strings) are frozen version
  snapshots that stay EN+ZH, per `TRANSLATION_PLAN.md`.
- `components/articles/rf100vl/datasets.js` (24) is a dataset-name table;
  those are proper nouns from Roboflow Universe and should not be translated.
- `opengraph-image.jsx` files (1-2 each) render social cards. Worth doing
  eventually, but nobody reads them on the site.

## What to fix, in order

1. **`DocsShell` + `DocsSearch` + `ModelBlocks`.** Shared chrome, so fixing
   three files improves all 174 docs pages in 13 languages at once. Highest
   leverage in the codebase.
2. **`docs/page.jsx` and `docs/[section]/page.jsx`.** The landing page and the
   six section indexes are where a reader enters the docs, and they are
   currently the *least* translated pages in the whole site.
3. **`benchmarks`, `librevlm`, `experimental`.** Standalone pages, each self
   contained.
4. **Articles.** Content, not code: see `ARTICLES_TRANSLATION_PLAN.md`.

## How to fix it

Every one of these needs the same treatment the marketing pages already got:
move the prose into `messages/<locale>.json` under a namespace and read it with
`useTranslations`. The infrastructure is already there and proven; these files
simply never adopted it.

Two things make this cheaper than it looks:

- The English strings are already written. Extracting them into `en.json` is
  mechanical, and the other twelve locales can then be produced by translating
  one JSON file per language rather than editing JSX.
- `messages/en.json` is the single source of truth for key structure, and the
  existing check (`node -e` comparing key sets) already catches a locale that
  drifts out of sync.

Re-run `node scripts/translation/i18n-audit.mjs` after each step; the total
string count is the progress metric.
