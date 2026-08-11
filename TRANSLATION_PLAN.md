# Live Docs Translation Plan — 7 New Languages

Goal: translate the **live docs** (`content/docs/`, 176 pages, ~169k EN words) into
Spanish, Italian, Russian, Chinese, Portuguese, Arabic and French.

**In scope:** `content/docs/**` only.
**Out of scope:** versioned snapshot pages (`v1.1.0`–`v1.4.0` stay EN+ZH forever),
articles, marketing pages, `/docs/experimental`, `/docs/librevlm`. Untranslated
pages already fall back to English with an English canonical, so partial rollout
is always safe.

## Final language roster (13 locales + English)

Status as of 2026-08-11. "Pages" counts `<slug>.<locale>.md` twins out of 174.
Every locale now has its style guide (`scripts/translation/STYLE-<locale>.md`),
its UI strings (`messages/<locale>.json`) and its routing entry, so nothing but
translation volume is left.

| Locale | hreflang | OG locale | Label | Pages | Notes |
|--------|----------|-----------|-------|-------|-------|
| `es` | `es` | `es_ES` | Español | **174/174** | Complete. Neutral international Spanish, `tú` |
| `zh` | `zh-CN` | `zh_CN` | 中文 | **174/174** | Complete. ~25% of site traffic, the highest-value locale |
| `pt` | `pt-BR` | `pt_BR` | Português | **174/174** | Complete. pt-BR; glossary marks pt-PT splits for a later fork |
| `it` | `it` | `it_IT` | Italiano | **174/174** | Complete. `libreria` is correct here, unlike es |
| `ru` | `ru` | `ru_RU` | Русский | **174/174** | Complete. Still wants ICU plurals in `Models.familyCount` |
| `fr` | `fr` | `fr_FR` | Français | 69/174 | NBSP typography in `fr.json`, do not let a formatter strip it |
| `de` | `de` | `de_DE` | Deutsch | 48/174 | Industrial machine vision. `task` -> "die Aufgabe" except the `task=` concept |
| `id` | `id` | `id_ID` | Bahasa Indonesia | 48/174 | Never calque *instance* as `instansi`, that means a government agency |
| `vi` | `vi` | `vi_VN` | Tiếng Việt | 31/174 | `lớp` collides: bare for *layer*, `lớp đối tượng` for *class* |
| `pl` | `pl` | `pl_PL` | Polski | 28/174 | Impersonal register, since 2nd person would gender the reader |
| `ja` | `ja` | `ja_JP` | 日本語 | 18/174 | No space at the JA/Latin boundary, the opposite of the zh rule |
| `uk` | `uk` | `uk_UA` | Українська | 15/174 | Terminology built independently of `ru`, never forked from it |
| `ko` | `ko` | `ko_KR` | 한국어 | 15/174 | Particle choice after Latin identifiers follows the spoken reading |

**Remaining: 1,120 pages** across the eight incomplete locales.

**Dropped, deliberately:** `ar` (Arabic) and `th` (Thai). Arabic was the only
right-to-left locale, so no RTL support is needed anywhere in the site; Thai was
the only script needing special line-breaking. Both had reached the
style-guide/UI-strings stage — recoverable from git history at commit `8322aeb`
if that decision is ever revisited.

### Sequencing for the remaining work

1. **Finish what is started** — `zh` (68 pages left), then `pt`, `it`, `fr`, `ru`.
   Cheapest possible value: the assets and conventions already exist.
2. **`de`, `ja`, `ko`** — the industrial-CV trio. Highest expected return per
   page of any *new* language for this product.
3. **`pl`, `uk`, `vi`, `id`** — reach plays; each needs assets built first.
   Build `uk` terminology independently, never forked from `ru`.

### Scale consequence (decide before step 2)

13 locales × 174 pages = **2,262 twin files**. Every English docs edit then
implies up to 15 twin updates. The `source_hash` frontmatter check plus the CI
validator (Phase 3) stops being a nice-to-have and becomes the thing that keeps
the corpus from rotting silently. Wire it before adding an eighth language.

## Phase 0 — Platform (one PR, before any content)

1. `src/i18n/routing.js`: extend `locales`, `localeHtmlLang`, `localeLabels`.
   `src/i18n/metadata.js`: extend `OG_LOCALES`.
2. **`messages/<locale>.json` × 6 new** (~285 lines each, translated from `en.json`).
   Mandatory on day one: `src/i18n/request.js` imports the file per locale with no
   fallback — a missing file crashes every page in that locale.
3. Generalize the ~10 hard-coded `locale === 'zh'` conditionals (untranslated-notice
   banners in `articles/[slug]`, `articles/page`, `DocsKit.jsx`, etc.) to
   `locale !== 'en'`, with banner text moved into the message files.
4. Versioned snapshot pages: no change needed — their `if (locale === 'zh')`
   branches fall through to English for every other locale. Verify only.
5. **Build-cost check:** each locale adds ~176 static docs pages plus per-page OG
   images. Measure `next build` time right after this PR (all fallback pages) and
   again after the first full language ships. If Vercel build time balloons,
   mitigate then (e.g. static-param only translated locales per slug).
6. Merge gate: build green; `/es/docs/models/yolov9` serves English with English
   canonical; language switcher shows all 8; sitemap/hreflang emit automatically
   (both derive from `routing.locales` — verified, no edits needed).

## Phase 1 — Translation pipeline (build once, reuse for every campaign)

1. **Style guide + per-language glossary.** Seed the zh glossary from the 10
   existing `.zh.md` articles; draft the other six from a ~100-term list extracted
   from the docs (checkpoint, weights, inference, quantization, fine-tune, export,
   augmentation, …). House rules, all languages:
   - Full twin file `<slug>.<locale>.md`; loader picks it up automatically.
   - Translate: `title` (task/concept pages), `seo_title`, `description`, `lead`,
     FAQ, body prose, comments inside code blocks.
   - **Localize** `keywords` for each market's real search phrasing — never
     literal translation.
   - Keep byte-identical: code, snippet structure and labels, `families`,
     `last_verified`, media paths, internal links. Model names stay in Latin script.
2. **Validation script** (`scripts/validate-translations.mjs`), run in CI on any PR
   touching `content/docs`: per twin — frontmatter keys match EN; heading count and
   levels match; code-block count matches and code bodies identical modulo
   comments; internal link targets match; leftover-English heuristic (Latin-char
   ratio) for non-Latin-script locales (zh, ru, uk, ja, ko).
3. Agent prompt templates: one for translation (glossary + style guide + file),
   one for an independent review pass (fluency + terminology + nothing-invented).

## Phase 2 — Per-language campaigns, strictly one language at a time

**Order:** see the sequencing under the roster above — finish `zh`, `pt`, `it`,
`fr`, `ru` first, then `de`/`ja`/`ko`, then `pl`/`uk`/`vi`/`id`.
Rationale: zh has proven demand (~25% of site traffic); the industrial-CV trio
outperforms raw speaker counts for this product.

Each campaign is identical (174 files — see exclusions below):

| Wave | Files | Content |
|------|-------|---------|
| A — pilot | 10 | `start/install`, `start/quickstart`, `start/faq`, `tasks/object-detection`, `models/yolov9`, `models/rf-detr`, `export/onnx`, `train/datasets`, `predict/results`, `cli/predict`. Review by hand, lock the glossary. |
| B | ~50 | rest of `start` + `tasks` + `predict` + `train` |
| C | ~44 | `export` + `cli` + `reference` |
| D | ~82 | `models` long tail (formulaic; fastest per word) |

Per wave: translate (20–30 files in parallel) → validator → independent review
pass → `next build` → deploy → spot-check 5 rendered pages.

**Excluded from every language:** `start/changelog.md`, `start/versions.md` —
they churn every release; English fallback handles them permanently.

**Checkpoint between campaigns:** after each language ships, watch Search Console
impressions for `/<locale>/docs` for 6–8 weeks. If es/pt show no traction, later
campaigns can be re-ordered or dropped — nothing depends on completing all seven.

## Phase 3 — Maintenance (starts the day zh ships)

- Release checklist addition: `git diff --name-only <prev-release> -- content/docs`
  → queue the changed files' twins for retranslation in every shipped locale.
- Validator runs in CI; a translated twin whose EN source changed gets flagged
  (store the EN source hash in the twin's frontmatter, e.g. `source_hash:`).
- Every new EN docs page ships with twins for all live locales, or consciously
  rides the English fallback until the next batch.

## Scale summary

- New files: 174 × 7 = **1,218** (~1.18M source words to translate).
- Agent runs: ~1,220 translate + ~1,220 review ≈ 2,400, batched per wave.
- Recurring cost: every EN docs edit now implies up to 7 twin updates — this,
  not the initial translation, is the long-term commitment.

## Risks

1. **Build/page explosion** — ~1,200 extra static pages + OG images. Measured at
   Phase 0.5; mitigation decided on data, not guessed.
2. **Quality in languages Xuban can't self-review** (ru, ar, it, pt) — independent
   review pass is mandatory there; pilot waves get extra scrutiny.
3. **Sync drift** — mitigated by `source_hash` + CI validator; accepted as the
   main ongoing cost.
4. OG card text remains English everywhere — accepted for now.
