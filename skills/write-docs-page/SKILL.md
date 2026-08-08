---
name: write-docs-page
description: Write or edit a page in the v2 documentation tree at libreyolo.com/docs (content/docs/**). Use when adding or changing a model, task, export-format, CLI or guide page. Covers file layout, frontmatter, the generated-data blocks, verification procedure, and the house rules for prose and presentation.
---

# Write a docs page

Pages are markdown in `content/docs/<section>/<slug>.md`, rendered by
`src/app/[locale]/docs/<section>/[slug]/page.jsx`. Facts come from
`src/data/docs/registry.json`. The full page plan is `PLAN_docs_revamp.md` in
the `libreyolo` repo.

Only the model page type is specified below. **Before writing the first page of
a new type, add its spec to this file.** A page type invented on the fly will
not match the others, and inconsistency across pages costs more than any single
page gains.

## Three rules, in priority order

**1. Facts are generated. Prose is authored.** Never type a support matrix, a
checkpoint filename, a parameter count, a benchmark number or a BibTeX entry
into markdown. Place the block, and the registry fills it. Anything typed by
hand is wrong one release later, and hand-transcription has already been the
direct cause of two shipped defects.

**2. Every claim is traceable, and stopping short is the hard part.** See
Verification below. This is the rule that gets broken.

**3. Nothing is behind an interaction.** No tabs, accordions, show-more, or
disclosure widgets. A large share of readers are agents: they fetch the page or
its `.md` twin, read once, and act. They do not click. Neither does a person
scanning, and a tab strip only advertises the variant currently selected.
Hidden content is also absent from the markdown twin, `llms-full.txt`, print,
and screen readers. Stack it and label it. Vertical space is cheap. The copy
button is the one allowed interaction, because it hides nothing.

## Verification

The failure mode is not invented numbers. It is **stepping one pace past what
you know and asserting the next thing as if it had been checked**. This is a
characteristic AI writing pattern and the most damaging one here, because the
unverified clause sits beside a verified one and borrows its credibility.

> Before: The sizes differ mainly by input resolution, so choosing one is a latency decision more than a memory one.
> After: The sizes carry similar parameter counts and differ mainly in input resolution.

The first clause was measured. The second sounded reasonable, was never checked,
and is probably wrong, since activation memory scales with input area.

Rules that follow:

- State the measurement, not the conclusion you drew from it.
- **General machine-learning knowledge is not evidence about this library.**
  Before describing what a model can do, open the family's `model.py` and read
  `UNSUPPORTED_TRAIN_PARAMS` and its capability flags. Writing that a model can
  train from scratch, when the family silently ignores `pretrained`, costs a
  reader an afternoon.
- **Check whether the library already solves the problem** before documenting a
  manual workaround. Exported artifacts load through `LibreYOLO()` by file
  suffix; a page that reaches for raw `onnxruntime` first makes the product look
  weaker than it is.
- Check argument defaults against the source before giving guidance about them.
- Never publish a latency figure without hardware, runtime, precision and batch
  size. A bare millisecond is not a fact; link the Vision Analysis embed.
- "Mainly", "mostly" and "more than" mark unverified inference. Find the number
  or cut the clause.

## Prose

Read the `humanizer` skill in this repo and run its checklist. Two overrides:
spelling is **US**, not British; and reference pages carry **no first person and
no personality**, so ignore its voice section.

On top of it:

- The lead is a definition, not a pitch.
- No throat-clearing under a heading. The reader knows what section they opened.
- One term per concept, every time. Synonyms imply new concepts.
- **Never the word "experimental"**, nor beta, preview, alpha or unstable. It
  says nothing actionable and reads as a disclaimer. Say what is true: validated
  and parity checked, available with the specific caveat, or not supported.
- Never name a competing library. Upstream authors and their organizations are
  always credited; that is provenance and it is required.

## Model page

Sections in this order. Install and run come before anything descriptive;
reference material sits low.

| Section | Contents |
|---|---|
| Install | The extra it needs, and the one pip line |
| Predict | `<code-tabs name="predict" />` and what the Results carry |
| Variants | `<benchmark-table />`, `<va-embed />`. How a reader picks a size |
| Train | `<code-tabs name="train" />` and the arguments that matter here |
| Validate | `<code-tabs name="val" />`, including the command reproducing our number |
| Export | `<export-matrix />`, `<code-tabs name="export" />` |
| Checkpoints | `<checkpoint-table />` |
| Licensing | `<provenance-box>` |
| Citation | `<citation-block />` |

Omit a section that has no content; never reorder.

Do not add: an architecture section (we are not the paper; one sentence is fine
where it changes what the reader types), an FAQ, or a related-models list.

## Blocks

Place the tag; the registry supplies the data.

| Tag | Renders |
|---|---|
| `<benchmark-table task="detect" />` | Benchmark rows and their source line |
| `<va-embed />` | The Vision Analysis chart |
| `<checkpoint-table />` | Published weights, grouped by task |
| `<export-matrix />` | Task by format |
| `<code-tabs name="predict" />` | A snippet group from frontmatter, by key |
| `<provenance-box></provenance-box>` | License caveats, provenance rows, interpretation |
| `<citation-block />` | Verified BibTeX plus its source link |

**Trap:** HTML5 ignores the closing slash on unknown elements, so
`<checkpoint-table />` would swallow the rest of the page.
`expandSelfClosingTags()` in `src/lib/docs.js` rewrites them first. Keep a
hyphen in any new tag name.

## Frontmatter

```yaml
title: RF-DETR
families: [rfdetr]            # registry keys; lineage pages list several
seo_title: "..."              # <=60 chars
description: "..."            # <=155 chars
lead: "..."                   # a definition, not a pitch
keywords: [...]
last_verified: "1.5.0"
hero: {src, poster, caption}  # optional
snippets:
  predict: [{label, language, code, expect?}, ...]
  train: [...]
  val: [...]
  export: [...]
```

Snippets live here, not in the body, so CI can extract and run them. Each must
run as pasted, on CPU, from a clean environment.

## Presentation

Conventions follow reference sites that dense-data readers rate highly
(rustdoc, pkg.go.dev, MDN, caniuse, Wikipedia infoboxes, PostgreSQL docs).

Never: stat tiles, status pills, badge rows, card wrappers, icons beside
headings, colored table cells, or a column that is mostly empty (nineteen rows
with four parameter counts reads as broken, not precise).

Tables are bare: hairline row rules, a stronger rule under the header, no fill,
no radius, no hover, no zebra, inside an `overflow-x: auto` container. Units go
in the column head (`Params (M)`, cell `30.47`). Figures are right-aligned and
`tabular-nums`. An empty cell is the single missing-data convention; never a
dash, never `n/a`.

A support matrix is binary: a tick means supported, empty means not, there is no
third state. Finer grading the library keeps internally is not something a
reader can act on inside a cell.

Color is for links and matrix state. Not labels, headings, backgrounds, or
making numbers look important.

Metadata is a key-value list in a fixed order, the same order on every page of
that type. The label is the quieter element; emphasis and links go on the value.

## Citations

A citation is an attribution. Getting it wrong credits the wrong people. This is
the one block with a procedure rather than a style rule.

1. Copy the BibTeX **verbatim** from the authors' own citation block, normally a
   `## Citation` section in the upstream README or a `CITATION.cff`. Never
   assemble one from paper metadata, never retype it.
2. Cross-check against `https://export.arxiv.org/api/query?id_list=<id>` or the
   publisher: every author, in order, plus venue and year. Preprints get
   accepted, so an `@article` arXiv entry is often an `@inproceedings` by now.
3. Store `upstream.bibtex`, `upstream.bibtex_source_url` and
   `upstream.bibtex_verified` in the registry. The page renders the source link.

The first RF-DETR draft carried a hand-assembled entry wrong five ways at once:
wrong key, two of five authors missing, wrong entry type, wrong venue, wrong
title.

## Licensing

`<provenance-box>` renders two caveats itself so they cannot go missing: check
the license on the Hugging Face repository of the specific weights, which is
authoritative and not uniform across a family; and this is a description, not
legal advice. Do not undercut them in prose.

Rows, in fixed order: Original work, Upstream license, Upstream source,
LibreYOLO code, Weights, Interpretation. Commercial use is the Interpretation
row, not a separate heading. Never state a license more permissively than the
repository does, and when weights are non-commercial say so in the same breath
as the code license.

## Before you commit

1. `curl -s localhost:3000/docs/<path>` and grep for content you expect. A tag
   that silently rendered nothing is the most common failure.
2. Dark mode, and 390 px wide.
3. Every checkpoint name appears in the registry, which is gated on the Hugging
   Face org listing. Not in the registry means not on the page, whatever the
   code implies.
4. Run the humanizer checklist.
5. Grep the diff for em dashes, emoji, competitor library names, and the word
   "experimental".
