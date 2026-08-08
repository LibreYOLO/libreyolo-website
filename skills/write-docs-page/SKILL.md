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
- **A filename suffix names the task, never the dataset.** `-seg` means masks,
  `-obb` means oriented boxes. Neither says what the weights were trained on.
  Four pages said the oriented checkpoints were trained on DOTA, the benchmark
  everyone associates with the task; their model cards say in those words that
  they were not, and name a six-class vehicle set. Read the card, or write
  nothing about the training data.
- **The only images that ship are `parkour.jpg` and `guggenheim-bilbao.jpg`**,
  in `libreyolo/assets/`. In Python use `SAMPLE_IMAGE`, which resolves to the
  first. In shell use its raw URL on the `release` branch. A plausible filename
  from another library's tutorials, `bus.jpg` being the one that got through,
  fails on line one of a reader's first session. Any path in a snippet is a
  claim that the file exists.

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

## Task page

`content/docs/tasks/<task>.md`, rendered by `docs/tasks/[slug]/page.jsx`. The
hub node: it answers "how do I do X in Python" and links out to every model
that serves the task.

| Section | Contents |
|---|---|
| Definition | What the task is and what it returns. No pitch, no history |
| Models | The families that serve it, as prose plus links to their pages |
| Predict | `<code-tabs name="predict" />` with the recommended default |
| Dataset format | The layout this task's loaders expect, with a folder tree |
| Train | `<code-tabs name="train" />` |
| Validate | `<code-tabs name="val" />` and what each metric key means |
| Export | `<code-tabs name="export" />`, linked to the format pages |

Omit a section with no content; never reorder. A task with no trainable
family says so in one sentence rather than dropping the Train section
silently. The metric paragraphs name the literal `metrics/` keys the
validator returns, because that is what the reader prints.

**The generated blocks do not work here.** `benchmark-table`,
`checkpoint-table`, `export-matrix`, `va-embed`, `provenance-box` and
`citation-block` all read the page's registry family, and a task page has
none. Only `code-tabs` renders. So the Models section is a linked list in
prose, and per-model numbers stay on the model pages where the blocks work.
Link, do not transcribe: a hand-typed accuracy column here is exactly the
drift rule 1 exists to prevent.

## Workflow page

`content/docs/train/<page>.md` and `content/docs/predict/<page>.md`. One page
answers one question, end to end and runnable.

No fixed section list, because the questions differ. The rules that hold:
the first paragraph is the answer, not a preamble; every page carries the
runnable snippet for its own subject; and **no page explains a neighbouring
concern**, it links it. The augmentation page does not explain the optimizer.
Where behavior differs per family, say which families, or place a generated
block; never write "some models".

## Export format page

`content/docs/export/<format>.md`.

| Section | Contents |
|---|---|
| Install | The extra, and the one pip line |
| Export | `<code-tabs name="export" />` and the arguments this format adds |
| Run the artifact | Loading it back through `LibreYOLO()`, then the bare-runtime path |
| Constraints | Fixed shapes, precision limits, unsupported ops, per-task gaps |

Same block limitation as task pages: only `code-tabs` renders, because there
is no family. Point at `/docs/reference/export-matrix` for coverage rather
than typing a family list that goes stale in one release.

"Run the artifact" leads with `LibreYOLO("model.onnx")`, because the library
routes on file suffix and a page that reaches for the raw runtime first makes
the product look weaker than it is. The bare-runtime version comes second, for
readers who genuinely have no LibreYOLO installed, and it states that
preprocessing and postprocessing become theirs.

## CLI page

`content/docs/cli/<command>.md`. Small by design: it exists so that
"libreyolo <command> flags" is answered in one screen.

| Section | Contents |
|---|---|
| Synopsis | One fenced `bash` block with the general form |
| Arguments | A table: argument, default, meaning. Defaults read from the source |
| Examples | Three, ordered simple to advanced, each runnable |
| Notes | Exit behavior, `--json`, and a link to the workflow page |

Every default in the argument table is read from the CLI definition, not
recalled. A CLI argument that the family's `train()` ignores is worth a line,
because the CLI accepts it silently.

## Reference page

`content/docs/reference/<page>.md`. Curated, never an autodumped API listing.
Document the deliberate public surface: what a reader is meant to call, with
the real signature and a short example. Schema pages mirror the library's own
`docs/*_schema.md` at the release they were checked against.

## Standalone page

`content/docs/start/<slug>.md`, served prefix-free at `/docs/<slug>`:
install, quickstart, concepts, weights, migrate, faq, licensing, citation,
versions. These URLs are permanent, so the slug is chosen once and never
changed.

Same prose rules as everything else. `faq` is the one page allowed a
question-heading structure, because that is its subject.

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
