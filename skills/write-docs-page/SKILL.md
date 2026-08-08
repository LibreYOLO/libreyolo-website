---
name: write-docs-page
description: Write or edit a page in the v2 documentation tree at libreyolo.com/docs (content/docs/**). Use when adding a model, task, export-format, CLI or guide page, or when editing an existing one. Covers the file layout, the frontmatter schema, the generated-data tags, the house visual rules, and the prose rules.
---

# Write a docs page

The v2 docs are markdown files in `content/docs/<section>/<slug>.md`, rendered by
`src/app/[locale]/docs/<section>/[slug]/page.jsx`. Page structure and the full
page list live in `PLAN_docs_revamp.md` in the `libreyolo` repo.

Two rules carry most of the quality:

1. **Prose is authored. Facts are generated.** You write sentences. You never
   type a support matrix, a checkpoint filename, a parameter count or a
   benchmark number into prose; you place a tag and the pipeline fills it from
   `src/data/docs/registry.json`. Anything typed by hand is wrong one release later.
2. **Read the `humanizer` skill in this repo before writing a word of prose,**
   and run its final checklist before you call the page done. Documentation that
   reads as machine-written destroys trust faster than a missing page does.

## File layout

```
content/docs/models/rf-detr.md        English source
content/docs/models/rf-detr.zh.md     optional translation, same frontmatter shape
src/data/docs/registry.json           generated facts (never hand-edit)
src/data/docs/nav.json                sidebar manifest
```

## Frontmatter

```yaml
title: RF-DETR                # H1 and breadcrumb
families: [rfdetr]            # registry keys; lineage pages list several
seo_title: "..."              # <=60 chars, the <title>
description: "..."            # <=155 chars, meta description
lead: "..."                   # 1-2 sentences under the H1. A definition, not a pitch.
keywords: [...]
last_verified: "1.5.0"
hero: {src, poster, caption}  # optional
snippets:                     # runnable code lives HERE, not in the body
  predict: [{label, language, code, expect?}, ...]
  train: [...]
  val: [...]                  # include a tab reproducing our published number
  export: [...]
```

Model pages carry no `faq` and no `related` keys. See the structure section
above for why.

Snippets sit in frontmatter so the same strings can be extracted and executed by
CI. Every snippet must run as pasted, on CPU, from a clean environment.

## Model page structure

A model page is a usage reference for that model, not an encyclopedia entry
about it. The reader arrived to run the thing. Sections, in this order:

| Section | Contents |
|---|---|
| Install | The extra it needs, if any, and the one pip line |
| Predict | `<code-tabs name="predict" />`, plus what the Results carry |
| Variants | `<task-support />` and `<checkpoint-table />`. What exists to load |
| Benchmarks | `<benchmark-table />` and `<va-embed />` when there is data |
| Train | `<code-tabs name="train" />`, plus the arguments that matter for this family |
| Validate | `<code-tabs name="val" />`, including the command that reproduces our published number |
| Export | `<export-matrix />` and `<code-tabs name="export" />` |
| Licensing | `<provenance-box>`, which carries the caveats, provenance rows and interpretation |
| Citation | `<citation-block />`, verbatim from upstream with its source URL |

**Do not add these to a model page:**

- **An architecture or "how it works" section.** We are not the paper. Link the
  paper in the header metadata and spend the page on usage. A sentence about
  architecture is fine where it changes what the reader types (for example, that
  there is no NMS, so `conf` and `max_det` behave differently); a section
  explaining the decoder is not.
- **An FAQ.** If a question is worth answering it belongs in the section it is
  about. Google stopped showing FAQ rich results in May 2026, so there is not
  even an SEO reason left.
- **A related-models section.** The sidebar and the task pages already do
  navigation. Cross-link inline where a comparison actually helps the reader
  choose.

Keep the page answering: how do I install it, run it, train it, validate it,
export it, what variants exist, and what am I allowed to do with it.

## Licensing section

Always open with the two standing caveats before any specific claim. They are
rendered by `<provenance-box>` itself rather than typed per page, so they cannot
go missing, but do not undercut them in your own prose:

1. Check the license on the Hugging Face repository of the specific weights
   being downloaded. It is the authoritative source, every LibreYOLO checkpoint
   carries one, and licenses are not always uniform across a family.
2. What the page states is a description, not legal advice. If it matters
   commercially, the reader reads the licenses and takes their own counsel.

Then the provenance rows. The commercial-use answer is **a row in that same
list**, labelled Interpretation, not a separate heading. The reader comparing
licenses across model pages should find every fact in the same slot every time:

    Original work / Upstream license / Upstream source / LibreYOLO code /
    Weights / Interpretation

Never state a license more permissively than the repository does, and when
weights are non-commercial, say so plainly in the same breath as the code
license so the two are never confused.

## Citations

A citation is an attribution. Getting it wrong sends credit to the wrong people
and sends readers to a paper that does not exist, so this is the one block on
the page with a verification procedure rather than a style rule.

1. **Copy the BibTeX verbatim from the authors' own citation block**, normally a
   `## Citation` section in the upstream README or a `CITATION.cff`. Never
   assemble one from a paper's metadata, and never retype it.
2. **Cross-check it** against the arXiv API
   (`https://export.arxiv.org/api/query?id_list=<id>`) or the publisher: every
   author present and in order, the venue, and the year. Papers get accepted
   after preprinting, so an `@article ... arXiv preprint` entry often should be
   an `@inproceedings` by now.
3. **Store both the BibTeX and its source URL** in the registry
   (`upstream.bibtex`, `upstream.bibtex_source_url`). The page renders the URL
   beneath the entry so a reader can check us.
4. **Record when it was verified and against what**, in
   `upstream.bibtex_verified`.

This is not hypothetical. The first draft of the RF-DETR page carried a
hand-assembled entry that was wrong five ways at once: wrong citation key, two
of five authors missing, wrong entry type, wrong venue, and a title that did not
match the one the authors ask you to cite.

Render it with `<citation-block />`. Do not paste BibTeX into the markdown body.

## Generated-data tags

Place these in the body where the block belongs. They read from the registry
using the page's `families`.

| Tag | Renders |
|---|---|
| `<task-support />` | Tasks, checkpoint counts, training, export counts, since-version |
| `<benchmark-table task="detect" />` | Benchmark rows plus a source line |
| `<va-embed />` | The Vision Analysis chart, when the family has one |
| `<checkpoint-table />` | Published weights, grouped by task |
| `<export-matrix />` | Task x format ticks, plus per-combination notes |
| `<code-tabs name="predict" />` | A snippet group from frontmatter, by key |
| `<provenance-box>...</provenance-box>` | License caveats, provenance rows, interpretation |
| `<citation-block />` | Verified upstream BibTeX plus a link to its source |

**Trap:** self-closing custom tags are expanded to open/close pairs by
`expandSelfClosingTags()` in `src/lib/docs.js` before parsing. HTML5 ignores the
slash on unknown elements, so without that step a `<checkpoint-table />` swallows
the rest of the page. If you add a new tag, keep the hyphen in its name.

## Everything is visible. Nothing is behind an interaction.

No tabs. No accordions. No "show more". No disclosure triangles, hover-only
tooltips carrying information you cannot get otherwise, or anything else that
requires a click to read.

The reason is who reads these pages. A large and growing share of the audience
is an agent working on someone's behalf: it fetches the page or its `.md` twin,
reads it once, and acts. It does not click a tab to discover that a CLI form of
the command exists. Neither does a human scanning for the CLI form, because a
tab strip only advertises the variant currently selected. The same content is
also lost in the markdown twin, in `llms-full.txt`, in print, and in a
screen-reader pass.

So the Python example, the CLI example and the standalone-runtime example all
render, stacked, each labelled, in one pass down the page. It costs vertical
space, which is the cheapest thing a docs page has.

Two consequences worth stating:

- Prefer a server-rendered block to a client component. If a block needs state
  to show its content, the content is hidden from something.
- When you are tempted to collapse a long list to keep the page tidy, cut the
  list instead, or move it to its own page. Tidiness that hides content is not
  tidiness.

The copy button is the one interaction that survives, because it adds a
convenience and hides nothing.

## Visual rules

These are not preferences. They come from how reference sites that people
actually praise (rustdoc, pkg.go.dev, MDN, caniuse, Wikipedia infoboxes, PyPI,
PostgreSQL docs) present dense structured facts, and from the fact that the
first draft of this docs system was rejected for looking machine-generated.

**Never use:**

- Stat tiles. No fact is ever set as a big number in a box. Not one of the
  reference sites surveyed does this for a static property. Counts go inline at
  body size next to their label.
- Status pills or badge rows. A support tier, a license and a version are
  metadata rows, not chips.
- Card wrappers. A table is a table; it does not live inside a bordered rounded
  container. Separation comes from hairline rules and whitespace.
- Accordions, tabs, or any other widget that hides content behind a click. See
  the section above; this is the rule that matters most for agent readers.
- Icons as decoration next to headings.
- Colored chips inside table cells.

**Do use:**

- A key-value metadata list under the H1, with a fixed field order that is the
  same on every page of that type. The label is the quieter element; emphasis
  and links go on the value.
- Bare tables: hairline row rules, a slightly stronger rule under the header,
  no fill, no radius, no hover, no zebra. Wrap in an `overflow-x: auto`
  container so the page body never scrolls sideways.
- `tabular-nums` and right alignment for any column of figures.
- Monospace for every identifier: filenames, flags, arguments, class names.
- For a support matrix: **a tick means supported, an empty cell means not
  supported, and there is no third state.** A grid answers one binary question.
  Any finer grading the library keeps internally is a distinction the reader
  cannot act on inside a cell, so it belongs in the notes underneath, stated in
  words, naming the combination and the measured caveat. Nothing longer than a
  mark goes inside a cell. Color goes on the mark, never on the cell background.
- Units belong in the column heading, never repeated in every cell. Write
  `Params (M)` and put `30.47` in the cell, not `30.47 M`.
- One missing-data convention across every table: an empty cell means the value
  is not recorded. Never a dash, never `n/a`, never mixed.
- Color for exactly two things: links, and state inside a matrix. Not for
  labels, headings, panel backgrounds, or making numbers look important.

**Density target:** at least 200 words of real content in the first 1440x900
screen. Below that the page is decoration-heavy. `scripts/` has no checker for
this yet; eyeball it.

## Prose

**Read the `humanizer` skill in this repo and apply it.** It is the full
treatment of AI writing patterns (promotional language, participle padding,
hedging, rule of three, filler, em dashes) and it is what keeps docs prose from
reading as generated. Run its checklist before calling a page done.

Two places where docs override it:

- **Spelling is US**, not British. The humanizer's checklist item 5 says to
  default to British English; ignore that here. The site is US throughout
  (`color`, `behavior`, `optimize`, `license`).
- **No first person and no personality.** The humanizer's "Personality and soul"
  section is written for articles and blog posts, where opinions and "I keep
  coming back to..." belong. Reference pages are not that. Address the reader as
  *you*, keep yourself out of it, and let the facts carry the page. The rest of
  the humanizer applies unchanged.

### Never call anything experimental

The word does not appear anywhere in the docs: not in prose, not in a legend,
not as a label, not in a table cell, not in a tooltip. Same for its relatives:
*beta*, *preview*, *alpha*, *unstable*, *use at your own risk*.

It fails on two counts. It tells the reader nothing they can act on, because it
does not say what will go wrong or when. And it reads as a liability disclaimer,
which invites the reader to assume the worst about a capability that usually
works fine.

Say what is actually true instead. For export support the vocabulary is the one
in ADR 0011:

- **Validated**, parity checked against PyTorch, and name the test if there is one.
- **Available**, conversion is implemented but numeric runtime parity evidence is
  incomplete or has not been recorded.
- **Not supported**, the exporter refuses the combination before it runs.

Then give the specific reason. `libreyolo/export/support.py` carries a measured
`reason=` string for every non-validated combination, and those sentences are
the most useful content on the page. Pass them through verbatim.

> Before: TensorRT export is experimental for pose.
> After: TensorRT export is available for pose. Matched boxes fall to 0.704 IoU with 41.4 px of coordinate drift, so it is not parity validated.

The same rule covers models and training. Do not label a family experimental.
State which paths are covered and which are not: "Inference is bit-equivalent to
upstream on the same checkpoint. Full-dataset convergence, multi-GPU behavior
and augmentation beyond horizontal flipping have not been validated."

Docs-specific rules on top:

- The lead is a definition, not a pitch. "A detection transformer that predicts
  a fixed set of objects instead of a dense grid", not "a powerful and flexible
  detector that unlocks real-time performance".
- No throat-clearing under a heading. The reader knows what section they are in.
- Every number carries its source in the sentence or the line beneath.
- The same concept keeps the same word, every time.
- Never name a competing library. Upstream model authors and their
  organizations are always credited; that is provenance and it is required.
- Surface the library's own words where it has them. `support.py` records a
  measured reason for every non-validated export cell; those sentences are more
  useful than any summary you would write, so pass them through.

Length band for a model page: 600 to 1200 words of prose, tables and code
excluded.

## Before you commit

1. `curl -s localhost:3000/docs/<path> | grep` for the data you expect. A custom
   tag that silently rendered nothing is the most common failure.
2. Check the page in dark mode and at 390 px wide.
3. Every checkpoint filename you mention exists in the registry, which is gated
   on the Hugging Face org listing. If it is not in the registry, it does not go
   on the page, no matter what the code implies.
4. Run the humanizer checklist.
5. Confirm no `—`, no emoji, no competitor library names, and no use of the word "experimental" (see the rule above).
