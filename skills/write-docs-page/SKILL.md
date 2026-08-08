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
  quickstart: [{label, language, code, expect?}, ...]
  train: [...]
  reproduce: [...]
  export: [...]
faq: [{q, a}, ...]            # renders the FAQ section AND the FAQPage JSON-LD
related: [{href, label, note}, ...]
```

Snippets sit in frontmatter so the same strings can be extracted and executed by
CI. Every snippet must run as pasted, on CPU, from a clean environment.

## Generated-data tags

Place these in the body where the block belongs. They read from the registry
using the page's `families`.

| Tag | Renders |
|---|---|
| `<task-support />` | Tasks, checkpoint counts, training, export counts, since-version |
| `<benchmark-table task="detect" />` | Benchmark rows plus a source line |
| `<va-embed />` | The Vision Analysis chart, when the family has one |
| `<checkpoint-table />` | Published weights, grouped by task |
| `<export-matrix />` | Task x format support, with legend |
| `<code-tabs name="quickstart" />` | A snippet group from frontmatter |
| `<provenance-box>...</provenance-box>` | Upstream and license rows, your prose inside |

**Trap:** self-closing custom tags are expanded to open/close pairs by
`expandSelfClosingTags()` in `src/lib/docs.js` before parsing. HTML5 ignores the
slash on unknown elements, so without that step a `<checkpoint-table />` swallows
the rest of the page. If you add a new tag, keep the hyphen in its name.

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
- Accordions for content the reader came to read. An FAQ is headings and
  paragraphs.
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
- For a support matrix: one short token per cell, three distinguishable icon
  *shapes* so it survives greyscale and colorblindness, color on the mark and
  never on the cell background, and a legend below written as full sentences in
  a definition list. Nothing longer than a token goes inside a cell; caveats go
  under the table.
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
