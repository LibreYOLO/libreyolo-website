---
name: humanizer
description: Strip AI writing patterns out of prose so it reads like a person wrote it. Use when writing or reviewing any user-facing text in this repo (docs pages, articles, model cards, release notes, UI copy), and whenever text feels padded, promotional, or generically fluent.
---

# Humanizer

AI-written prose has a recognizable smell: promotional adjectives, throat-clearing
before every point, participle clauses that add no information, hedging, and a
tidy uplifting sentence at the end of every section. Readers notice, and on
technical documentation it actively costs trust, because padding is what people
write when they do not know the answer.

Apply this while drafting, not only as a cleanup pass. The goal is prose that a
competent engineer would have written in a hurry and not bothered to decorate.

## The two tests

**Read it aloud.** Anything you would not say to a colleague at a desk comes out.

**Delete the sentence.** If the paragraph still says the same thing, the sentence
was decoration. This kills most of what follows.

## Cut these

### Promotional language

Words to watch: powerful, seamless, robust, comprehensive, rich, flexible,
cutting-edge, state-of-the-art, blazing fast, effortless, elegant, unlock,
leverage, empower, simply, just, easily.

> Before: LibreYOLO's powerful export system seamlessly supports a comprehensive range of deployment targets.
> After: LibreYOLO exports to twelve formats. Support varies per family; see the matrix.

A capability is described by what it does and where it stops, never by how good
it is. If a claim cannot be checked, it does not belong in documentation.

### Throat-clearing and signposting

Sentences that announce what the next sentences will do, restate the heading, or
explain that a topic is important.

> Before: In this section, we will explore how training works. Training is a crucial part of any computer vision workflow. Let's dive in.
> After: (delete all of it, start with the first real instruction)

Under a heading called Training, the reader already knows what the section is
about. Start with content.

### Participle clauses that add nothing

Words to watch: highlighting, underscoring, showcasing, emphasizing, ensuring,
reflecting, enabling, allowing for, making it easy to, providing.

> Before: The decoder emits a fixed number of queries, enabling real-time inference and ensuring stable output in crowded scenes.
> After: The decoder emits a fixed number of queries. Because there is no NMS step, overlapping objects are not merged.

### Vague attribution

Words to watch: studies show, it is widely known, experts recommend, generally
considered, industry standard, best practice.

Every number and every claim of fact carries a source: a benchmark run, a paper,
a file in the repo. If there is no source, cut the claim rather than soften it.

> Before: RF-DETR is widely regarded as one of the strongest real-time detectors.
> After: RF-DETR reaches 58.6 mAP at 704 px on COCO val2017 (RTX 5070 Ti, TensorRT fp16).

### Hedging

Words to watch: might, could, may, generally, typically, often, in some cases,
it depends, relatively, fairly, quite.

One hedge in a sentence is sometimes honest. Two is evasion. If behavior is
conditional, state the condition instead of hedging around it.

> Before: This may generally work better in some cases, depending on your setup.
> After: This is faster when batch size is above 8. Below that, the kernel launch overhead dominates.

### Uplifting closers

The generic sentence that ends a section on an encouraging note.

> Before: With these tools at your disposal, you are well equipped to build powerful vision applications.
> After: (delete)

A section ends when the information ends.

### Rule of three

Forcing ideas into groups of three to sound complete. If there are two real
items, write two. If there are five, write five.

> Before: The API is fast, flexible, and easy to use.
> After: The API takes the same arguments for every model family.

### Synonym cycling

The same thing must keep the same name. Prose variety is a virtue in essays and
a defect in reference material, where a new word implies a new concept.

> Before: The model produces detections. These predictions are then filtered, and the resulting outputs are drawn.
> After: The model returns boxes. Boxes below `conf` are dropped, and the rest are drawn.

Pick the registry's term (`checkpoint`, `family`, `task`, `export format`) and
never swap in a synonym.

### Copula avoidance

Words to watch: serves as, stands as, represents, boasts, features, offers,
functions as.

> Before: `Results` serves as the container that offers access to model outputs.
> After: `Results` holds the model's outputs.

### Negative parallelism and tailing fragments

> Before: It is not just a detector, it is a complete pipeline. One command, no config files.
> After: The CLI runs training, validation and export. It reads settings from flags, so there is no config file to write.

### False ranges

"From X to Y" where X and Y are not ends of a scale.

> Before: LibreYOLO covers everything from detection to depth estimation to OCR.
> After: LibreYOLO covers seventeen tasks, including detection, depth estimation and OCR.

### Subjectless fragments

> Before: No configuration needed. Results saved automatically.
> After: You do not need a configuration file. LibreYOLO writes results to `runs/`.

## Formatting habits to avoid

- **Em and en dashes.** House rule, and they are an AI tell. Use commas, colons,
  parentheses or a full stop. Number ranges take a plain hyphen.
- **Bold scattered through paragraphs** for emphasis. Bold marks a defined term
  on first use, nothing else.
- **Bullet lists whose items begin with a bolded phrase and a colon.** If each
  item is a definition, use a definition list or a table. If each item is a
  sentence, write a paragraph.
- **Emoji** anywhere in documentation prose.
- **Title Case Headings.** Use sentence case.
- **Curly quotes** in code or identifiers.
- **A bulleted list where a table belongs.** Anything with two or more attributes
  per item is a table.

## Technical documentation specifics

- Address the reader as *you*. Describe the software as the actor: "LibreYOLO
  writes", not "it will be written".
- Instructions are imperative: "Set `imgsz` to 640", not "you may want to
  consider setting".
- Prefer present tense. Not "this will download the weights", just "this
  downloads the weights".
- Lead with the fact, then the qualification. "Export to ONNX is validated for
  all four tasks. TensorRT is validated for detection only."
- One idea per sentence. Long sentences in reference material are where
  ambiguity hides.
- Never explain why the documentation is structured as it is. The reader does
  not care.
- Never write a sentence whose content is "this is easy". Difficulty is the
  reader's to judge, and saying it is easy reads as mockery when it is not.

## Before you call a page done

1. Read the first sentence of every section. Does any of them just restate the
   heading? Delete those.
2. Search for the promotional and participle words listed above.
3. Check every number has a source next to it.
4. Check every term for the same thing is actually the same word.
5. Delete the last sentence of each section and see whether anything is lost.
6. Search the diff for `—`, `–`, and emoji.

## Source

Distilled and rewritten for this repo from
[blader/humanizer](https://github.com/blader/humanizer) (MIT, Siqi Chen), which
in turn builds on Wikipedia's "Signs of AI writing". The rule names and the
words-to-watch vocabulary come from that work; the explanations and all examples
here were rewritten for technical documentation. See `SOURCE.md` for the licence.
