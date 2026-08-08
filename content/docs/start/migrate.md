---
title: Import existing weights
seo_title: "Load upstream weights in LibreYOLO"
description: "Point LibreYOLO at a checkpoint from an upstream project. Autoconversion rewraps it at load time, keeping its class count and names."
lead: "LibreYOLO ports its model families from upstream projects, so their released checkpoints are almost loadable already. What they lack is metadata. Autoconversion supplies it at load time."
keywords: [libreyolo convert weights, load upstream checkpoint, libreyolo migration, convert pth to libreyolo, autoconversion]
last_verified: "1.5.0"
meta:
  - label: Entry point
    value: LibreYOLO("path/to/upstream.pth")
    mono: true
  - label: Written beside the source as
    value: "<source>-<Prefix><size>[-task].pt"
    mono: true
  - label: Scripted converters
    value: weights/ in the repository
    mono: true
snippets:
  convert:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Substitute the path to a checkpoint you already have. A recognized
        # upstream layout is converted on the fly, written next to the
        # source, and then loaded.
        model = LibreYOLO("path/to/upstream-checkpoint.pth")

        # Class count and names come from the tensors and the file's own
        # metadata, so a fine-tune keeps its label set instead of COCO's.
        print(model.family, model.size, model.task, model.nb_classes)
        print(model.names)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=path/to/upstream-checkpoint.pth \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Check the result
      language: bash
      code: |
        # The converted file satisfies the same schema as a published one.
        libreyolo metadata path=path/to/upstream-checkpoint-LibreYOLO9t.pt
---

## What happens when you load a foreign file

`LibreYOLO()` loads any weight file through the restricted, weights-only path
first. If the result carries complete LibreYOLO metadata, it is used directly.
If it does not, the file goes to the autoconverter before anything else is
attempted. If the restricted load fails outright, which happens when a
checkpoint has a third-party object pickled into it, the autoconverter is tried
with a loader that neutralizes those objects instead.

Autoconversion does four things. It unwraps the tensor dictionary from whichever
layout the upstream project used. It asks every registered family whether it
recognizes the resulting keys, remapping names where the upstream naming differs
from LibreYOLO's port. It wraps the winner in a checkpoint that satisfies
metadata schema v1.0, reading size, task and class count from the tensors
themselves. Then it writes the result next to the source file and loads that.

<code-tabs name="convert" />

The conversion is not silent. A converted file is logged with the family, the
source name, the output name and the resulting class count, so a run's log
records exactly what was loaded.

## The layouts it unwraps

Upstream checkpoints nest their weights in a handful of conventional places, and
the converter tries them in order until one holds tensors: an EMA block under
`ema.module` or a flat `ema`, an `ema_state_dict` with its `module.` prefix
stripped, then `params_ema`, `params`, `ema_net`, `net`, `model`, `state_dict`,
and finally the object itself. Trying several rather than the first means an
`ema` block holding only counters does not mask the real weights below it.

Wrapper prefixes come off too: `module.` from distributed training,
`_orig_mod.` from a compiled model, and a `model.model.` nesting some
redistributions add.

## What it reads, and from where

Size, task and class count come from the tensors, not from the filename, which
is why a fine-tuned checkpoint converts with its own class count instead of the
architecture's default. Class names are taken from the checkpoint's own metadata
when present, from an `args` or `hyper_parameters` block if the names sit there,
and are trimmed to the detected class count so a fine-tune that kept its base
label set does not carry indices its head no longer has.

Dense tasks are handled explicitly rather than being given fabricated labels. A
depth checkpoint gets one class named `depth`, a restore checkpoint one class
named `image`. A pose checkpoint must yield a keypoint count, either from the
tensors or from the family; if neither produces one, conversion is refused
rather than writing an incomplete file.

RF-DETR gets its own recognizer, because size detection needs the whole
checkpoint and because its head has 91 outputs where LibreYOLO uses the
80-class COCO convention. A checkpoint is normalized to 80 classes when it
carries exactly 80 names, or declares a class count of 80, or names COCO as its
dataset, or carries no class or dataset metadata at all. A genuine 90-class
model, identified by its names, an explicit non-80 count or a non-COCO dataset
hint, is preserved as it is.

## Where the converted file goes

The output is written beside the source, named after it:

```text
<source stem>-<FilenamePrefix><size>[-<task suffix>].pt
```

A tiny YOLOv9 detector saved as `upstream-checkpoint.pth` therefore becomes
`upstream-checkpoint-LibreYOLO9t.pt`. Naming it after the source rather than
after the family means two fine-tunes of the same family and size in one
directory do not overwrite each other, and neither collides with an official
checkpoint. The file is rewritten on every load, so it never goes stale against
its source. If the directory is read-only, the converted file goes to a fresh
private temporary directory instead and the log says where.

From then on it is an ordinary LibreYOLO checkpoint: it loads through the
metadata path, and `libreyolo metadata` reports it as valid.

## Cases that need a hand

Two families sit outside the generic recognizer. The gaze family is excluded
outright: it is inference only and its released weights carry redistribution
restrictions. RF-DETR is excluded because it has the dedicated recognizer
described above, which is what handles it instead.

Raw upstream PIDNet checkpoints are refused, with an error pointing at
`weights/convert_pidnet_weights.py`. That script writes the Cityscapes semantic
metadata the checkpoint needs.

D-FINE and DEIM share the same architecture keys, so tensors alone cannot
separate them. When both claim a file and no sibling family with a
distinguishing marker is in the running, the filename decides: a name in the
shape of `dfine_hgnetv2_n_coco.pth` or `deim_hgnetv2_n_coco.pth` settles it, and
a name that says nothing is refused with that explanation rather than guessed.
Instantiating `LibreDFINE` or `LibreDEIM` directly also resolves it.

When several families legitimately claim one file, a subclass beats the base
class it refines, and registry order decides the rest, since that order encodes
how specific each family's check is. The filename is consulted only for the
D-FINE and DEIM tie, so a file's name can never promote a broad match over a
precise one.

## The scripted converters

The repository carries per-family conversion scripts under `weights/`, plus
shared helpers for the repeated plumbing. They are the route for a file the
runtime path declines, for producing a checkpoint ahead of time rather than at
load time, and for the families whose metadata has to be supplied rather than
inferred from tensors.

Those scripts are part of the repository, not the installed package, so using
one means cloning:

```bash
git clone https://github.com/LibreYOLO/libreyolo.git
cd libreyolo
python weights/convert_pidnet_weights.py --help
```

Every script writes a checkpoint that satisfies schema v1.0, which is the same
bar autoconversion meets and the same bar published weights meet. See
[checkpoints and weights](/docs/weights) for what that schema contains.
