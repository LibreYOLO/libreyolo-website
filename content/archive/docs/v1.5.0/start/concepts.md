---
title: Core concepts
seo_title: "LibreYOLO core concepts"
description: "How tasks, model families, sizes and checkpoint filenames fit together in LibreYOLO, and what each support tier promises."
lead: "Four ideas describe every model in LibreYOLO: the task it performs, the family it belongs to, the size within that family, and the support tier the family sits in. The checkpoint filename encodes the first three."
keywords: [libreyolo concepts, libreyolo tasks, libreyolo model families, libreyolo checkpoint naming, libreyolo support tiers]
last_verified: "1.5.0"
meta:
  - label: Filename schema
    value: "Libre<FAMILY><size>[-<task>].pt"
    mono: true
  - label: Canonical tasks
    value: 17
  - label: Support tiers
    value: Flagship, Core, Supported, Inference only, Museum, Sibling tier
snippets:
  inspect:
    - label: List families
      language: bash
      code: |
        # Tasks, sizes and input resolutions for every registered family.
        libreyolo models
    - label: One model
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        print(model.family, model.size, model.task)
        print(model.input_size)
        print(model.nb_classes, model.names[0])
    - label: Pick a task
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Aliases normalize at the API boundary: "keypoints" resolves to
        # "pose", "det" to "detect", "semantic-segmentation" to "semantic".
        model = LibreYOLO("LibreYOLO9t.pt", task="det")
        print(model.task)
---

## Tasks

A task is what a model returns. LibreYOLO has seventeen canonical task names,
and each one names the field on the `Results` object that carries its output.

| Task | Returns |
|---|---|
| `detect` | Axis-aligned boxes with a class and a confidence |
| `segment` | Per-instance masks, one mask per detected object |
| `semantic` | One class label per pixel, with no instance separation |
| `panoptic` | One non-overlapping label per pixel, merging countable things with amorphous stuff |
| `pose` | Per-instance keypoints, rows aligned with the boxes |
| `classify` | A probability over a label set for the whole image |
| `obb` | Oriented boxes, with a rotation angle |
| `point` | One image coordinate per detection, rather than a box |
| `depth` | A dense relative inverse-depth map |
| `normal` | A dense unit-vector surface-normal field |
| `edge` | A dense edge-probability map |
| `restore` | A restored RGB image, for deblurring, denoising or super-resolution |
| `matte` | A soft foreground map from 0 to 1, for background removal |
| `ocr` | Text quads with transcripts, in reading order |
| `embed` | An L2-normalized vector whose dot product measures agreement |
| `gaze` | A gaze direction per detected face |
| `mesh` | A posed 3D body per detected person |

Those are the names that appear in checkpoint metadata and in filenames.
Familiar aliases are accepted wherever a task is passed and normalized before
anything else happens: `detection` and `det` become `detect`, `keypoints`
becomes `pose`, `cls` becomes `classify`, `deblur`, `denoise` and
`super-resolution` all become `restore`, `face-recognition` and `reid` become
`embed`. An unrecognized name raises rather than silently defaulting.

`segment`, `semantic` and `panoptic` are three different tasks, not three words
for one. Instance masks, per-pixel labels and the merged thing-plus-stuff map
have different ground truth, different metrics and different result fields.

## Model families

A family is one architecture lineage with its own loading, preprocessing and
postprocessing code. Every family declares a `FAMILY` identifier such as
`yolo9`, `rfdetr` or `dfine`, the tasks it supports, and the input resolution
for each size it ships.

`LibreYOLO()` is a factory rather than a class. Given a path it loads the file,
identifies the family from the checkpoint's metadata or, failing that, from the
tensor keys themselves, and returns an instance of that family's model. This is
why swapping detectors is a one-line change: the object that comes back exposes
the same `predict`, `train`, `val` and `export` surface and returns the same
`Results` type.

<code-tabs name="inspect" />

A family that serves more than one task usually publishes a separate checkpoint
per task, often with a different set of sizes for each; a few share one artifact
between two runtime tasks instead. Either way the supported tasks are a fixed
list, and asking for one outside it raises with the supported list in the
message rather than loading something approximate.

The full list, with per-family benchmarks and published weights, is at
[all models](/docs/models).

## Sizes

A size is a variant within a family, written as a lowercase code attached
directly to the family prefix. The common letters are `n` for nano, `t` for
tiny, `s` for small, `m` for medium, `l` for large and `x` for xlarge, but the
codes are family-specific and several families use something else entirely:
backbone-named codes such as `r50` or `r101` where the size is a ResNet depth,
compound-scaling codes such as `b0` through `b3`, or a name that identifies the
one released checkpoint. YOLOv9 uses `c` for compact where other families use
`l`.

Size also fixes the input resolution, and for families with several tasks the
resolution can differ per task. Both are read from the family, never assumed;
`libreyolo models` prints them.

## Checkpoint filenames

Every published weight file follows one schema:

```text
Libre<FAMILY><size>[-<task>].pt
```

The family prefix is a fixed string per family, the size is lowercase and
attached with no separator, and the task suffix is hyphen-prefixed. Detection
carries no suffix, following the convention YOLO checkpoints have always used,
so `LibreYOLO9t.pt` is a detector and `LibreRFDETRn-seg.pt` is a segmentation
model of the same family.

| Task | Suffix |
|---|---|
| `detect` | |
| `segment` | `-seg` |
| `semantic` | `-sem` |
| `panoptic` | `-panoptic` |
| `pose` | `-pose` |
| `classify` | `-cls` |
| `gaze` | `-gaze` |
| `obb` | `-obb` |
| `point` | `-point` |
| `depth` | `-depth` |
| `edge` | `-edge` |
| `normal` | `-normal` |
| `restore` | `-restore` |
| `matte` | `-matte` |
| `ocr` | `-ocr` |
| `embed` | `-embed` |
| `mesh` | `-mesh` |

A family with no suffixless task can require the suffix, so that a name without
one is not accepted as a valid checkpoint for it. A family that publishes
weights trained on a dataset other than its default appends the dataset name as
a further suffix, and that variant stays part of the repository name the file is
downloaded from.

Three tiers stand outside this schema. The promptable segmentation families, the
vision-language families and the open-vocabulary detectors are not registered
into the checkpoint factory and emit no `Libre<FAMILY><size>.pt` file. Their
prefix names a downloaded Hugging Face snapshot or a promptable checkpoint
instead, and upstream brand casing is preserved there on purpose.

## How the task is decided

When several signals could name a task, they are consulted in a fixed order and
the first one that is present wins: the `task` argument you passed, then the
task recorded in the checkpoint metadata, then the task suffix in the filename,
then the family's default task. The result is checked against the family's
supported tasks before the model is built, so a mismatch fails at load time
rather than producing wrong output later.

## Support tiers

Families are enrolled in exactly one tier. A tier is a statement about
engineering attention, not about accuracy: it tells you where a new feature
lands first and what is kept green.

| Tier | What it means |
|---|---|
| Flagship | Features are designed and fully GPU-validated here first |
| Core | Core trainable detectors. Features follow the flagships in the same release wave |
| Supported | Supporting trainable families. Kept green in CI, features land opportunistically |
| Inference only | Predict, validate and export. Training features do not apply |
| Museum | A frozen exhibit. Bug fixes only |
| Sibling tier | A separate product surface with its own factory and contract |

Each model page carries its family's tier in the header. The two flagship
families are [YOLOv9](/docs/models/yolov9) for the CNN detectors and
[RF-DETR](/docs/models/rf-detr) for the transformer detectors; start there
unless you have a reason not to.

Inference only says what is missing, which is a training loop in LibreYOLO.
Predict, validate and, where the family supports it, export all work. Calling
`train()` on such a family raises `NotImplementedError` naming the reason.
