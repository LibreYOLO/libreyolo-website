---
title: Promptable segmentation API
seo_title: 'LibreSAM API: prompts, aliases and signatures'
description: >-
  The LibreSAM factory, its size aliases, the point, box and concept-text prompt
  types, the encode-once set_image lifecycle, and what the tier does not
  support.
lead: >-
  LibreSAM is the factory for promptable segmentation. A forward pass needs a
  per-image prompt supplied at call time, so the tier owns its own predict
  surface rather than routing through the promptless inference runner.
keywords:
  - LibreSAM
  - promptable segmentation
  - SAM point prompt
  - SAM box prompt
  - set_image
  - segment everything
  - libreyolo sam extra
last_verified: 1.5.0
verification: >-
  Factory aliases, sizes and repositories read from
  libreyolo/models/sam/model.py, sam2.py, edgetam.py, sam3.py,
  libreyolo/models/mobilesam/model.py and libreyolo/models/picosam3/model.py.
  Prompt contract and defaults read from libreyolo/models/sam/base.py. Design
  intent from docs/adr/0007-libresam-contract.md, all at v1.5.0.
snippets:
  install:
    - label: bash
      language: bash
      code: |
        pip install 'libreyolo[sam]'
  usage:
    - label: Point and box prompts
      language: python
      code: |
        from libreyolo import LibreSAM, SAMPLE_IMAGE

        model = LibreSAM("base")

        r = model.predict(SAMPLE_IMAGE, points=[900, 370], labels=[1])
        print(r.masks.xy)
        print(r.boxes.xyxy)

        r = model.predict(SAMPLE_IMAGE, bboxes=[100, 100, 200, 200])
        print(len(r))
    - label: 'Encode once, prompt many times'
      language: python
      code: |
        from libreyolo import LibreSAM, SAMPLE_IMAGE

        model = LibreSAM("base")
        model.set_image(SAMPLE_IMAGE)

        a = model.predict(points=[500, 375], labels=[1])
        b = model.predict(bboxes=[100, 100, 200, 200])
        print(len(a), len(b))

        model.reset_image()
source_hash: 18e8206c10ce17fd
---

## Install

The tier needs the `sam` extra.

<code-tabs name="install" />

## The factory

```python
LibreSAM(model: str = "base", **kwargs) -> LibreSAMModel
```

`model` is a size alias, not a path. `**kwargs` reaches the family
constructor, which takes `device` and `multimask`. An unknown alias raises
`ValueError` and the message lists every known alias.

<code-tabs name="usage" />

## Aliases

| Family | Aliases | Sizes | Weights |
|---|---|---|---|
| SAM-1 | `base`, `large`, `huge`, `b`, `l`, `h`, `sam-base`, `sam-large`, `sam-huge`, `sam_b`, `sam_l`, `sam_h` | `base`, `large`, `huge` | `facebook/sam-vit-base`, `-large`, `-huge` |
| SAM-2 | `sam2-tiny`, `sam2-small`, `sam2-base-plus`, `sam2-baseplus`, `sam2-large`, and the short forms `sam2-t`, `sam2-s`, `sam2-bp`, `sam2-l`, `sam2_t`, `sam2_s`, `sam2_bp`, `sam2_l` | `tiny`, `small`, `base-plus`, `large` | `LibreYOLO/LibreSAM2tiny`, `-small`, `-base-plus`, `-large` |
| EdgeTAM | `edgetam`, `edge-tam`, `edgetam-edge` | `edge` | `LibreYOLO/LibreEdgeTAM` |
| SAM 3 | `sam3`, `sam-3`, `sam3-large` | `large` | `facebook/sam3` |
| MobileSAM | `mobilesam`, `mobilesam-tiny`, `mobilesam_t`, `mobile-sam`, `mobile-sam-tiny` | `tiny` | `LibreYOLO/LibreMobileSAM` |
| PicoSAM3 | `picosam3`, `picosam3-pico`, `picosam3_pico`, `pico-sam3` | `pico` | `LibreYOLO/LibrePicoSAM3` |

The default is `base`. SAM-1, SAM-2, EdgeTAM and MobileSAM run at a nominal
1024 pixel canvas, SAM 3 at 1008, PicoSAM3 at 96.

SAM 3 weights are gated. They download from `facebook/sam3` under Meta's
custom SAM License, which is neither MIT nor Apache-2.0 and is not
redistributed by LibreYOLO. Accept the terms on the repository page and
authenticate with Hugging Face before loading; the loader logs the notice
first.

The family classes are exported too, so `LibreSAM1`, `LibreSAM2`,
`LibreSAM3`, `LibreEdgeTAM`, `LibreMobileSAM` and `LibrePicoSAM3` can be
constructed directly with `size=`.

## predict

```python
model.predict(
    source=None,
    *,
    points=None,
    bboxes=None,
    labels=None,
    masks=None,
    text=None,
    conf=None,
    multimask=None,
    max_det=300,
    device=None,
    color_format="auto",
    points_per_side=None,
) -> Results
```

| Argument | Default | Meaning |
|---|---|---|
| `source` | `None` | Image to segment; `None` reuses the image cached by `set_image()` |
| `points` | `None` | Point prompt in pixel coordinates |
| `bboxes` | `None` | Box prompt as `[x1, y1, x2, y2]`, or a list of them for one mask per box |
| `labels` | `None` | Point labels, `1` positive and `0` negative, shaped to match `points`; all positive when omitted |
| `masks` | `None` | Reserved; passing one raises `NotImplementedError` |
| `text` | `None` | Concept prompt; SAM 3 only |
| `conf` | `None` | Predicted mask-IoU floor |
| `multimask` | `None` | Return all ambiguity masks per prompt; defaults to the construction setting |
| `max_det` | `300` | Cap on returned masks |
| `device` | `None` | Move the model for this and later calls, invalidating cached embeddings |
| `color_format` | `"auto"` | Color format hint for in-memory arrays |
| `points_per_side` | `None` | Grid density for segment-everything; defaults to 32 |

The return is an ordinary `Results` carrying `masks`, plus tight `boxes`
derived from those masks, with class `0` named `"object"`.

## Prompt shapes

`points` accepts the nested forms `[x, y]` for one object, `[[x, y], ...]` for
N objects, and `[[[x, y], ...], ...]` for points grouped per object. Numpy
arrays work everywhere a list does. Coordinates are plain pixels on the source
image.

Omitting every spatial prompt runs segment-everything, a grid automatic mask
generator with a predicted-IoU threshold and box-IoU deduplication. The
default `points_per_side` of 32 runs roughly 1024 decoder passes, which is
slow on CPU; lower it for interactive use. The generator omits
stability-score filtering, multi-crop and mask-IoU deduplication, so it is an
approximation of the prompted path rather than a match for it.

## Confidence

`conf` filters by predicted mask-IoU, which is a mask-quality score and not a
detection confidence. `None` keeps every mask in the prompted path and applies
the family grid threshold in segment-everything. `0.0` disables filtering in
either mode.

On SAM 3's text path, `conf` is the Promptable Concept Segmentation detection
score instead. `None` there means the standard 0.3 threshold, and `0.0` keeps
all candidates.

## Text prompts

`text=` is SAM 3 only; every spatial-prompt family raises
`NotImplementedError` for it. Text is mutually exclusive with points and
boxes. The returned `names` maps class `0` to the requested concept. A text
call with `source=None` re-encodes the cached image, because the tracker and
the concept encoder do not share a cache.

The keyword `exemplars=` is reserved for a future image-exemplar extension and
is not implemented.

## The encode-once lifecycle

```python
model.set_image(source, color_format="auto") -> LibreSAMModel
model.reset_image() -> LibreSAMModel
```

`set_image` runs the heavy image encoder once and caches the embeddings, so
every later `predict()` with `source=None` is cheap. Both methods return the
model so calls can chain. Passing `device=` to `predict` moves the model and
invalidates the cache.

## PicoSAM3

PicoSAM3 accepts `bboxes=` only. Point, text, mask, multimask and
segment-everything prompts raise. The box is expanded by 10 percent and run
through a 96 pixel ROI network, and PicoSAM3 is the one family in the tier
that exports, to ONNX only.

## Not supported

`train()`, `val()` and `track()` raise `NotImplementedError` on every family
in the tier. Promptable masks have no fixed class set to score against, so
mAP has no meaning here. `export()` raises for SAM-1, SAM-2, SAM 3, EdgeTAM
and MobileSAM.

Video and memory paths for SAM-2, SAM 3 and EdgeTAM are out of scope for this
version, as are SAM 3 image exemplars and mask prompts.


