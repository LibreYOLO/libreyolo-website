---
title: MobileSAM
families: [mobilesam]
seo_title: "MobileSAM: lightweight promptable segmentation in LibreYOLO"
description: "Use MobileSAM in LibreYOLO for promptable point and box segmentation with a TinyViT encoder. Install and predict the tiny checkpoint under Apache-2.0."
lead: "MobileSAM replaces SAM's ViT-H image encoder with a distilled TinyViT encoder, so the same promptable point-and-box workflow runs on lighter hardware. LibreYOLO carries a native port of it through a dedicated LibreSAM factory, separate from the LibreYOLO() detector factory."
keywords: [MobileSAM, Segment Anything, TinyViT, promptable segmentation, interactive segmentation, point prompt, box prompt, lightweight segmentation]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Point and box prompts
      language: python
      code: |
        from libreyolo import LibreSAM, SAMPLE_IMAGE

        # MobileSAM has a single size, "tiny", so no other alias is needed.
        model = LibreSAM("mobilesam")

        # A point prompt: [x, y] in pixel coordinates, label 1 = foreground.
        result = model.predict(SAMPLE_IMAGE, points=[640, 420], labels=[1])
        print(result.masks.xy)      # polygon per mask
        print(result.boxes.xyxy)    # tight box derived from the mask

        # A box prompt instead of a point.
        result = model.predict(SAMPLE_IMAGE, bboxes=[300, 200, 900, 700])

        # No prompt at all segments the whole image (a simplified automatic
        # mask generator, not the exhaustive reference one).
        result = model.predict(SAMPLE_IMAGE)
    - label: Encode once, prompt many
      language: python
      code: |
        from libreyolo import LibreMobileSAM, SAMPLE_IMAGE

        model = LibreMobileSAM()

        # The image encoder is the expensive part. set_image() runs it once;
        # every predict() call after that reuses the cached embedding.
        model.set_image(SAMPLE_IMAGE)
        a = model.predict(points=[640, 420], labels=[1])
        b = model.predict(bboxes=[300, 200, 900, 700])
        model.reset_image()
---

## Install

MobileSAM needs the `sam` extra: LibreYOLO's own weight download still goes
through `transformers`' Hugging Face snapshot tooling, even though inference
runs on a native, non-`transformers` decoder.

```bash
pip install "libreyolo[sam]"
```

## Predict

`LibreSAM(...)` (or the family-specific `LibreMobileSAM(...)`) is a separate
entry point from `LibreYOLO(...)`: it returns a promptable segmenter rather
than a detector, because a forward pass here is meaningless without a
spatial prompt. There is no `libreyolo predict` CLI command for this family;
use the Python API.

<code-tabs name="predict" />

A point prompt accepts `[x, y]` for one object, `[[x, y], ...]` for several, or
numpy arrays; `labels` marks each point `1` (foreground) or `0` (background)
and defaults to all foreground. A box prompt takes `[x1, y1, x2, y2]` or a list
of boxes, one mask per box. Omitting both prompts segments the whole image by
prompting a dense grid and keeping the confident, non-overlapping masks; this
"segment everything" mode is simplified against the reference automatic mask
generator and can under-segment crowded scenes, so a real point or box prompt
is the precise path. `conf` filters by predicted mask quality (IoU), not a
detection confidence: pass `0.0` to keep every candidate. `multimask=True`
returns all three of SAM's whole-versus-part ambiguity masks per prompt
instead of the single best one. `device=` moves the model and, if a
`set_image()` session is active, its cached embedding. Every mask carries
class id `0`, named `"object"`, since a promptable mask has no fixed class
set. `train()`, `val()`, `export()` and `track()` all raise
`NotImplementedError` for this family: MobileSAM is predict-only in
LibreYOLO. See [prediction](/docs/predict) for source types.

## Variants

One size, tiny, at a fixed 1024 px input: MobileSAM ships a single TinyViT
encoder rather than the base/large/huge ladder SAM-1 offers.

## Checkpoints

Every published weight file for this family.

<checkpoint-table />

## Licensing

<provenance-box></provenance-box>

## Citation

<citation-block />
