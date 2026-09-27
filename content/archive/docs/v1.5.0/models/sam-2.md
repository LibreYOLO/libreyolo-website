---
title: SAM 2
families: [sam2]
seo_title: "SAM 2: promptable image segmentation in LibreYOLO"
description: "Use SAM 2 in LibreYOLO for promptable point and box segmentation. Install and predict the tiny, small, base-plus and large checkpoints, Apache-2.0."
lead: "SAM 2 extends SAM with a streaming-memory architecture built for video, and turns a point or box click into an object mask. LibreYOLO supports its image segmentation path through a dedicated LibreSAM factory, separate from the LibreYOLO() detector factory."
keywords: [SAM 2, Segment Anything, promptable segmentation, interactive segmentation, point prompt, box prompt, Meta AI, Hiera]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Point and box prompts
      language: python
      code: |
        from libreyolo import LibreSAM, SAMPLE_IMAGE

        # Size aliases: "sam2-tiny", "sam2-small", "sam2-base-plus",
        # "sam2-large" (also the short forms "sam2-t"/"sam2-s"/"sam2-bp"/"sam2-l").
        model = LibreSAM("sam2-large")

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
        from libreyolo import LibreSAM2, SAMPLE_IMAGE

        # The family-specific class takes the size without the "sam2-" prefix.
        model = LibreSAM2("large")

        # The image encoder is the expensive part. set_image() runs it once;
        # every predict() call after that reuses the cached embedding.
        model.set_image(SAMPLE_IMAGE)
        a = model.predict(points=[640, 420], labels=[1])
        b = model.predict(bboxes=[300, 200, 900, 700])
        model.reset_image()
---

## Install

SAM 2 needs the `sam` extra, which pulls in `transformers` and `timm`.

```bash
pip install "libreyolo[sam]"
```

## Predict

`LibreSAM(...)` (or the family-specific `LibreSAM2(...)`) is a separate entry
point from `LibreYOLO(...)`: it returns a promptable segmenter rather than a
detector, because a forward pass here is meaningless without a spatial
prompt. There is no `libreyolo predict` CLI command for this family; use the
Python API. Only image segmentation is supported; SAM 2's video-memory
tracking is out of scope here.

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
`NotImplementedError` for this family: image inference is what LibreYOLO
supports here. See [prediction](/docs/predict) for source types.

## Variants

Four Hiera-backbone sizes: tiny, small, base-plus and large, all at the same
input resolution. No accuracy or latency benchmark is published for this
family yet, so choosing a size trades encoder weight for mask quality
directly: tiny is the fastest to encode, large the heaviest.

## Checkpoints

Every published weight file for this family.

<checkpoint-table />

## Licensing

<provenance-box></provenance-box>

## Citation

<citation-block />
