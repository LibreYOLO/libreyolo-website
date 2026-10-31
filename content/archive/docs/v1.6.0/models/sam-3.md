---
title: SAM 3
families: [sam3]
seo_title: "SAM 3: promptable and concept segmentation in LibreYOLO"
description: "Use SAM 3 in LibreYOLO for point, box and text-concept segmentation. Install and predict the large checkpoint, gated under Meta's SAM License."
lead: "SAM 3 extends SAM with a text-concept prompt on top of the usual points and boxes, so a phrase like \"yellow school bus\" returns every matching instance. LibreYOLO supports its image path through a dedicated LibreSAM factory, separate from the LibreYOLO() detector factory."
keywords: [SAM 3, Segment Anything, promptable segmentation, concept segmentation, text prompt, point prompt, box prompt, Meta AI]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Point and box prompts
      language: python
      code: |
        from libreyolo import LibreSAM, SAMPLE_IMAGE

        # "sam3" is the only size ("large"); aliases: "sam3", "sam-3", "sam3-large".
        model = LibreSAM("sam3")

        # A point prompt: [x, y] in pixel coordinates, label 1 = foreground.
        result = model.predict(SAMPLE_IMAGE, points=[640, 420], labels=[1])
        print(result.masks.xy)      # polygon per mask
        print(result.boxes.xyxy)    # tight box derived from the mask

        # A box prompt instead of a point.
        result = model.predict(SAMPLE_IMAGE, bboxes=[300, 200, 900, 700])
    - label: Text (concept) prompt
      language: python
      code: |
        from libreyolo import LibreSAM3, SAMPLE_IMAGE

        model = LibreSAM3("large")

        # Finds every instance matching the phrase, not just one object.
        # text= is mutually exclusive with points, bboxes, labels and masks.
        result = model.predict(SAMPLE_IMAGE, text="a person")
        print(result.names)         # {0: "a person"}
        print(result.boxes.conf)    # the PCS detection score per instance
    - label: Encode once, prompt many
      language: python
      code: |
        from libreyolo import LibreSAM3, SAMPLE_IMAGE

        model = LibreSAM3("large")

        # The image encoder is the expensive part. set_image() runs it once;
        # every predict() call after that reuses the cached embedding. A
        # text= call re-encodes internally, since the tracker and the
        # concept-segmentation encoder do not share a cache.
        model.set_image(SAMPLE_IMAGE)
        a = model.predict(points=[640, 420], labels=[1])
        b = model.predict(bboxes=[300, 200, 900, 700])
        model.reset_image()
---

## Install

SAM 3 needs the `sam` extra, which pulls in `transformers` and `timm`.

```bash
pip install "libreyolo[sam]"
```

The weights are gated: visit
[huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3), accept
Meta's SAM License, then run `hf auth login` (or set `HF_TOKEN`) before the
first download. LibreYOLO logs a license notice the first time it downloads
this family.

## Predict

`LibreSAM(...)` (or the family-specific `LibreSAM3(...)`) is a separate entry
point from `LibreYOLO(...)`: it returns a promptable segmenter rather than a
detector, because a forward pass here is meaningless without a prompt. There
is no `libreyolo predict` CLI command for this family; use the Python API.
Only image inference is supported; SAM 3's video models are out of scope
here.

<code-tabs name="predict" />

The point and box path matches the rest of the SAM family: a point prompt
accepts `[x, y]` for one object or `[[x, y], ...]` for several, `labels` marks
each point `1` (foreground) or `0` (background), and a box prompt takes
`[x1, y1, x2, y2]` or a list of boxes. `conf` on this path filters by
predicted mask quality (IoU), not a detection confidence.

The `text=` path is SAM 3's addition: a concept string returns every matching
instance in the image through Promptable Concept Segmentation, and cannot be
combined with points, boxes, labels or masks. `conf` there is the PCS
detection score instead of mask IoU; leaving it at the default applies the
model's own 0.3 threshold, and `conf=0.0` keeps every candidate. The returned
`names` maps class id `0` to the requested concept string, since a promptable
mask has no fixed class set otherwise. `device=` moves the model and, if a
`set_image()` session is active, its cached embedding. `train()`, `val()`,
`export()` and `track()` all raise `NotImplementedError` for this family: SAM
3 is predict-only in LibreYOLO, and video tracking is out of scope. See
[prediction](/docs/predict) for source types.

## Variants

One size, large, at a fixed 1008 px input. SAM 3.1 is not supported: its
implementation carries a custom license that cannot be vendored into this
MIT repository, and the Transformers version LibreYOLO depends on does not
yet load its checkpoint format.

## Licensing

<provenance-box>

LibreYOLO does not host its own copy of the SAM 3 weights and does not
redistribute them. `LibreSAM("sam3")` downloads directly from Meta's gated
`facebook/sam3` repository on Hugging Face, which requires accepting Meta's
SAM License and authenticating before the first download.

</provenance-box>

## Citation

<citation-block />
