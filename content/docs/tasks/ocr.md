---
title: OCR
seo_title: "OCR: text detection and recognition in LibreYOLO"
description: "Find and read text in images with LibreYOLO. Predict quads and transcripts, label a JSONL dataset, and validate with hmean, end-to-end F1 and 1-NED."
lead: "OCR locates text in an image and reads it. LibreYOLO exposes it as the ocr task, which returns one four-point polygon plus one transcript per text region, in reading order."
keywords: [ocr python library, scene text recognition, text detection quads, PP-OCRv5 python, end-to-end text spotting]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Read the text in an image
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # The t tier is the lighter of the two, built for CPU. SAMPLE_IMAGE
        # keeps this runnable; point it at an image with text of your own.
        model = LibreYOLO("LibrePPOCRt-ocr.pt")
        results = model(SAMPLE_IMAGE)

        regions = results[0].ocr
        print(len(regions), "regions")
        for text, score in zip(regions.texts, regions.conf):
            print(repr(text), float(score))
    - label: Read the quads
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibrePPOCRt-ocr.pt")
        results = model(SAMPLE_IMAGE)

        regions = results[0].ocr
        print(regions.data.shape)   # (N, 4, 2) polygons, TL TR BR BL
        print(regions.xyxy)         # axis-aligned hulls of those polygons
        print(regions.det_conf)     # detection score, separate from .conf
    - label: Filter by recognition confidence
      language: python
      code: |
        import numpy as np
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibrePPOCRt-ocr.pt")
        results = model(SAMPLE_IMAGE)

        # Index with positions, not a boolean mask: slicing carries the
        # transcripts and both score arrays along with the geometry.
        regions = results[0].ocr.numpy()
        keep = regions[np.flatnonzero(regions.conf >= 0.9)]
        print(keep.texts)
  val:
    - label: Validate and read the metric keys
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePPOCRt-ocr.pt")
        metrics = model.val(data="my-ocr-dataset")

        print(metrics["metrics/det_precision"], metrics["metrics/det_recall"])
        print(metrics["metrics/det_hmean"])
        print(metrics["metrics/e2e_f1"])       # fitness
        print(metrics["metrics/rec_1-NED"])
---

## Definition

The `ocr` task does two things in one call: it locates every text region in an
image and transcribes it. Regions come back as four-point polygons rather than
axis-aligned boxes, because scene text is often rotated, and in reading order,
top to bottom then left to right.

A prediction fills `results[0].ocr`, an `OCRRegions` payload. `.data` is an
`(N, 4, 2)` float array of polygons in original-image pixels, ordered top-left,
top-right, bottom-right, bottom-left; `.texts` is the list of N transcripts;
`.conf` is the per-region recognition score and `.det_conf` the detection score;
`.xyxy` gives the axis-aligned hull of each polygon. Because the quads are
genuine polygons, they do not populate `results[0].boxes`. Slicing an
`OCRRegions` carries the transcripts and both score arrays along with the
geometry.

## Models

Two families serve `ocr`.

[PP-OCRv5](/docs/models/pp-ocrv5) is the dedicated pipeline: a
differentiable-binarization detector finds the text quads and an SVTR/CTC
recognizer reads them, with both stages bundled into one `.pt` file along with
the recognition charset. It ships in two tiers, a lighter one for CPU and a
server one for higher accuracy, and one dictionary covers Simplified and
Traditional Chinese, English, Japanese and pinyin.

[SenseNova-Vision](/docs/models/sensenova-vision) reaches OCR by generating the
words as tagged text from the same 7B checkpoint that serves its six other
tasks, loaded with `LibreVLM("sensenova-vision", task="ocr")`. It needs the
`sensenova` extra, and its weights are restricted to non-commercial use; the
license is on its page.

## Predict

Weights download from Hugging Face on first use and are cached locally.

<code-tabs name="predict" />

PP-OCRv5 runs detection at a fixed long-side limit and then recognizes the
cropped regions in batches, with `rec_batch` controlling how many crops go
through the recognizer per forward pass. Multi-image sources run sequentially,
because a two-stage pipeline does not batch across images. See
[prediction](/docs/predict) for sources, streaming and result handling.

## Dataset format

OCR labels are one JSONL file per split, one JSON object per image, beside the
images themselves.

```text
my-ocr-dataset/
  images/
    val/receipt.jpg
  labels/
    val.jsonl
```

Each line names an image and lists its regions:

```json
{"image": "receipt.jpg", "regions": [{"polygon": [[10, 12], [118, 14], [117, 40], [9, 38]], "text": "TOTAL 12.50"}]}
```

`polygon` is a four-point quad in absolute pixel coordinates, ordered top-left,
top-right, bottom-right, bottom-left. A region whose text cannot be read is
labeled `"text": "###"`, the ICDAR don't-care convention: it is excluded from
recognition scoring, and a prediction overlapping it is ignored rather than
counted as a false positive.

Passing the root directory as `data=` is enough. A dataset YAML is the
alternative, with `path` plus optional `images` and `labels` directory names,
and `nc: 1` with `names: {0: text}` as schema placeholders, since an OCR model
returns `Results.ocr` rather than detections. See
[dataset formats](/docs/reference/dataset-formats) for the full contract.

## Train

Neither OCR family has a training implementation: `train()` raises
`NotImplementedError` on both, and OCR support covers prediction and validation
only. PP-OCRv5's page names the Apache-2.0 upstream training code and the
conversion script that brings a fine-tuned checkpoint back into LibreYOLO.

## Validate

`val()` scores the whole pipeline, detection and recognition together, matching
predicted polygons to ground-truth polygons one-to-one at IoU above 0.5.

<code-tabs name="val" />

`metrics/det_precision`, `metrics/det_recall` and `metrics/det_hmean` score
localization alone: a match needs only the polygon overlap, whatever the
transcript says. `metrics/e2e_precision`, `metrics/e2e_recall` and
`metrics/e2e_f1` add the reading: a match needs the same polygon overlap and an
exact transcript match after NFKC normalization and whitespace removal, and
comparison stays case-sensitive. `metrics/e2e_f1` is also `fitness`, the number
best-checkpoint selection reads.

`metrics/rec_1-NED` grades the recognizer on its own, over the pairs detection
already matched: one minus the normalized edit distance, so a transcript off by
a character scores near 1 where end-to-end F1 scores it 0.

## Export

No export format is available for this task. PP-OCRv5 is two networks moving
together rather than one traceable graph, and `export()` raises for every format
on both families. To deploy outside LibreYOLO, fine-tune upstream and use the
upstream deployment path.
