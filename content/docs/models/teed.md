---
title: TEED
families: [teed]
seo_title: "TEED: edge detection, bring your own checkpoint"
description: "Use TEED in LibreYOLO for dense edge-probability prediction. Convert a licensed checkpoint, then predict, validate and export it."
lead: "TEED (Tiny and Efficient Edge Detector) is a small convolutional network that predicts a dense edge-probability map from one RGB image. LibreYOLO wraps its architecture for edge detection only; no checkpoint ships with the library."
keywords: [TEED, Tiny and Efficient Edge Detector, edge detection, BIPED, dense prediction]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreTEEDt-edge.pt")
        results = model(SAMPLE_IMAGE, save=True)

        edges = results[0].edges
        print(edges.array.shape)        # (H, W) float32 in [0, 1]
        print(edges.binary(0.5).sum())  # thresholded edge-pixel count
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=weights/LibreTEEDt-edge.pt source=bus.jpg save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("weights/LibreTEEDt-edge.pt")
        metrics = model.val(data="my-dataset.yaml", imgsz=352)

        print(metrics["metrics/ODS"])   # optimal-dataset-scale F-measure
        print(metrics["metrics/OIS"])   # optimal-image-scale F-measure
    - label: CLI
      language: bash
      code: |
        libreyolo val model=weights/LibreTEEDt-edge.pt data=my-dataset.yaml imgsz=352
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("weights/LibreTEEDt-edge.pt")
        model.export(format="onnx", imgsz=352)
        model.export(format="tensorrt", imgsz=352, half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=weights/LibreTEEDt-edge.pt format=onnx imgsz=352
        libreyolo export model=weights/LibreTEEDt-edge.pt format=tensorrt imgsz=352 half=True
    - label: Use the exported file
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreTEEDt-edge.onnx")
        results = model(SAMPLE_IMAGE)

        print(results[0].edges.array.shape)
---

## Install

TEED needs no optional extra. Everything it imports is in the base install.

```bash
pip install libreyolo
```

## Predict

LibreYOLO ships no TEED checkpoint. The officially released weights are
trained on BIPED, whose published dataset terms restrict use to
non-commercial purposes, so LibreYOLO does not mirror them. Convert a
checkpoint you are licensed to use with `weights/convert_teed_weights.py`,
which checks the tensor keys against the runtime architecture before writing
a file LibreYOLO can load directly:

```bash
python weights/convert_teed_weights.py upstream.pth weights/LibreTEEDt-edge.pt --verify
```

<code-tabs name="predict" />

`results[0].edges` holds the result: an `(H, W)` float32 array in `[0, 1]`,
with `.binary(threshold)` returning a boolean edge mask. There are no boxes,
so `conf`, `iou` and `max_det` have no effect. See
[prediction](/docs/predict) for sources, streaming and result handling.

## Variants

TEED ships one size in LibreYOLO. LibreYOLO's benchmark harness has not
measured this family, so there are no published numbers to compare it
against.

## Validate

`val()` reports BSDS-style ODS and OIS F-measures against a paired edge
dataset: images beside same-stem edge maps, with an optional validity mask so
padded pixels never count. `imgsz` must be divisible by the network's
downsample stride, and LibreYOLO raises a clear error if it is not.

<code-tabs name="val" />

## Export

<export-matrix />

Edge export uses a fixed-resolution, batch-1 runtime contract: `dynamic` and a
`batch` other than 1 are rejected, and the exported graph outputs a single
fused probability map. An exported artifact loads back through `LibreYOLO()`
on its file suffix, so a `.onnx` file behaves like a checkpoint and returns
the same `Results`.

<code-tabs name="export" />

## Licensing

<provenance-box>

LibreYOLO publishes no TEED checkpoint. Nothing is mirrored under the
LibreYOLO organization; convert a checkpoint you hold a license for with
`weights/convert_teed_weights.py` instead.

</provenance-box>

## Citation

<citation-block />
