---
title: MoGe-2
families: [moge2]
seo_title: "MoGe-2: predict, validate and export surface normals"
description: "Use MoGe-2 in LibreYOLO for dense surface-normal prediction. Install, predict, validate and export the official ViT-S, ViT-B and ViT-L checkpoints."
lead: "MoGe-2 is a single-forward monocular geometry model that predicts a dense surface-normal field from one RGB image. LibreYOLO supports it for normal estimation only, through the official ViT-S, ViT-B and ViT-L checkpoints."
keywords: [MoGe-2, MoGe 2, surface normal estimation, monocular geometry, normal map, dense prediction, DINOv2]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreMoGe2s-normal.pt")
        result = model(SAMPLE_IMAGE, save=True)

        normal = result.normal_map
        print(normal.array.shape)   # (H, W, 3) float32 unit vectors
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreMoGe2s-normal.pt source=bus.jpg save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMoGe2s-normal.pt")
        metrics = model.val(data="my-dataset.yaml", imgsz=518)

        print(metrics["metrics/mean_angular_error"])   # degrees
        print(metrics["metrics/median_angular_error"])
        print(metrics["metrics/within_11_25"])          # percent of pixels
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreMoGe2s-normal.pt data=my-dataset.yaml imgsz=518
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMoGe2s-normal.pt")
        model.export(format="onnx", imgsz=518)
        model.export(format="tensorrt", imgsz=518, half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreMoGe2s-normal.pt format=onnx imgsz=518
        libreyolo export model=LibreMoGe2s-normal.pt format=tensorrt imgsz=518 half=True
    - label: Use the exported file
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreMoGe2s-normal.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.normal_map.array.shape)
---

## Install

MoGe-2 needs no optional extra. Everything it imports is in the base install.

```bash
pip install libreyolo
```

## Predict

Weights download automatically on first use: LibreYOLO fetches the matching
size directly from the official checkpoints and caches it locally.

<code-tabs name="predict" />

MoGe-2 returns a dense field rather than a set of detections, so
`result.boxes` is empty and `conf`, `iou` and `max_det` have no effect.
`result.normal_map` holds the result: an `(H, W, 3)` array of unit vectors
in the OpenCV camera frame, where `+x` is right, `+y` is down, `+z` is into the
scene, and a surface facing the camera reads `(0, 0, -1)`. Predicting a list of
images runs one forward pass per image; this family has no stacked-batch fast
path. See [prediction](/docs/predict) for sources, streaming and result
handling.

## Variants

Three encoder sizes ship as separate checkpoints: ViT-S, ViT-B and ViT-L, all
at the same input resolution. LibreYOLO's benchmark harness has not measured
this family, so there are no published accuracy numbers to compare them by;
pick a size against your own compute budget.

## Validate

`val()` measures angular error against a paired normal-map dataset: images
beside same-stem 16-bit normal PNGs, with an optional validity mask so padded
and invalid pixels never count. It returns the mean and median angular error
in degrees, plus the percentage of pixels within 11.25, 22.5 and 30 degrees.

<code-tabs name="val" />

## Export

<export-matrix />

Normal export uses a fixed-resolution, batch-1 runtime contract: `dynamic` and
a `batch` other than 1 are rejected, and `imgsz` must be divisible by the ViT
encoder's patch size, which LibreYOLO checks before the run starts. An
exported artifact loads back through `LibreYOLO()` on its file suffix, so a
`.onnx` file behaves like a checkpoint and returns the same `Results`.

<code-tabs name="export" />

## Licensing

<provenance-box>

LibreYOLO does not copy these checkpoints into its own organization.
`LibreYOLO("LibreMoGe2s-normal.pt")` downloads the matching size directly from
the official Hugging Face repositories at a pinned revision, and verifies the
file against a recorded SHA-256 checksum before use.

</provenance-box>

## Citation

<citation-block />
