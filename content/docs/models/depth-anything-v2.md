---
title: Depth Anything V2
families: [depth_anything]
seo_title: "Depth Anything V2: predict and validate monocular depth"
description: "Use Depth Anything V2 in LibreYOLO for monocular depth estimation. Install, predict and validate; Small ships Apache-2.0, Base and Large are CC-BY-NC-4.0."
lead: "Depth Anything V2 is a DINOv2 encoder paired with a DPT decoder that predicts a dense relative inverse-depth map from a single image. LibreYOLO supports it for the depth task: predict and zero-shot validation, with no training path."
keywords: [Depth Anything V2, monocular depth estimation, DPT, DINOv2, relative depth, depth map]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")
        result = model(SAMPLE_IMAGE, save=True)

        depth = result.depth_map
        print(depth.min, depth.max, depth.mean)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreDepthAnythingV2s-depth.pt source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg save=True
    - label: Read the depth map
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")
        result = model(SAMPLE_IMAGE)

        depth = result.depth_map    # DepthMap: dense (H, W), higher = closer
        raw = depth.data                # tensor, no metric unit or cross-image scale
        normalized = depth.normalized() # rescaled to [0, 1] for visualization
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/abs_rel"])
        print(metrics["metrics/rmse"])
        print(metrics["metrics/delta1"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDepthAnythingV2s-depth.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreDepthAnythingV2s-depth.pt format=onnx
        libreyolo export model=LibreDepthAnythingV2s-depth.pt format=tensorrt half=True
    - label: Use the exported file
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # The factory routes on the file suffix, so an exported artifact loads
        # like any checkpoint and returns the same Results object.
        model = LibreYOLO("LibreDepthAnythingV2s-depth.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.depth_map.data.shape)
---

## Install

Depth Anything V2 needs no optional extra. Everything it imports is in the base install.

```bash
pip install libreyolo
```

## Predict

Weights download from Hugging Face on first use and are cached locally.

<code-tabs name="predict" />

`result.depth_map` carries a dense relative inverse-depth map: higher values
mean closer to the camera, and the values have no metric unit or cross-image
scale. `save=True` writes a colormapped visualization of that map to disk;
`Results.plot()` does not cover this family, since it is defined for surface
normals and edges only. Input resolution must divide evenly by 14, the DINOv2
patch grid the DPT head builds on; LibreYOLO checks this before running and
raises if it does not. See [prediction](/docs/predict) for sources, streaming
and result handling.

## Variants

Four encoder sizes, s/b/l/g, corresponding to ViT-S/B/L/G. The checkpoint table
below lists only s, b and l; no Giant checkpoint is published. All four share
the same input resolution, so choosing a size trades encoder capacity, not
image size. Licensing is also a factor: the Small checkpoint is Apache-2.0,
while Base and Large are CC-BY-NC-4.0, see Licensing below.

Training and fine-tuning are not offered for this family. `LibreDepthAnythingV2.train()`
raises `NotImplementedError` unconditionally; convert a compatible upstream
checkpoint instead, with `weights/convert_depth_anything_v2_weights.py`.

## Validate

`val()` runs the shared depth validator: it aligns each prediction to its
ground truth with a per-image least-squares scale and shift, then reports the
standard zero-shot relative-depth metrics, AbsRel, RMSE and the three delta
thresholds.

<code-tabs name="val" />

## Export

<export-matrix />

An exported artifact loads back through `LibreYOLO()` on its file suffix, so a
`.onnx` or `.engine` file behaves like a checkpoint and returns the same
`Results`, with `depth_map` in place of boxes. [Export](/docs/export) lists the
arguments every format accepts.

<code-tabs name="export" />

## Checkpoints

Every published weight file for this family.

<checkpoint-table />

## Licensing

<provenance-box></provenance-box>

## Citation

<citation-block />
