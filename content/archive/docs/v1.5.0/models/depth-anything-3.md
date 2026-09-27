---
title: Depth Anything 3
families: [depth_anything3]
seo_title: "Depth Anything 3: predict monocular depth in LibreYOLO"
description: "Use Depth Anything 3 in LibreYOLO for monocular depth estimation. Install, predict, validate and export the DA3MONO-LARGE checkpoint, Apache-2.0."
lead: "Depth Anything 3 is a plain DINOv2 transformer trained to predict depth and camera geometry from one or more views with no architectural specialization. LibreYOLO ports its DA3MONO-LARGE checkpoint for the depth task: predict and zero-shot validation, with no training path."
keywords: [Depth Anything 3, DA3, monocular depth estimation, DINOv2, relative depth, depth map]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDepthAnything3l-depth.pt")
        result = model(SAMPLE_IMAGE, save=True)

        depth = result.depth_map
        print(depth.min, depth.max, depth.mean)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreDepthAnything3l-depth.pt source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg save=True
    - label: Read the depth map
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDepthAnything3l-depth.pt")
        result = model(SAMPLE_IMAGE)

        depth = result.depth_map    # DepthMap: dense (H, W), higher = closer
        raw = depth.data                # tensor, no metric unit or cross-image scale
        normalized = depth.normalized() # rescaled to [0, 1] for visualization
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDepthAnything3l-depth.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/abs_rel"])
        print(metrics["metrics/rmse"])
        print(metrics["metrics/delta1"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDepthAnything3l-depth.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDepthAnything3l-depth.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreDepthAnything3l-depth.pt format=onnx
        libreyolo export model=LibreDepthAnything3l-depth.pt format=tensorrt half=True
    - label: Use the exported file
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # The factory routes on the file suffix, so an exported artifact loads
        # like any checkpoint and returns the same Results object.
        model = LibreYOLO("LibreDepthAnything3l-depth.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.depth_map.data.shape)
---

## Install

Depth Anything 3 needs no optional extra. Everything it imports is in the base
install.

```bash
pip install libreyolo
```

## Predict

Weights download from Hugging Face on first use and are cached locally.

<code-tabs name="predict" />

`result.depth_map` carries a dense relative inverse-depth map: higher
values mean closer to the camera, and the values have no metric unit or
cross-image scale. The upstream checkpoint emits positive relative depth;
LibreYOLO's network wrapper inverts it and reproduces the official sky
handling so the output follows LibreYOLO's shared depth contract. `save=True`
writes a colormapped visualization of that map to disk; `Results.plot()` does
not cover this family, since it is defined for surface normals and edges only.
See [prediction](/docs/predict) for sources, streaming and result handling.

## Variants

One size, `l`, at a fixed input resolution. Upstream DA3 also publishes Small
and Base any-view checkpoints, a metric-depth checkpoint, and Nested and Giant
checkpoints; LibreYOLO exposes none of them. Metric depth needs a different
public contract than LibreYOLO's relative-inverse-depth task, and the any-view
and Nested checkpoints need a multi-image camera API LibreYOLO does not offer.
The Large and Giant any-view checkpoints are also CC-BY-NC-4.0 and are not
referenced by any LibreYOLO download path.

Training is not offered for this family. `LibreDepthAnything3.train()` raises
`NotImplementedError` unconditionally; train upstream and convert a compatible
DA3MONO-LARGE checkpoint with `weights/convert_depth_anything3_weights.py`.

## Validate

`val()` runs the shared depth validator: it aligns each prediction to its
ground truth with a per-image least-squares scale and shift, then reports the
standard zero-shot relative-depth metrics, AbsRel, RMSE and the three delta
thresholds.

<code-tabs name="val" />

## Export

<export-matrix />

Export is restricted to five formats for this family: ONNX, TorchScript,
ExecuTorch, TensorRT and OpenVINO. Requesting any other format raises
`NotImplementedError` rather than attempting an unvalidated conversion. An
exported artifact loads back through `LibreYOLO()` on its file suffix, so a
`.onnx` or `.engine` file behaves like a checkpoint and returns the same
`Results`, with `depth_map` in place of boxes.

<code-tabs name="export" />

## Checkpoints

Every published weight file for this family.

<checkpoint-table />

## Licensing

<provenance-box></provenance-box>

## Citation

<citation-block />
