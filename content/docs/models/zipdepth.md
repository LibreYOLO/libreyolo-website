---
title: ZipDepth
families: [zipdepth]
seo_title: "ZipDepth: lightweight monocular depth in LibreYOLO"
description: "Use ZipDepth in LibreYOLO for lightweight monocular depth estimation. Install, predict, validate and export two MIT-licensed checkpoints."
lead: "ZipDepth is a compact reparameterizable CNN distilled from Depth Anything V2 Large that predicts a dense relative inverse-depth map. LibreYOLO supports it for the depth task: predict and zero-shot validation, with no training path."
keywords: [ZipDepth, monocular depth estimation, edge depth model, relative depth, depth map, reparameterizable CNN]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreZipDepthb-depth.pt")
        results = model(SAMPLE_IMAGE, save=True)

        depth = results[0].depth_map
        print(depth.min, depth.max, depth.mean)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreZipDepthb-depth.pt source=bus.jpg save=True
    - label: NPU/edge checkpoint
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Same encoder, an unfold-free upsampling head for compilers that lack
        # gather/unfold support. Output is visually equivalent to the b checkpoint.
        model = LibreYOLO("LibreZipDepthbnpu-depth.pt")
        results = model(SAMPLE_IMAGE, save=True)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreZipDepthb-depth.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/abs_rel"])
        print(metrics["metrics/rmse"])
        print(metrics["metrics/delta1"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreZipDepthb-depth.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreZipDepthb-depth.pt")
        model.export(format="onnx")
        model.export(format="ncnn")
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreZipDepthb-depth.pt format=onnx
        libreyolo export model=LibreZipDepthbnpu-depth.pt format=ncnn
    - label: Use the exported file
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # The factory routes on the file suffix, so an exported artifact loads
        # like any checkpoint and returns the same Results object.
        model = LibreYOLO("LibreZipDepthb-depth.onnx")
        results = model(SAMPLE_IMAGE)

        print(results[0].depth_map.data.shape)
---

## Install

ZipDepth needs no optional extra. Everything it imports is in the base
install.

```bash
pip install libreyolo
```

## Predict

Weights download from Hugging Face on first use and are cached locally.

<code-tabs name="predict" />

`results[0].depth_map` carries a dense relative inverse-depth map: higher
values mean closer to the camera, and the values have no metric unit or
cross-image scale. `save=True` writes a colormapped visualization of that map
to disk; `Results.plot()` does not cover this family, since it is defined for
surface normals and edges only. See [prediction](/docs/predict) for sources,
streaming and result handling.

## Variants

Two checkpoints, both the same encoder capacity, differing only in the trained
upsampling head. `b` uses convex upsampling and runs on GPU or CPU. `bnpu`
swaps in an unfold-free decoder for NPU and edge compilers that lack
gather/unfold support; its output is documented as visually equivalent to `b`.
Pick `bnpu` when the export target is a constrained runtime, `b` otherwise.

Both checkpoints were distilled from Depth Anything V2 Large pseudo-labels, so
this family is the compact, edge-oriented tier of LibreYOLO's depth task,
alongside the larger Depth Anything V2 encoders.

Training is not offered for this family. `LibreZipDepth.train()` raises
`NotImplementedError` unconditionally: the upstream recipe distills pseudo-labels
over a large image set that is not reproducible as a LibreYOLO training run.
Train upstream at [fabiotosi92/ZipDepth](https://github.com/fabiotosi92/ZipDepth)
and convert the result with `weights/convert_zipdepth_weights.py`.

## Validate

`val()` runs the shared depth validator: it aligns each prediction to its
ground truth with a per-image least-squares scale and shift, then reports the
standard zero-shot relative-depth metrics, AbsRel, RMSE and the three delta
thresholds.

<code-tabs name="val" />

## Export

<export-matrix />

Export follows a fixed-resolution dense contract: the source image is
stretch-resized to the exported canvas, and the returned depth map is resized
back to the original canvas afterward. An exported artifact loads back through
`LibreYOLO()` on its file suffix, so a `.onnx` or `.ncnn` file behaves like a
checkpoint and returns the same `Results`, with `depth_map` in place of boxes.

<code-tabs name="export" />

## Checkpoints

Every published weight file for this family.

<checkpoint-table />

## Licensing

<provenance-box></provenance-box>

## Citation

<citation-block />
