---
title: Surface normals
seo_title: "Surface normal estimation in LibreYOLO"
description: "Predict a dense surface-normal field from one image in LibreYOLO. Read the camera frame convention, validate angular error, and export a model."
lead: "Surface-normal estimation predicts the direction each visible surface faces. LibreYOLO exposes it as the normal task, which returns a dense field of unit vectors on the original image canvas."
keywords: [surface normal estimation python, normal map from image, monocular geometry, angular error metric, dense normal prediction]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Predict a normal field
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreMoGe2s-normal.pt")
        results = model(SAMPLE_IMAGE, save=True)

        normals = results[0].normal_map
        print(normals.data.shape)      # (H, W, 3) float32 unit vectors
        normals.assert_normalized()    # raises if any pixel is not unit length
    - label: Read one pixel
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreMoGe2s-normal.pt")
        results = model(SAMPLE_IMAGE)

        # OpenCV camera frame: +x right, +y down, +z into the scene. A surface
        # facing the camera reads close to (0, 0, -1).
        field = results[0].normals.data
        h, w = field.shape[:2]
        print(field[h // 2, w // 2])
    - label: Save the visualization
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreMoGe2s-normal.pt")
        results = model(SAMPLE_IMAGE)

        # plot() renders the field; it is defined for normal and edge results.
        results[0].plot().save("normals.png")
  val:
    - label: Validate and read the metric keys
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMoGe2s-normal.pt")
        metrics = model.val(data="my-dataset.yaml", imgsz=518)

        print(metrics["metrics/mean_angular_error"])     # degrees
        print(metrics["metrics/median_angular_error"])   # degrees
        print(metrics["metrics/within_11_25"])           # percent of pixels
        print(metrics["metrics/within_22_5"], metrics["metrics/within_30"])
  export:
    - label: Export
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMoGe2s-normal.pt")
        model.export(format="onnx", imgsz=518)
    - label: Run the exported file
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # The factory routes on the file suffix, so an exported artifact loads
        # like any checkpoint and returns the same Results object.
        model = LibreYOLO("LibreMoGe2s-normal.onnx")
        results = model(SAMPLE_IMAGE)

        print(results[0].normal_map.data.shape)
---

## Definition

The `normal` task predicts a three-component unit vector per pixel from a single
RGB image: the direction the surface at that pixel faces. Unlike depth, the
output has no free scale, so two predictions are directly comparable without
alignment.

A prediction fills `results[0].normal_map`, a `NormalMap` payload holding an
`(H, W, 3)` float32 array on the original image canvas, also reachable as
`results[0].normals`. Vectors use LibreYOLO's OpenCV camera frame, with `+x`
right, `+y` down and `+z` into the scene, and they face the camera, so a
fronto-parallel surface reads `(0, 0, -1)`. `.assert_normalized()` checks that
every pixel is finite and unit length within a tolerance. `results[0].boxes`
stays empty, so `conf`, `iou` and `max_det` have no effect, and
`Results.plot()` covers this task.

## Models

Two families serve `normal`.

[MoGe-2](/docs/models/moge-2) is the dedicated one: a single-forward monocular
geometry model in three encoder sizes. LibreYOLO does not copy these checkpoints
into its own organization; loading one downloads the matching size from the
official repositories at a pinned revision and verifies it against a recorded
SHA-256.

[LibreMODUS](/docs/models/libremodus) produces normals as one target of an
any-to-any model, and can take a depth map rather than an RGB image as its
input. It needs the `modus` extra and your own authenticated Hugging Face
account, and it offers neither `val()` nor `export()`, so it does not take part
in the validation and export sections below.

## Predict

MoGe-2 weights download on first use and are cached locally.

<code-tabs name="predict" />

`imgsz` must be divisible by the ViT encoder's patch size, which LibreYOLO
checks before the run starts. Predicting a list of images runs one forward pass
per image; this task has no stacked-batch fast path. See
[prediction](/docs/predict) for sources, streaming and result handling.

## Dataset format

Normal validation pairs each image with a same-stem three-channel 16-bit PNG of
the same resolution, plus an optional validity mask.

```text
dataset/
  data.yaml
  images/
    val/room.jpg
  normals/
    val/room.png
  masks/
    val/room.png
```

```yaml
path: dataset
train: images/train
val: images/val
normals_dir: normals
masks_dir: masks
nc: 1
names: {0: normal}
```

The target PNG is exactly three-channel `uint16` with channels stored as RGB.
Decoding is `n = png / 65535 * 2 - 1` followed by renormalizing each vector, and
the decoded vectors use the same OpenCV camera frame as the predictions. A mask
pixel counts as valid when nonzero; without a mask file, every finite nonzero
decoded vector is valid. Invalid and padded target pixels are held internally as
`(0, 0, 0)` and never contribute to a metric. See
[dataset formats](/docs/reference/dataset-formats) for the full contract.

## Train

Neither normal family has a training implementation: `train()` raises
`NotImplementedError` on both. MoGe-2's page points at its pinned official
checkpoints for predict, validate and export.

## Validate

`val()` measures the angle between each predicted vector and its ground-truth
vector, over the pixels the dataset marks valid.

<code-tabs name="val" />

`metrics/mean_angular_error` and `metrics/median_angular_error` are that angle in
degrees, and lower is better. `metrics/within_11_25`, `metrics/within_22_5` and
`metrics/within_30` are the percentage of valid pixels whose angular error falls
within 11.25, 22.5 and 30 degrees, so higher is better. Note the unit: those
three are percentages, not fractions. `fitness` is `metrics/within_11_25`
divided by 100, which puts best-checkpoint selection on the same `[0, 1]` scale
as every other task.

## Export

An exported normal model loads back through `LibreYOLO()` on its file suffix, so
a `.onnx` file behaves like a checkpoint and returns the same `Results`.

<code-tabs name="export" />

Normal export uses a fixed-resolution, batch-1 runtime contract: `dynamic` and a
`batch` other than 1 are rejected, and `imgsz` must be divisible by the encoder's
patch size. Per-format coverage is on the [MoGe-2 page](/docs/models/moge-2) and
in the [full export matrix](/docs/reference/export-matrix).
[Export](/docs/export) lists the arguments every format accepts.
