---
title: Depth estimation
seo_title: Monocular depth estimation in LibreYOLO
description: >-
  Predict a dense relative depth map from one image in LibreYOLO. Compare the
  depth families, read the depth metrics, and export a depth model.
lead: >-
  Depth estimation predicts how far each pixel is from the camera using a single
  image. LibreYOLO exposes it as the depth task, which returns a dense relative
  inverse-depth map on the original image canvas.
keywords:
  - monocular depth estimation python
  - depth map from single image
  - relative depth model
  - depth anything libreyolo
  - dense depth prediction
last_verified: 1.5.0
snippets:
  predict:
    - label: Predict a depth map
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")
        result = model(SAMPLE_IMAGE, save=True)

        depth = result.depth_map
        print(depth.data.shape)              # (H, W) on the original canvas
        print(depth.min, depth.max, depth.mean)
    - label: Work with the values
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")
        result = model(SAMPLE_IMAGE)

        depth = result.depth_map
        raw = depth.data          # higher is closer; no metric unit, no scale
        gray = depth.normalized() # rescaled to [0, 1] for visualization
        print(raw.shape, float(gray.max()))
    - label: A compact alternative
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Same task contract, a much smaller network built for edge runtimes.
        model = LibreYOLO("LibreZipDepthb-depth.pt")
        result = model(SAMPLE_IMAGE)

        print(result.depth_map.data.shape)
  val:
    - label: Validate and read the metric keys
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/abs_rel"])
        print(metrics["metrics/rmse"])
        print(metrics["metrics/delta1"])   # fitness
        print(metrics["metrics/delta2"], metrics["metrics/delta3"])
  export:
    - label: Export
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")
        model.export(format="onnx")
    - label: Run the exported file
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # The factory routes on the file suffix, so an exported artifact loads
        # like any checkpoint and returns the same Results object.
        model = LibreYOLO("LibreDepthAnythingV2s-depth.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.depth_map.data.shape)
source_hash: e0612c59f9c999b4
---

## Definition

The `depth` task predicts one value per pixel from a single RGB image. LibreYOLO
defines that value as relative inverse depth: higher means closer to the camera,
and the numbers carry no metric unit and no scale that holds across two images.
Comparing depth between two pixels of the same prediction is meaningful;
comparing a value to a value from another image is not.

A prediction fills `result.depth_map`, a `DepthMap` payload holding an
`(H, W)` array on the original image canvas. `.min`, `.max` and `.mean` read the
finite values, and `.normalized()` rescales the map to `[0, 1]` for display.
`result.boxes` stays empty, so `conf`, `iou` and `max_det` have no effect,
and `save=True` writes a colormapped image of the map rather than an annotated
photo.

## Models

Six families serve `depth`.

[Depth Anything V2](/docs/models/depth-anything-v2) pairs a DINOv2 encoder with
a DPT decoder and is the general-purpose default here. Licensing decides the
size as much as accuracy does: the Small checkpoint is Apache-2.0 while Base and
Large are non-commercial, so check the checkpoint table on its page before
picking one.

[Depth Anything 3](/docs/models/depth-anything-3) ports the DA3MONO-LARGE
checkpoint, a plain transformer with no architectural specialization for depth.

[ZipDepth](/docs/models/zipdepth) is the compact tier: a reparameterizable CNN
distilled from Depth Anything V2 Large, with a second checkpoint whose decoder
avoids gather and unfold operations for NPU compilers that lack them.

[MiDaS](/docs/models/midas) is the line of work that established the zero-shot
relative-depth protocol the other families are measured with. It is the one
depth family LibreYOLO does not republish: requesting a checkpoint downloads the
official asset from its authors' GitHub release and checks a pinned SHA-256.

[LibreMODUS](/docs/models/libremodus) reaches depth as one target of an
any-to-any model rather than as a dedicated head. It needs the `modus` extra and
your own authenticated Hugging Face account, and it offers neither `val()` nor
`export()`.

[SenseNova-Vision](/docs/models/sensenova-vision) generates the depth map as an
image through a diffusion decode, from the same 7B checkpoint that serves its
six other tasks. It needs the `sensenova` extra, and its weights are restricted
to non-commercial use; the license is on its page.

## Predict

Weights download from Hugging Face on first use and are cached locally, except
for the two families noted above.

<code-tabs name="predict" />

Input resolution is constrained per family. Depth Anything V2 and Depth Anything
3 build on a DINOv2 patch grid, so `imgsz` must divide evenly by 14, which
LibreYOLO checks before running. `Results.plot()` does not cover this task; it
is defined for surface normals and edges only. See [prediction](/docs/predict)
for sources, streaming and result handling.

## Dataset format

Depth validation pairs each image with a dense single-channel depth map that has
the same resolution, found by substituting the depth directory into the image
path.

```text
dataset/
  data.yaml
  images/
    val/room.jpg
  depths/
    val/room.png
```

```yaml
path: dataset
val: images/val
depths_dir: depths
nc: 1
names: {0: depth}
```

Maps are single-channel PNG or TIF, or `.npy`. Values are plain depth in a unit
the dataset keeps consistent, and `0`, negative, NaN and infinite pixels mark
invalid samples that are excluded from the metrics. Integer maps are divided by
`depth_scale`, which defaults to `256.0`, the 16-bit PNG convention; float
`.npy` maps are used as they are. `depth_stem_suffix` and `depth_mask_suffix`
cover datasets that name their depth files or validity masks differently. See
[dataset formats](/docs/reference/dataset-formats) for the full contract.

## Train

No depth family in LibreYOLO has a training implementation: `train()` raises
`NotImplementedError` on all six. Each model page names the conversion script
that turns a checkpoint trained upstream into one LibreYOLO can load.

## Validate

`val()` runs the shared depth validator. Relative depth has no absolute scale,
so each prediction is first fitted to the inverse of its ground truth with a
per-image least-squares scale and shift, then inverted back to depth. Every
metric below is computed per image on that aligned map and averaged over the
dataset, counting only pixels the dataset marks valid.

<code-tabs name="val" />

`metrics/abs_rel` is the mean absolute relative error, the residual divided by
the ground-truth depth, and lower is better. `metrics/rmse` is the root mean
squared error in the dataset's own depth unit, also lower is better.
`metrics/delta1`, `metrics/delta2` and `metrics/delta3` are the threshold
accuracies: the fraction of valid pixels whose ratio to ground truth, taken in
whichever direction is larger, falls under 1.25, 1.25 squared and 1.25 cubed, so
higher is better. `metrics/delta1` is also `fitness`, the number
best-checkpoint selection reads.

## Export

An exported depth model loads back through `LibreYOLO()` on its file suffix, so
a `.onnx` or `.engine` file behaves like a checkpoint and returns the same
`Results`, with `depth_map` in place of boxes.

<code-tabs name="export" />

Coverage differs per family, and Depth Anything 3 rejects any format outside its
validated set rather than attempting an unvalidated conversion. Check the model
page and the [full export matrix](/docs/reference/export-matrix) before
committing to a target. LibreMODUS and SenseNova-Vision do not export at all.
[Export](/docs/export) lists the arguments every format accepts.


