---
title: Edge detection
seo_title: "Edge detection in LibreYOLO"
description: "Predict a dense edge-probability map from one image in LibreYOLO. Convert a checkpoint, threshold the map, validate with ODS and OIS, and export."
lead: "Edge detection predicts how likely each pixel is to lie on an object boundary. LibreYOLO exposes it as the edge task, which returns a dense probability map on the original image canvas rather than a set of line segments."
keywords: [edge detection python, boundary detection deep learning, edge probability map, ODS OIS F-measure, dense edge prediction]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Predict an edge map
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # No edge checkpoint ships with LibreYOLO; convert one first (below).
        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")
        result = model(SAMPLE_IMAGE, save=True)

        edges = result.edges
        print(edges.array.shape)          # (H, W) float32 in [0, 1]
        print(edges.binary(0.5).sum())    # edge-pixel count at 0.5
    - label: Choose your own threshold
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")
        result = model(SAMPLE_IMAGE)

        # The continuous map is kept so the threshold stays your decision.
        for t in (0.3, 0.5, 0.7):
            print(t, int(result.edges.binary(t).sum()))
    - label: Save the visualization
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")
        result = model(SAMPLE_IMAGE)

        # plot() renders the map; it is defined for edge and normal results.
        result.plot().save("edges.png")
  val:
    - label: Validate and read the metric keys
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")
        metrics = model.val(data="my-dataset.yaml", imgsz=352)

        print(metrics["metrics/ODS"])              # fitness
        print(metrics["metrics/OIS"])
        print(metrics["metrics/best_threshold"])
    - label: Change the sweep and the match tolerance
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")
        metrics = model.val(
            data="my-dataset.yaml",
            imgsz=352,
            edge_thresholds=(0.1, 0.2, 0.3, 0.4, 0.5),
            edge_max_dist=0.0075,
        )

        print(metrics["metrics/ODS"], metrics["metrics/best_threshold"])
  export:
    - label: Export
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")
        model.export(format="onnx", imgsz=352)
    - label: Run the exported file
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # The factory routes on the file suffix, so an exported artifact loads
        # like any checkpoint and returns the same Results object.
        model = LibreYOLO("weights/LibreDexiNedb-edge.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.edges.array.shape)
---

## Definition

The `edge` task predicts one probability per pixel from a single RGB image:
`0` means non-edge and `1` means edge. The map stays continuous, so choosing the
threshold that turns it into a binary boundary image is left to the caller, and
the right threshold depends on the dataset and the downstream use.

A prediction fills `result.edges`, an `EdgeMap` payload holding an `(H, W)`
float32 array in `[0, 1]` on the original image canvas. `.array` returns that
map as NumPy and `.binary(threshold)` returns a boolean mask. `result.boxes`
stays empty, so `conf`, `iou` and `max_det` have no effect. `Results.plot()`
covers this task and renders the map directly.

## Models

Three families serve `edge`.

[DexiNed](/docs/models/dexined), the Dense Extreme Inception Network, fuses
several side outputs into one probability map and runs at a native 352 px.

[TEED](/docs/models/teed), the Tiny and Efficient Edge Detector, is a small
network at the same native 352 px, with a downsample stride of 4 against
DexiNed's 16, so it accepts more values of `imgsz`.

[LibreMODUS](/docs/models/libremodus) produces Canny-style edges as one target
of an any-to-any model. It needs the `modus` extra and your own authenticated
Hugging Face account, and it offers neither `val()` nor `export()`, so it does
not take part in the validation and export sections below.

## Predict

LibreYOLO publishes no edge checkpoint. The officially released DexiNed and TEED
weights are trained on BIPED, whose published dataset terms restrict use to
non-commercial purposes, so LibreYOLO does not mirror them. Convert a checkpoint
you are licensed to use, then load the converted file by path:

```bash
python weights/convert_dexined_weights.py upstream.pth weights/LibreDexiNedb-edge.pt --verify
```

<code-tabs name="predict" />

The filename has to carry the `-edge` task suffix for the loader to recognize
it. `imgsz` must be divisible by the network's downsample stride, and LibreYOLO
raises a clear error naming the divisor when it is not. See
[prediction](/docs/predict) for sources, streaming and result handling.

## Dataset format

Edge validation pairs each RGB image with a same-stem single-channel map of the
same resolution, plus an optional validity mask.

```text
dataset/
  data.yaml
  images/
    val/scene.jpg
  edges/
    val/scene.png
  masks/
    val/scene.png
```

```yaml
path: dataset
train: images/train
val: images/val
edges_dir: edges
masks_dir: masks
nc: 1
names: {0: edge}
```

The target is a single-channel PNG or TIF, not an RGB visualization. Integer
maps are divided by the maximum of their dtype; float maps must already be
finite and in `[0, 1]`. Mask pixels count as valid when nonzero, and padded
pixels never contribute to a metric. `edge_invert: true` covers sources that
store black edges on white. See
[dataset formats](/docs/reference/dataset-formats) for the full contract.

## Train

No edge family in LibreYOLO has a training implementation: `train()` raises
`NotImplementedError` on all three. Each model page names the conversion script
that turns a checkpoint trained elsewhere into one LibreYOLO can load.

## Validate

`val()` reports the BSDS-style F-measures. Continuous predictions are thinned
first with four-direction gradient non-maximum suppression, then predicted and
ground-truth edge pixels are matched one-to-one within a distance tolerance.

<code-tabs name="val" />

`metrics/ODS` is the optimal-dataset-scale F-measure: match counts are pooled
across the dataset at each threshold, and the best of those pooled F-measures is
reported. It is also `fitness`, the number best-checkpoint selection reads.
`metrics/OIS` is the optimal-image-scale F-measure, the mean over images of each
image's own best F-measure, so it lets every image pick its own threshold.
`metrics/best_threshold` is the single threshold that produced ODS, which is the
one to reuse in `edges.binary()` at inference.

Two arguments shape the sweep. `edge_thresholds` is the set of thresholds tried,
defaulting to 0.01 through 0.99 in hundredths. `edge_max_dist` is the match
tolerance as a fraction of the image diagonal, defaulting to `0.0075`; a pair
further apart than that is not a match.

## Export

An exported edge model loads back through `LibreYOLO()` on its file suffix, so a
`.onnx` file behaves like a checkpoint and returns the same `Results`.

<code-tabs name="export" />

Edge export uses a fixed-resolution, batch-1 runtime contract: `dynamic` and a
`batch` other than 1 are rejected, and the exported graph emits a single fused
probability map. Per-format coverage is on the [DexiNed](/docs/models/dexined)
and [TEED](/docs/models/teed) pages and in the
[full export matrix](/docs/reference/export-matrix). [Export](/docs/export)
lists the arguments every format accepts.
