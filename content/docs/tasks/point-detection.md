---
title: Point detection
seo_title: "Point detection and counting in LibreYOLO"
description: "Locate objects as single points instead of boxes in LibreYOLO. Predict centroids, count objects, train FOMO, and read the point metrics."
lead: "Point detection returns one x, y location per object instead of a bounding box. LibreYOLO exposes it as the point task, and a prediction carries one row of x, y, class and confidence per object."
keywords: [point detection python, object counting python, centroid detection, FOMO point localization, counting objects in images, point localization]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Predict points and count them
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # LibreFOMO weights are not auto-downloaded. Fetch a checkpoint from
        # https://huggingface.co/LibreYOLO first and load it by local path.
        model = LibreYOLO("./LibreFOMOs-point.pt")
        result = model(SAMPLE_IMAGE, save=True)

        points = result.points
        print(len(points))     # object count
        print(points.xy)       # (N, 2) centers in original-image pixels
        print(points.cls, points.conf)
    - label: Normalized coordinates and per-class counts
      language: python
      code: |
        from collections import Counter

        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("./LibreFOMOs-point.pt")
        result = model(SAMPLE_IMAGE)

        points = result.points.numpy()
        print(points.xyn)                          # same centers in [0, 1]
        print(Counter(points.cls.astype(int).tolist()))
  train:
    - label: Train FOMO on a YOLO dataset
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("./LibreFOMOs-point.pt")
        model.train(data="my-dataset.yaml", epochs=40, batch=32, lr0=3e-4)
    - label: Predict with the trained checkpoint
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("./LibreFOMOs-point.pt")
        results = model.train(data="my-dataset.yaml", epochs=40)

        # train() reloads the best checkpoint into the same object, so the
        # model predicts with the trained weights when the call returns.
        print(results["best_checkpoint"])
        print(model(SAMPLE_IMAGE).points.xy)
  val:
    - label: Validate and read the metric keys
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("./LibreFOMOs-point.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/precision"], metrics["metrics/recall"])
        print(metrics["metrics/f1"])
        print(metrics["metrics/mAP@[0.01:0.10]"])   # fitness
        print(metrics["metrics/MLE"])               # mean localization error
        print(metrics["metrics/MAE"], metrics["metrics/RMSE"])   # count error
    - label: Change the distance thresholds
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("./LibreFOMOs-point.pt")

        # The sweep bounds are part of the key text, so a custom sweep
        # renames the mAP keys it produces.
        metrics = model.val(data="my-dataset.yaml", dist_thresholds=[0.02, 0.05])

        print(metrics["metrics/mAP@0.02"])
        print(metrics["metrics/mAP@[0.02:0.05]"])
  export:
    - label: Export
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("./LibreFOMOs-point.pt")
        model.export(format="onnx")
    - label: Run the exported file
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # The factory routes on the file suffix, so an exported artifact loads
        # like any checkpoint and returns the same Results object.
        model = LibreYOLO("./LibreFOMOs-point.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.points.xy)
---

## Definition

The `point` task locates each object with a single x, y coordinate and a class,
with no width, height or mask. Because a prediction is a flat list of objects,
the row count is the object count, which is what makes this the counting task.

A prediction fills `result.points`, a `Points` payload wrapping an `(N, 4)`
array of `x, y, class, confidence` rows in original-image pixels. `.xy` returns
the coordinates, `.xyn` the same coordinates divided by the image size, `.cls`
the class indices and `.conf` the scores; `len()` returns the number of points.
`result.boxes` stays empty, so `iou` and `max_det` have nothing to act on.

## Models

Three families serve `point`, and they are not interchangeable.

[FOMO](/docs/models/fomo) is the fixed-vocabulary option: a grid classifier that
labels each cell of a low-resolution grid as background or an object center. It
is the only point family LibreYOLO can train, and the only one that exports.

[LocateAnything](/docs/models/locate-anything) takes text instead of a class
index, so the vocabulary is whatever phrase you write. It needs the `vlm` extra,
is constructed as `LibreLocateAnything` rather than through the `LibreYOLO()`
factory, and its weights are restricted to non-commercial use. The exact terms,
and the two further licenses the checkpoint composes, are on its page.

[SenseNova-Vision](/docs/models/sensenova-vision) reaches `point` through the
same prompted-generation checkpoint it uses for six other tasks, loaded with
`LibreVLM("sensenova-vision", task="point")`. It needs the `sensenova` extra,
and every prediction is a generation pass over a 7B model, so expect noticeably
higher per-image latency than a purpose-built detector. Its weights are
non-commercial; the license is on its page.

## Predict

LibreFOMO weights are the one exception to automatic download on this site.
`LibreYOLO("LibreFOMOs-point.pt")` looks for that file on disk and raises a
`ValueError` naming it rather than fetching it. Download a checkpoint from the
[LibreYOLO organization](https://huggingface.co/LibreYOLO) on Hugging Face
first and load it by local path, or train your own.

<code-tabs name="predict" />

The filename has to carry the `-point` task suffix for the loader to recognize
it. `predict(..., nms_radius=1)` controls how many grid cells apart two FOMO
detections must be to both survive. See [prediction](/docs/predict) for sources,
streaming and result handling.

## Dataset format

`point` has no label format of its own. The point families read the standard
YOLO detection layout and derive one center from each box row, so `cx cy` is the
point and `w h` only decide whether the row is valid.

```text
dataset/
  data.yaml
  images/
    train/scene.jpg
    val/scene.jpg
  labels/
    train/scene.txt
    val/scene.txt
```

Each label file holds one row per object, with normalized coordinates:

```text
<class_id> <cx> <cy> <w> <h>
```

```yaml
path: dataset
train: images/train
val: images/val
nc: 1
names: {0: seedling}
```

A missing or empty label file means no objects. See
[dataset formats](/docs/reference/dataset-formats) for the full contract.

## Train

FOMO is the only point family with a training implementation. `train()` on
LocateAnything and on SenseNova-Vision raises `NotImplementedError`; fine-tune
those upstream and load the result.

<code-tabs name="train" />

`imgsz` is not a free choice for FOMO: it defaults to the loaded checkpoint's
native resolution, and passing a different value raises `ValueError` naming the
size it expects. See [training](/docs/train) for datasets, loggers and
multi-GPU, and the [FOMO page](/docs/models/fomo) for this family's defaults.

## Validate

`val()` matches predicted points to ground-truth points one-to-one with the
Hungarian algorithm, over a sweep of distance thresholds. A threshold is a
Euclidean distance in normalized image coordinates, and the default sweep is ten
values from 0.01 to 0.10.

<code-tabs name="val" />

`metrics/precision`, `metrics/recall` and `metrics/f1` are macro-averaged over
classes at the strictest threshold in the sweep, 0.01 by default.
`metrics/mAP@0.01` is average precision at that same threshold, and
`metrics/mAP@[0.01:0.10]` is the mean over the whole sweep. That sweep value is
also `fitness`, the number best-checkpoint selection reads. Both mAP keys are
built from the thresholds in use, so passing `dist_thresholds=` renames them.

`metrics/MLE` is the mean distance between matched pairs at the strictest
threshold, in the same normalized units. `metrics/MAE` and `metrics/RMSE` are
counting metrics rather than localization ones: they measure the per-image
difference between the number of predicted and ground-truth points.

FOMO adds a second, grid-level group on top of these. It sweeps confidence and
`nms_radius` and publishes the best-F1 combination as `metrics/grid_F1`,
`metrics/grid_precision`, `metrics/grid_recall`, `metrics/grid_mean_distance`,
`metrics/grid_TP`, `metrics/grid_FP` and `metrics/grid_FN`, with the settings
that produced it under `decode/threshold` and `decode/nms_radius`.

## Export

FOMO exports through the shared export path, and an exported artifact loads back
through `LibreYOLO()` on its file suffix, so a `.onnx` or `.engine` file behaves
like a checkpoint and returns the same `Results`.

<code-tabs name="export" />

Per-format coverage is on the [FOMO page](/docs/models/fomo) and in the
[full export matrix](/docs/reference/export-matrix). LocateAnything and
SenseNova-Vision do not export: `export()` raises on both, because a generative
model has no traceable detection graph.
