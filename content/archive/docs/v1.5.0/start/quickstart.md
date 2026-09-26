---
title: Quickstart
seo_title: "LibreYOLO quickstart"
description: "Run a detector on an image, fine-tune it on a small dataset and export it to TorchScript or ONNX, all on CPU, in about ten lines of Python."
lead: "The shortest path through LibreYOLO: predict on one image, train on a small dataset, then export the result. Every command here runs on CPU."
keywords: [libreyolo quickstart, libreyolo tutorial, libreyolo predict, libreyolo train, libreyolo export, yolo python example]
last_verified: "1.5.0"
meta:
  - label: Install
    value: pip install libreyolo
    mono: true
  - label: Checkpoint
    value: LibreYOLO9t.pt
    mono: true
  - label: Hardware
    value: CPU is enough for everything on this page
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Downloads the checkpoint on first use, then caches it in weights/.
        model = LibreYOLO("LibreYOLO9t.pt")

        # A single image returns one Results object.
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(result.names[int(box.cls)], float(box.conf), box.xyxy.tolist())
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=yolo9-t save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Video and streams
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # stream=True yields one Results per frame instead of building a list.
        # Replace the path with a webcam index, an RTSP URL or a folder.
        for result in model.predict("clip.mp4", stream=True, save=True):
            print(len(result.boxes))
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # coco8 is an 8-image dataset bundled with the library. It downloads
        # from a URL on first use, so no script has to be executed.
        results = model.train(
            data="coco8.yaml",
            epochs=1,
            imgsz=640,
            batch=4,
            device="cpu",
        )

        print(results["save_dir"])
        print(results["best_checkpoint"])
    - label: CLI
      language: bash
      code: |
        libreyolo train model=yolo9-t data=coco8.yaml \
          epochs=1 imgsz=640 batch=4 device=cpu
    - label: Validate
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # val() returns a plain dict, not an object.
        metrics = model.val(data="coco8.yaml", device="cpu")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
  export:
    - label: TorchScript
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9t.pt")

        # export() returns the path it wrote.
        path = model.export(format="torchscript")
        print(path)

        # The factory routes on file suffix, so the artifact loads back like a
        # checkpoint and returns the same Results object.
        exported = LibreYOLO(path)
        result = exported(SAMPLE_IMAGE)
        print(len(result.boxes))
    - label: ONNX
      language: bash
      code: |
        pip install "libreyolo[onnx]"
        libreyolo export model=yolo9-t format=onnx imgsz=640
---

## Install

```bash
pip install libreyolo
```

That is everything the predict and train sections below need. Export to ONNX
adds one extra; see [install](/docs/install) for the full list.

## Predict

<code-tabs name="predict" />

`LibreYOLO()` is a factory. It reads the file, works out which family the weights
belong to, and returns that family's model, so swapping in a different detector
is a one-line change. Passing `LibreYOLO9t.pt` with no directory looks for
`weights/LibreYOLO9t.pt` relative to the working directory and downloads it
there when it is missing. See [checkpoints and weights](/docs/weights) for the
download rules and how to work offline.

`save=True` writes an annotated copy under `runs/detect/`, into a `predict`
directory that increments per run. The returned `Results` carries `boxes`, and
`names` maps a class index to its label. A single image path returns one
`Results`; a directory, a list of images or `stream=True` returns a list or a
generator of them.

## Train

<code-tabs name="train" />

`data` is a dataset YAML. `coco8.yaml` ships with the library, which is why the
snippet runs as pasted; a name that is not bundled is read as a path. Datasets
resolve under `~/datasets`, or under `LIBREYOLO_DATASETS_DIR` when that variable
is set.

A run writes to `project/name`, defaulting to a directory below `runs/train`,
with `weights/best.pt` and `weights/last.pt` inside it. `train()` returns a
dictionary that includes `save_dir`, `best_checkpoint`, `last_checkpoint`,
per-epoch losses and per-epoch validation metrics. The trained checkpoint loads
through `LibreYOLO()` exactly like the pretrained one.

Not every family is trainable. Where a family ships inference only, `train()`
raises `NotImplementedError` and says so. [Core concepts](/docs/concepts)
explains which support tier means what.

## Export

<code-tabs name="export" />

TorchScript needs nothing beyond the base install. The other targets each have
their own extra, and coverage is per family and per task rather than uniform:
see [export and deploy](/docs/export).

Arguments accepted by every format include `imgsz` (an int, or a height and
width pair), `batch` (default 1), `half`, `int8` with a `data` YAML for
calibration, `dynamic` (default True), `simplify` (default True), `opset`,
`device` and `output_path`. When `output_path` is omitted the file is written
under `weights/` with a name derived from the checkpoint.

## Where to go next

- [Core concepts](/docs/concepts) for tasks, families, sizes and checkpoint names.
- [Checkpoints and weights](/docs/weights) for auto-download, offline use and loading safety.
- [Import existing weights](/docs/migrate) if you already have a checkpoint from an upstream project.
- [All models](/docs/models) for the family that fits your problem.
- [Train](/docs/train), [Predict](/docs/predict) and [Export](/docs/export) for the full workflows.
