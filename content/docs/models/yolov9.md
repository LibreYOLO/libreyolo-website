---
title: YOLOv9
families: [yolo9]
seo_title: "YOLOv9: predict, train and export under MIT"
description: "Run YOLOv9 in LibreYOLO, including the NMS-free end-to-end head and the stride-4 small-object head. Install, predict, train, validate and export."
lead: "A single-stage convolutional detector: one pass scores a dense grid of boxes and NMS drops the duplicates. LibreYOLO carries three variants of it, one of which has no NMS step."
keywords: [YOLOv9, YOLO9, object detection, NMS-free detection, end-to-end detection, small object detection, programmable gradient information, GELAN]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Without NMS
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Same call, different checkpoint. The end-to-end head returns its own
        # top-scoring predictions, so no NMS runs and iou is ignored.
        model = LibreYOLO("LibreYOLO9E2Es.pt")
        result = model(SAMPLE_IMAGE, conf=0.25, max_det=300)

        print(len(result.boxes))
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(data="my-dataset.yaml", epochs=100, imgsz=640, batch=16)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          epochs=100 imgsz=640 batch=16
    - label: Small objects
      language: python
      code: |
        from libreyolo import LibreYOLO9P2

        # The stride-4 variant has no COCO checkpoint of its own, so name a
        # base detection one: its backbone and neck load unchanged and the
        # stride-4 head tower starts from random initialization.
        model = LibreYOLO9P2(None, size="s")
        model.train(data="my-dataset.yaml", epochs=100, pretrained="LibreYOLO9s.pt")
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLO9s.pt data=my-dataset.yaml
    - label: Against COCO
      language: bash
      code: |
        # The bundled COCO yaml carries an embedded download script, so it
        # needs explicit permission unless the dataset is already local.
        libreyolo val model=LibreYOLO9c.pt data=coco.yaml imgsz=640 \
          allow_download_scripts=True
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.export(format="onnx", imgsz=640)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreYOLO9s.pt format=onnx imgsz=640
    - label: With NMS in the graph
      language: bash
      code: |
        libreyolo export model=LibreYOLO9s.pt format=onnx nms=True \
          conf=0.25 iou=0.45 max_det=300
    - label: Use the exported file
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # The factory routes on the file suffix, so an exported artifact loads
        # like any checkpoint and returns the same Results object.
        model = LibreYOLO("LibreYOLO9s.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
---

## Install

YOLOv9 needs no extra beyond the base package.

```bash
pip install libreyolo
```

## Predict

Weights download from Hugging Face on first use and are cached locally.

<code-tabs name="predict" />

The returned `Results` object is the one every family returns, so swapping in a
different detector is a one line change. On the base and stride-4 models, `conf`
sets the confidence threshold and `iou` the NMS threshold. The end-to-end model
runs no NMS and ignores `iou`, so `conf` and `max_det` are what shape its
output. See [prediction](/docs/predict) for sources, streaming and result
handling.

## Variants

Three variants share a backbone. All three detect only, and they take the same
arguments.

The base model predicts on three feature scales and clears duplicate boxes with
NMS.

The end-to-end model keeps that head and adds a one-to-one matching branch
beside it. Inference reads the one-to-one branch alone and takes its
top-scoring predictions, so no NMS runs. Choose it when the runtime you deploy
to has no NMS operator.

The stride-4 model surfaces one level further up the backbone, extends the neck
down to it and predicts on four scales instead of three. The extra scale is for
objects that cover few pixels; the one published checkpoint for it is trained
on aerial imagery. Base detection checkpoints transfer into it: the backbone
and neck load unchanged, the three pretrained head towers shift up one slot,
and the stride-4 tower starts from random initialization.

<benchmark-table task="detect" />

<va-embed />

## Train

<code-tabs name="train" />

`pretrained` decides what the run starts from. Pass `True` to load the
published checkpoint for the same model and size, or a name or path for
anything else. Tensors whose shape does not match are skipped rather than
refused, and the run logs how many loaded, so a checkpoint trained on a
different class count is still a usable starting point.

The stride-4 model has no published COCO checkpoint of its own, so `True`
resolves there to a file that does not exist and the download fails. Name a
base detection checkpoint instead.

See [training](/docs/train) for datasets, augmentation, multi-GPU and loggers.

## Validate

`val()` returns a dictionary of `metrics/` keys covering precision, recall,
mAP 50 and mAP 50-95, measured against any dataset in the format you trained on.

<code-tabs name="val" />

## Export

<export-matrix />

A tick holds for all three variants: where they differ, the matrix carries the
weakest of the three.

An exported artifact loads back through `LibreYOLO()` on its file suffix, so a
`.onnx` or `.engine` file behaves like a checkpoint and returns the same
`Results`. Running the graph in a bare runtime, with no LibreYOLO installed, is
also supported, but then preprocessing and postprocessing are yours to write.

For the base detection model, the postprocessing half of that can move into the
graph. `nms=True` on an ONNX export puts suppression inside the model, and the
first output becomes a fixed `(1, max_det, 6)` tensor whose rows are
`x1, y1, x2, y2, score, class`, zero-padded past the detection count. That graph
is batch 1 and carries no dynamic axes. The end-to-end and stride-4 models do
not accept the flag.

Each format installs a different extra and takes a few arguments of its own.
Both are on that format's page.

<code-tabs name="export" />

## Checkpoints

Every published weight file for this family.

<checkpoint-table />

## Licensing

<provenance-box>

One checkpoint here is not MIT. The stride-4 model trained on VisDrone2019-DET
inherits that dataset's CC BY-NC-SA 3.0 terms: non-commercial use only,
share-alike on anything derived from it, and outside the permissive license the
rest of this family ships under. It predicts the VisDrone aerial classes rather
than the COCO ones. The library prints all of this before it downloads the file.

</provenance-box>

## Citation

<citation-block />
