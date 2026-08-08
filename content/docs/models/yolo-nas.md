---
title: YOLO-NAS
families: [yolonas]
seo_title: "YOLO-NAS: predict, train and export in LibreYOLO"
description: "Use YOLO-NAS in LibreYOLO for detection and pose. Deci.AI's weights are proprietary and non-commercial, and LibreYOLO publishes none of them."
lead: "A convolutional detector whose backbone and neck came out of Deci.AI's architecture search, built from quantization-aware RepVGG blocks. Its weights are Deci.AI's, licensed for non-commercial use only, and LibreYOLO publishes none of them."
keywords: [YOLO-NAS, YOLONAS, Deci AI, SuperGradients, object detection, pose estimation, quantization aware detector, AutoNAC]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # A name not already on disk is fetched from Deci's CDN. The download
        # prints Deci's license terms first; taking the file accepts them.
        model = LibreYOLO("LibreYOLONASs.pt")
        results = model(SAMPLE_IMAGE, save=True)

        for box in results[0].boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLONASs.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Pose
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # The -pose suffix picks the pose head and its own set of weights.
        model = LibreYOLO("LibreYOLONASs-pose.pt")
        results = model(SAMPLE_IMAGE)

        print(results[0].keypoints.xy)
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLONASs.pt")
        model.train(data="my-dataset.yaml", epochs=100, imgsz=640, batch=16)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLONASs.pt data=my-dataset.yaml \
          epochs=100 imgsz=640 batch=16
    - label: From scratch
      language: python
      code: |
        from libreyolo import LibreYOLONAS

        # No Deci checkpoint is touched: the model starts from random weights,
        # so what comes out of the run derives only from your data.
        model = LibreYOLONAS(None, size="s")
        model.train(data="my-dataset.yaml", imgsz=640, batch=16)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLONASs.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLONASs.pt data=my-dataset.yaml
    - label: Against COCO
      language: bash
      code: |
        # The bundled COCO yaml carries an embedded download script, so it
        # needs explicit permission unless the dataset is already local.
        libreyolo val model=LibreYOLONASl.pt data=coco.yaml imgsz=640 \
          allow_download_scripts=True
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLONASs.pt")
        model.export(format="onnx", imgsz=640)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreYOLONASs.pt format=onnx imgsz=640
    - label: Use the exported file
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # The factory routes on the file suffix, so an exported artifact loads
        # like any checkpoint and returns the same Results object.
        model = LibreYOLO("LibreYOLONASs.onnx")
        results = model(SAMPLE_IMAGE)

        print(results[0].boxes.xyxy)
---

## Install

YOLO-NAS needs no extra beyond the base package.

```bash
pip install libreyolo
```

## Predict

A checkpoint name that is not already on disk is fetched from Deci's public
CDN, not from the LibreYOLO org, which hosts none of these weights. Before the
transfer starts the library prints Deci's license terms once per process, and
before the downloaded file is opened its SHA-256 is checked against a pinned
value. What those terms allow is in [licensing](#licensing).

<code-tabs name="predict" />

The returned `Results` object is the one every family returns, so swapping in a
different detector is a one line change. `conf` sets the confidence threshold
and `iou` the NMS threshold. See [prediction](/docs/predict) for sources,
streaming and result handling.

## Variants

Detection and pose are the same architecture under different heads, and they
take the same arguments. The sizes in the table below are the detection ones;
pose is published at those and at one smaller size. The pose head predicts the
COCO keypoint set.

<benchmark-table task="detect" />

<va-embed />

## Train

<code-tabs name="train" />

`epochs`, `lr0` and `amp` are resolved per task when you leave them out, so a
pose run starts from different defaults than a detection run. The optimizer
defaults to AdamW. The class count comes from the dataset YAML and the head is
rebuilt for it before the first epoch; on the pose head the keypoint count is
handled the same way, so a COCO pose checkpoint can be fine-tuned onto a
skeleton of a different size.

Fine-tuning starts from Deci's weights, which is what Deci's license covers.
Training from a randomly initialized model involves no Deci checkpoint at all,
and that is the third snippet above.

See [training](/docs/train) for datasets, augmentation, multi-GPU and loggers.

## Validate

`val()` returns a dictionary of `metrics/` keys covering precision, recall,
mAP 50 and mAP 50-95, measured against any dataset in the format you trained on.

<code-tabs name="val" />

## Export

<export-matrix />

An exported artifact loads back through `LibreYOLO()` on its file suffix, so a
`.onnx` or `.engine` file behaves like a checkpoint and returns the same
`Results`. Running the graph in a bare runtime, with no LibreYOLO installed, is
also supported, but then preprocessing and postprocessing are yours to write.
Each format installs a different extra and takes a few arguments of its own.
Both are on that format's page.

An export is another copy of the same weights in a different container.
Exporting a Deci checkpoint changes neither where the weights came from nor the
license that covers them.

<code-tabs name="export" />

## Checkpoints

There are none to list. Deci's license forbids redistribution, so the LibreYOLO
org publishes no YOLO-NAS weights and the download resolves elsewhere: a name
of the form `LibreYOLONAS<size>.pt`, or `LibreYOLONAS<size>-pose.pt` for pose,
maps to the matching object on Deci's public CDN.

Only the checkpoints whose SHA-256 the library pins can be fetched that way.
Anything else fails closed rather than opening an unverified third-party
pickle, and has to be downloaded by hand and passed as a path. A file already
on disk loads from its path, with no download and no checksum gate. That
includes a Deci `.pth` under its original name, which the loader recognizes.

## Licensing

<provenance-box>

LibreYOLO neither hosts nor mirrors these weights: nothing for this family
exists in the LibreYOLO Hugging Face org. Every auto-download goes to Deci's
public CDN instead, prints Deci's terms once per process before it starts, and
is checked against a pinned SHA-256 before the file is opened.

Training from a randomly initialized model is the alternative. The architecture
is Apache-2.0 upstream and MIT here, so a model trained that way on your own
data derives from no Deci checkpoint.

</provenance-box>

## Citation

YOLO-NAS was released without a paper. The entry below is the one its authors
ask for, covering SuperGradients, the library it shipped in.

<citation-block />
