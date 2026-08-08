---
title: RT-DETR
families: [rtdetr]
seo_title: "RT-DETR, RT-DETRv2 and RT-DETRv4 in LibreYOLO"
description: "Use RT-DETR, RT-DETRv2 and RT-DETRv4 in LibreYOLO for object detection. Install, predict, train, validate and export, with Apache-2.0 weights."
lead: "A detection transformer built for real-time inference: it decodes a fixed set of queries rather than a dense grid, so it runs no NMS. LibreYOLO carries three versions of it, told apart by the checkpoint you load."
keywords: [RT-DETR, RT-DETRv2, RT-DETRv4, real-time detection transformer, DETR, object detection]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreRTDETRr18.pt")
        results = model(SAMPLE_IMAGE, save=True)

        for box in results[0].boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreRTDETRr18.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Video
      language: python
      code: |
        from libreyolo import LibreYOLO

        # The version is part of the file name, and the factory routes on the
        # checkpoint, so all three load the same way.
        model = LibreYOLO("LibreRTDETRv4s.pt")

        # Any source the library accepts: file, folder, URL, webcam index,
        # RTSP stream, or a .streams list
        for result in model.predict("clip.mp4", stream=True, save=True):
            print(len(result.boxes))
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTDETRr18.pt")

        # coco128.yaml downloads a 128-image sample on first use. Point `data`
        # at your own dataset YAML for a real run.
        model.train(data="coco128.yaml", epochs=50, batch=4, lr0=1e-4)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreRTDETRr18.pt data=coco128.yaml \
          epochs=50 batch=4 lr0=1e-4
    - label: LoRA
      language: python
      code: |
        # Needs the lora extra: pip install "libreyolo[lora]"
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTDETRr18.pt")
        model.train(data="coco128.yaml", epochs=50, lora=True)
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreRTDETRr18.pt data=coco128.yaml \
          epochs=50 device=0,1
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTDETRr18.pt")

        # val() returns a plain dict, not an object
        metrics = model.val(data="coco128.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreRTDETRr18.pt data=coco128.yaml
    - label: Against COCO
      language: bash
      code: |
        # coco-val-only.yaml fetches the 5000 val2017 images and skips the
        # training set. It carries an embedded download script, so it needs
        # explicit permission unless the dataset is already local.
        libreyolo val model=LibreRTDETRr18.pt data=coco-val-only.yaml \
          allow_download_scripts=True
  export:
    - label: Python
      language: python
      code: |
        # Needs the onnx extra: pip install "libreyolo[onnx]"
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTDETRr18.pt")
        path = model.export(format="onnx")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreRTDETRr18.pt format=onnx
    - label: Use the exported file
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # The factory routes on the file suffix, so an exported artifact loads
        # like any checkpoint and returns the same Results object.
        model = LibreYOLO("LibreRTDETRr18.onnx")
        results = model(SAMPLE_IMAGE)

        print(results[0].boxes.xyxy)
---

## Install

RT-DETR needs no optional extra. Everything it imports is in the base install,
and the `rtdetr` extra is a stable name that adds nothing to it.

```bash
pip install libreyolo
```

Adapter fine-tuning with `lora=True` is the exception, and needs the `lora`
extra.

```bash
pip install "libreyolo[lora]"
```

## Predict

Weights download from Hugging Face on first use and are cached locally.

<code-tabs name="predict" />

The returned `Results` object is the one every family returns, so swapping in a
different detector is a one line change. `conf` and `max_det` filter a top-k
decode over queries and classes; there is no NMS step to tune, and `iou` is
accepted but unused. See [prediction](/docs/predict) for sources, streaming and
result handling.

## Variants

Three versions, one task between them, and the size codes do not run in a
single series. Version 1 names its sizes after the backbone, ResNet or
HGNetv2. Version 2 reuses the ResNet names only: version 1 already ships the
two HGNetv2 sizes, and version 2's results there were close enough that
LibreYOLO publishes no duplicate weights for them. Version 4 uses a plain
letter series, which collides with version 1's HGNetv2 names, so a size code
on its own does not identify a model. The version is written into the
checkpoint file name.

<benchmark-table task="detect" />

<va-embed />

Version 2 keeps version 1's architecture and state dict layout and changes how
the deformable attention samples, which is why the two are told apart by the
metadata in the checkpoint rather than by shape. Version 4 is a different
lineage: it reuses D-FINE's architecture and trainer, and its weights come from
distilling a DINOv3 vision foundation model teacher into an HGNetv2 student. In
LibreYOLO `LibreRTDETRv4` is a subclass of `LibreDFINE` with the mask head
pinned off, so it stays detection only.

## Train

Training starts from a published checkpoint. `pretrained` is accepted and then
dropped on all three versions, so `pretrained=False` does not give you a
randomly initialized model.

<code-tabs name="train" />

Learning rate is the argument to get right, and each version carries its own
default rather than the library-wide one. The Python `train()` signature reads
it from that version's training config, and the CLI resolves the same value
when `lr0` is not passed. Versions 1 and 2 also take `lr_backbone` and default
it to a twentieth of `lr0`, following the original recipe; version 4 runs
through the D-FINE trainer, which scales the backbone parameter group with
`backbone_lr_mult` instead.

Leave `imgsz` at the checkpoint's native size unless you have a reason to
change it. Validation and prediction at other sizes work, with one residual: a
rectangular size whose token count matches the native size still reuses an
embedding built for the wrong aspect ratio.

See [training](/docs/train) for datasets, augmentation, multi-GPU and loggers.

## Validate

`val()` returns a dictionary of `metrics/` keys covering precision, recall,
mAP 50 and mAP 50-95, measured against any dataset in the format you trained on.

<code-tabs name="val" />

The rows in the benchmark table above come from the LibreYOLO benchmark
harness; the note under that table records which dataset produced them and
links the run records.

## Export

<export-matrix />

The matrix covers the lineage as one page: where the three versions disagree
about a format, the cell shows the weakest of the three, so nothing here is
oversold for whichever version you load.

An exported artifact loads back through `LibreYOLO()` on its file suffix, so a
`.onnx` or `.engine` file behaves like a checkpoint and returns the same
`Results`.

<code-tabs name="export" />

## Checkpoints

Every published weight file for this family.

<checkpoint-table />

## Licensing

<provenance-box></provenance-box>

## Citation

<citation-block />

The block above is what the authors publish for versions 1 and 2. Version 4 is
a separate paper by a different group and has its own citation block at
[github.com/RT-DETRs/RT-DETRv4](https://github.com/RT-DETRs/RT-DETRv4#4-citation);
cite that one if you used a version 4 checkpoint.
