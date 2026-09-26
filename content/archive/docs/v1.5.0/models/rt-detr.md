---
title: RT-DETR
families: [rtdetr]
seo_title: "RT-DETR, RT-DETRv2 and RT-DETRv4 in LibreYOLO"
description: "Use RT-DETR, RT-DETRv2 and RT-DETRv4 in LibreYOLO for object detection, plus oriented boxes on RT-DETRv2. Install, predict, train, validate and export, with Apache-2.0 weights."
lead: "A detection transformer built for real-time inference: it decodes a fixed set of queries rather than a dense grid, so it runs no NMS. LibreYOLO carries three versions of it, told apart by the checkpoint you load, and version 2 also serves oriented boxes."
keywords: [RT-DETR, RT-DETRv2, RT-DETRv4, real-time detection transformer, DETR, object detection, oriented bounding box detection, OBB, DOTA]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreRTDETRr18.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
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
    - label: Oriented boxes
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Version 2 only. The -obb suffix selects the task, and the checkpoint
        # is recognized as oriented from its own tensors, so no task argument
        # is needed. These weights are DOTA v1.0, 15 aerial classes at 1024 px.
        model = LibreYOLO("LibreRTDETRv2n-obb.pt")
        result = model("aerial.png", save=True)

        obb = result.obb
        print(obb.xywhr)     # (N, 5): cx, cy, w, h, radians
        print(obb.xyxyxyxy)  # the same rows as four corner points
        print(result.boxes.xyxy)  # enclosing axis-aligned boxes
    - label: Oriented boxes, CLI
      language: bash
      code: |
        libreyolo predict model=LibreRTDETRv2n-obb.pt source=aerial.png save=True
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
    - label: Oriented boxes
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Oriented validation matches with rotated IoU, so a prediction in the
        # right place at the wrong angle counts as a miss.
        model = LibreYOLO("LibreRTDETRv2n-obb.pt")
        metrics = model.val(data="my-obb-dataset.yaml")

        print(metrics["metrics/mAP50-95(OBB)"])
        print(metrics["metrics/mAP50(OBB)"])
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
    - label: Oriented boxes
      language: bash
      code: |
        # ONNX and TorchScript are the validated targets for the oriented task,
        # at FP32, batch 1, on a fixed 1024 by 1024 canvas.
        libreyolo export model=LibreRTDETRv2n-obb.pt format=onnx imgsz=1024
        libreyolo export model=LibreRTDETRv2n-obb.pt format=torchscript imgsz=1024
    - label: Use the exported file
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # The factory routes on the file suffix, so an exported artifact loads
        # like any checkpoint and returns the same Results object.
        model = LibreYOLO("LibreRTDETRr18.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
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
accepted but unused. An oriented checkpoint fills `result.obb` natively and
also fills `result.boxes` with the enclosing axis-aligned rectangles. See
[prediction](/docs/predict) for sources, streaming and result handling.

## Variants

Three versions, two tasks between them, and the size codes do not run in a
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

### Oriented boxes on version 2

Version 2 is the one version that carries a second task. Its supported tasks
are `detect` and `obb`, and the two do not share a graph or a size series.
Detection uses the ResNet sizes at 640 px; oriented detection uses an HGNetv2
series, n, s, m, l and x, at 1024 px, and the input size resolves per task
rather than per family. A checkpoint is recognized as oriented from its own
tensors, by the five-coordinate box heads and the version 2 sampling
parameters, so `-obb` weights load into the oriented graph without a `task`
argument and a mismatch between the two is a hard error rather than a silent
reinterpretation.

The published files are `LibreRTDETRv2n-obb.pt` through
`LibreRTDETRv2x-obb.pt`. They are the official DOTA v1.0 single-scale
checkpoints converted into LibreYOLO's format, 15 aerial classes from plane and
ship through harbor and helicopter, and their class names are stamped into the
checkpoint. Unlike the detection side, the oriented task is inference only:
prediction, validation and export work, and `train()` on an oriented model
raises. Tracking and test-time augmentation do not support oriented boxes
either. [Oriented detection](/docs/tasks/oriented-detection) covers the task,
the label format and the metrics.

## Train

Training starts from a published checkpoint. `pretrained` is accepted and then
dropped on all three versions, so `pretrained=False` does not give you a
randomly initialized model. Everything in this section is about detection:
version 2's oriented task is inference only, and there is no transfer path from
detection weights to it, because the two use different backbones.

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

Oriented validation runs through the same call and reports the same keys, plus
four repeated under an `(OBB)` suffix. Matching uses rotated IoU rather than
the IoU of the enclosing rectangles, so an angle error is a miss. `augment=True`
is rejected on this task.

## Export

<export-matrix />

The matrix covers the lineage as one page: where the three versions disagree
about a format, the cell shows the weakest of the three, so nothing here is
oversold for whichever version you load. The oriented row belongs to version 2
alone. ONNX and TorchScript are validated there, at FP32, batch 1 and a fixed
1024 by 1024 canvas; OpenVINO, TensorRT and ExecuTorch convert and reload but
have not met raw-output parity across the full query set, so the top boxes
agree to a fraction of a pixel while the tail drifts.

An exported artifact loads back through `LibreYOLO()` on its file suffix, so a
`.onnx` or `.engine` file behaves like a checkpoint and returns the same
`Results`.

<code-tabs name="export" />

## Checkpoints

Every published weight file for this family.

<checkpoint-table />

The file name carries the version, then the size, then the task. Detection
weights are `LibreRTDETR<size>.pt`, `LibreRTDETRv2<size>.pt` and
`LibreRTDETRv4<size>.pt`, all at 640 px. Oriented weights exist for version 2
only and add the task suffix, `LibreRTDETRv2n-obb.pt` through
`LibreRTDETRv2x-obb.pt`, all at 1024 px and trained on DOTA v1.0 rather than
COCO.

## Licensing

<provenance-box></provenance-box>

## Citation

<citation-block />

The block above is what the authors publish for versions 1 and 2 detection.
Version 2's oriented weights have a third upstream, the Apache-2.0 RiO-DETR
repository at
[github.com/RicePasteM/RiO-DETR](https://github.com/RicePasteM/RiO-DETR), which
is where the DOTA checkpoints come from; cite that project if you used one.
Version 4 is
a separate paper by a different group and has its own citation block at
[github.com/RT-DETRs/RT-DETRv4](https://github.com/RT-DETRs/RT-DETRv4#4-citation);
cite that one if you used a version 4 checkpoint.
