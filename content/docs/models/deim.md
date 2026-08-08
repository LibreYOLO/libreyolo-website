---
title: DEIM
families: [deim]
seo_title: "DEIM and DEIMv2 in LibreYOLO"
description: "Use DEIM and DEIMv2 in LibreYOLO for object detection. Install, predict, train, validate and export, from a half-million-parameter size upward."
lead: "A detection transformer trained with dense one-to-one matching, which converges in far fewer epochs than the DETR recipes it builds on. LibreYOLO carries two versions of it, told apart by the checkpoint you load."
keywords: [DEIM, DEIMv2, DINOv3, detection transformer, DETR, object detection, real-time detection]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDEIMn.pt")
        results = model(SAMPLE_IMAGE, save=True)

        for box in results[0].boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreDEIMn.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Video
      language: python
      code: |
        from libreyolo import LibreYOLO

        # The version is part of the file name, and the factory routes on the
        # checkpoint, so both load the same way.
        model = LibreYOLO("LibreDEIMv2pico.pt")

        # Any source the library accepts: file, folder, URL, webcam index,
        # RTSP stream, or a .streams list
        for result in model.predict("clip.mp4", stream=True, save=True):
            print(len(result.boxes))
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDEIMn.pt")

        # coco128.yaml downloads a 128-image sample on first use. Point `data`
        # at your own dataset YAML for a real run.
        model.train(data="coco128.yaml", epochs=50, batch=8, lr0=1e-4)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreDEIMn.pt data=coco128.yaml \
          epochs=50 batch=8 lr0=1e-4
    - label: DEIMv2
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Left unset, epochs, batch, imgsz and lr0 come from the released
        # recipe for the size that was loaded.
        model = LibreYOLO("LibreDEIMv2pico.pt")
        model.train(data="coco128.yaml", epochs=50)
    - label: LoRA
      language: python
      code: |
        # Needs the lora extra: pip install "libreyolo[lora]"
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDEIMn.pt")
        model.train(data="coco128.yaml", epochs=50, lora=True)
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreDEIMn.pt data=coco128.yaml \
          epochs=50 device=0,1
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDEIMn.pt")

        # val() returns a plain dict, not an object
        metrics = model.val(data="coco128.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDEIMn.pt data=coco128.yaml
    - label: Against COCO
      language: bash
      code: |
        # coco-val-only.yaml fetches the 5000 val2017 images and skips the
        # training set. It carries an embedded download script, so it needs
        # explicit permission unless the dataset is already local.
        libreyolo val model=LibreDEIMn.pt data=coco-val-only.yaml \
          allow_download_scripts=True
  export:
    - label: Python
      language: python
      code: |
        # Needs the onnx extra: pip install "libreyolo[onnx]"
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDEIMn.pt")
        path = model.export(format="onnx")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreDEIMn.pt format=onnx
    - label: Use the exported file
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # The factory routes on the file suffix, so an exported artifact loads
        # like any checkpoint and returns the same Results object.
        model = LibreYOLO("LibreDEIMn.onnx")
        results = model(SAMPLE_IMAGE)

        print(results[0].boxes.xyxy)
---

## Install

Neither version needs an optional extra. Everything they import is in the base
install.

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

Version 1 ships five sizes, all at the same input size. Version 2 keeps those
five names and adds three smaller ones, `atto`, `femto` and `pico`, the first
two of which are native at a lower input size than the rest. Five size codes
therefore exist in both versions and name different models; the version is
written into the checkpoint file name.

<benchmark-table task="detect" />

<va-embed />

Version 1 keeps D-FINE's architecture and swaps its classification objective
for the matchability-aware loss from the dense one-to-one recipe, so the two
families share almost every state dict key and are told apart by the metadata
in the checkpoint. Version 2 keeps that training contract and mixes backbones:
HGNetv2 below `s`, and a DINOv3 vision transformer with a spatial tuning
adapter at `s` and above. That backbone is what puts a second license on those
four checkpoints, so read [licensing](#licensing) before you ship one.

## Train

Training starts from a published checkpoint. `pretrained` never reaches the
trainer: version 1 warns that the key is unknown and ignores it, version 2
removes it. Neither gives you a randomly initialized model.

<code-tabs name="train" />

Pass `lr0` yourself on version 1. Its Python `train()` signature defaults to
`4e-4`, the rate from the published COCO recipe, while the family's training
config carries `1e-4` as its fine-tune default, and that lower value is what
the CLI resolves when the argument is absent. The config records the
measurement behind it: at the batch sizes a fine-tune actually uses, on small
datasets, the COCO rate measurably degraded transfer.

Version 2 resolves those defaults itself. Leaving `epochs`, `batch`, `imgsz`
and `lr0` unset makes it read each one from the released recipe for the size
that was loaded, so the small sizes train at their own input resolution without
being told, and a value you pass overrides the recipe. `imgsz` is the argument
it constrains: it has to be a positive multiple of 32, and version 2 raises
before the run starts otherwise.

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

The matrix covers both versions as one page: where they disagree about a
format, the cell shows the weaker of the two, so nothing here is oversold for
whichever version you load.

An exported artifact loads back through `LibreYOLO()` on its file suffix, so a
`.onnx` or `.engine` file behaves like a checkpoint and returns the same
`Results`.

<code-tabs name="export" />

## Checkpoints

Every published weight file for this family.

<checkpoint-table />

## Licensing

<provenance-box>
The four DEIMv2 sizes from S upward take their backbone from DINOv3, so their
weight repositories carry both Apache-2.0 and Meta's DINOv3 License, and
LibreYOLO ships the DINOv3 backbone source under that same agreement. The rest
of this family, including every DEIMv2 size below S, is Apache-2.0 alone.
</provenance-box>

## Citation

<citation-block />

DEIMv2 is a separate paper and has its own citation block at
[github.com/Intellindust-AI-Lab/DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2#5-citation);
cite that one if you used a version 2 checkpoint.
