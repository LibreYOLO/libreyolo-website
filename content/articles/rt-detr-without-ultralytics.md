---
title: "RT-DETR Without Ultralytics: v1, v2 and v4 Under Apache-2.0"
description: "Run and fine-tune RT-DETR, RT-DETRv2 and RT-DETRv4 with Apache-2.0 weights in an MIT library: one API for predict, train, val and export."
date: 2026-09-24
author: Xuban
tags: [LibreYOLO, rt-detr, rtdetr, rt-detrv4, object-detection, license, tutorial]
faq:
  - q: "Is RT-DETR free for commercial use?"
    a: "Yes, if you use a permissive implementation. The original RT-DETR and RT-DETRv2 code (lyuwenyu/RT-DETR) and RT-DETRv4 (RT-DETRs/RT-DETRv4) are Apache-2.0, and LibreYOLO runs all three versions with Apache-2.0 weights in an MIT library. The RT-DETR implementation inside the ultralytics package is AGPL-3.0, with a paid Enterprise License as the alternative for closed-source use. This is general information, not legal advice."
  - q: "What license is RT-DETRv4?"
    a: "Apache-2.0. The LICENSE file at github.com/RT-DETRs/RT-DETRv4 is Apache-2.0. RT-DETRv4 uses a DINOv3 teacher during training only; the released student weights hold no DINOv3 parameters, so Meta's DINOv3 license does not reach them."
  - q: "Can I fine-tune RT-DETR without Ultralytics?"
    a: "Yes. LibreYOLO fine-tunes RT-DETR, RT-DETRv2 and RT-DETRv4 detection checkpoints with model.train(), or libreyolo train on the command line, starting from the published COCO weights. The oriented box models on RT-DETRv2 are inference only."
  - q: "Which RT-DETR weights does LibreYOLO ship?"
    a: "RT-DETR in r18, r34, r50, r50m, r101, l and x; RT-DETRv2 in r18, r34, r50, r50m and r101; RT-DETRv4 in s, m, l and x, all COCO detection at 640 px. RT-DETRv2 also has n, s, m, l and x oriented box models trained on DOTA v1.0 at 1024 px. Every file is Apache-2.0."
---

RT-DETR is a real-time detection transformer from Baidu. It decodes a fixed set of queries instead of a dense grid, so there is no NMS step to tune. Search for it and one of the first implementations you find is the one inside the `ultralytics` Python package. That raises a license question the model itself never had.

## The RT-DETR license depends on the repository

The license belongs to a repository, not to a model. RT-DETR has several, and they do not agree:

| Repository | Covers | License |
|:---|:---|:---|
| [lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR) | RT-DETR and RT-DETRv2, PyTorch and Paddle | Apache-2.0 |
| [RT-DETRs/RT-DETRv4](https://github.com/RT-DETRs/RT-DETRv4) | RT-DETRv4 | Apache-2.0 |
| [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) | RT-DETR inside the `ultralytics` package | AGPL-3.0 or Enterprise License |
| [LibreYOLO](https://github.com/LibreYOLO/libreyolo) | RT-DETR, RT-DETRv2 and RT-DETRv4 | MIT code, Apache-2.0 weights |

The Ultralytics implementation is a solid one. Its [RT-DETR page](https://docs.ultralytics.com/models/rtdetr/) lists pretrained `rtdetr-l.pt` and `rtdetr-x.pt` with training, validation, inference and export. It ships inside a package licensed AGPL-3.0, and Ultralytics sells an Enterprise License for teams that do not want to open-source their whole project. If neither fits, the architecture is available under Apache-2.0 from its own authors. The [YOLO licenses explainer](/articles/yolo-licenses-explained) covers the same pattern across the YOLO family, and [the commercial license guide](/articles/yolo-commercial-license) explains what AGPL-3.0 means for a closed-source product.

## Run RT-DETR with LibreYOLO

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO, SAMPLE_IMAGE

model = LibreYOLO("LibreRTDETRr18.pt")  # downloads from Hugging Face on first use
result = model(SAMPLE_IMAGE, save=True)

for box in result.boxes:
    print(box.cls, box.conf, box.xyxy)
```

The same call works from the command line:

```bash
libreyolo predict model=LibreRTDETRr18.pt source=image.jpg save=True
```

`conf` and `max_det` filter a top-k decode over queries and classes. `iou` is accepted but unused, because there is no NMS. The returned `Results` object is the one every LibreYOLO model returns, so swapping detectors is a one-line change.

## RT-DETR, RT-DETRv2 and RT-DETRv4 in one loader

The version is part of the file name, and `LibreYOLO()` routes on the checkpoint, so all three load the same way:

* **RT-DETR:** `LibreRTDETR` in r18, r34, r50, r50m, r101 (ResNet backbones) and l, x (HGNetv2).
* **RT-DETRv2:** `LibreRTDETRv2` in r18, r34, r50, r50m and r101. It keeps version 1's architecture and changes how deformable attention samples.
* **RT-DETRv4:** `LibreRTDETRv4` in s, m, l and x. It reuses D-FINE's architecture, and its weights come from distilling a DINOv3 teacher into an HGNetv2 student.

All detection weights are trained on COCO and run at 640 px. For reference, the upstream READMEs report 46.5 AP for RT-DETR-R18 and 53.0 AP for RT-DETR-L on COCO, and 49.8 AP for RT-DETRv4-S up to 57.0 AP for RT-DETRv4-X. The [RT-DETR docs page](/docs/models/rt-detr) has LibreYOLO's own benchmark table for every checkpoint.

RT-DETRv2 also does oriented boxes. `LibreRTDETRv2n-obb.pt` through `LibreRTDETRv2x-obb.pt` are the official DOTA v1.0 checkpoints, 15 aerial classes at 1024 px, converted to LibreYOLO's format:

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRTDETRv2n-obb.pt")
result = model("aerial.png", save=True)
print(result.obb.xywhr)  # (N, 5): cx, cy, w, h, radians
```

[Oriented detection](/docs/tasks/oriented-detection) covers the label format and metrics.

## Fine-tune RT-DETR on your own data

Training starts from a published checkpoint:

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRTDETRr18.pt")
model.train(data="coco128.yaml", epochs=50, batch=4, lr0=1e-4)
```

Point `data` at your own dataset YAML for a real run. Each version carries its own learning rate default, and versions 1 and 2 set the backbone rate to a twentieth of `lr0`, following the original recipe. Multi-GPU is `device=0,1` on the CLI, and [LoRA](/docs/train/lora) adapter fine-tuning is `lora=True` with the `lora` extra installed. The oriented models on RT-DETRv2 are inference only. See [training](/docs/train) for datasets, augmentation and loggers.

## Validate and export

```python
metrics = model.val(data="coco128.yaml")
print(metrics["metrics/mAP50-95"])

path = model.export(format="onnx")  # needs: pip install "libreyolo[onnx]"
```

`val()` returns a plain dictionary. Export targets include ONNX, TorchScript, ExecuTorch, TensorRT and OpenVINO; the export matrix on the [model page](/docs/models/rt-detr) shows which are validated for each task. An exported `.onnx` or `.engine` file loads back through `LibreYOLO()` and returns the same `Results`. See [export](/docs/export) for the per-format guides.

## Where the original repos are still the right choice

If you need to reproduce a paper result with the authors' exact configs, want the Paddle version, or want to run RT-DETRv4's DINOv3 teacher distillation yourself, use the upstream repositories. LibreYOLO fine-tunes the released v4 student; it does not run the teacher. And if your project is already AGPL-3.0 or covered by an Ultralytics Enterprise License, there is no license reason to move.

## A note on the license

LibreYOLO's code is MIT. Every RT-DETR weight file is Apache-2.0, and each Hugging Face weight repo ships the upstream LICENSE and NOTICE, which Apache-2.0 asks you to keep when you redistribute the weights. Weights you train on your own data are yours. RT-DETRv4 uses DINOv3 only as a training-time teacher, and the released students hold no DINOv3 parameters. This is general information, not legal advice. Check the current LICENSE files before you ship.

## Try it

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRTDETRv4s.pt")
result = model("image.jpg", save=True)
print(result.boxes.xyxy)
```

LibreYOLO is MIT-licensed, runs on Linux, Mac and Windows, and works on GPU, Apple Silicon and plain CPU with no code change.

[GitHub](https://github.com/LibreYOLO/libreyolo) | [RT-DETR docs](/docs/models/rt-detr)
