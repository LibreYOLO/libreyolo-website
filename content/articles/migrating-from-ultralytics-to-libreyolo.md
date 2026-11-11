---
title: Migrating from Ultralytics to LibreYOLO
description: What stays the same and what changes when you move a project from the Ultralytics package to LibreYOLO. The API, your dataset, your weights and the licence.
date: 2026-11-11
author: Xuban
tags: [LibreYOLO, Ultralytics, migration, MIT, AGPL, tutorial]
---

If your project uses the `ultralytics` package and you want an MIT library, this is what the move looks like. Your dataset comes with you unchanged. Your code changes in a few lines. Your trained weights stay behind.

## The API side by side

| Step | Ultralytics | LibreYOLO |
| --- | --- | --- |
| Install | `pip install ultralytics` | `pip install libreyolo` |
| Import | `from ultralytics import YOLO` | `from libreyolo import LibreYOLO` |
| Load | `YOLO("yolo11n.pt")` | `LibreYOLO("LibreYOLO9t.pt")` |
| Predict | `model("image.jpg")` | `model("image.jpg")` |
| Train | `model.train(data="data.yaml", epochs=100)` | `model.train(data="data.yaml", epochs=100)` |
| Validate | `model.val(data="data.yaml")` | `model.val(data="data.yaml")` |
| Export | `model.export(format="onnx")` | `model.export(format="onnx")` |
| Track | `model.track("video.mp4")` | `model.track("video.mp4")` |
| CLI | `yolo train model=yolo11n.pt data=data.yaml` | `libreyolo train model=yolo9-t data=data.yaml` |

A full round in LibreYOLO:

```python
from libreyolo import LibreYOLO

if __name__ == "__main__":                       # needed on macOS and Windows
    model = LibreYOLO("LibreYOLO9t.pt")          # downloads on first use

    result = model("guggenheim-bilbao.jpg")      # one image gives one result
    print(result.boxes.xyxy, result.boxes.conf, result.boxes.cls)

    out = model.train(data="data.yaml", epochs=100, imgsz=640)
    best = LibreYOLO(out["best_checkpoint"])
    metrics = best.val(data="data.yaml")
    print(metrics["metrics/mAP50-95"])

    best.export(format="onnx")                   # pip install "libreyolo[onnx]"
```

## What changes

| | Ultralytics | LibreYOLO |
| --- | --- | --- |
| One image returns | a list, so `results[0]` | one result, so `result` |
| `val()` returns | an object, `metrics.box.map` | a dict, `metrics["metrics/mAP50-95"]` |
| `train()` returns | | a dict with `best_checkpoint` |
| Save the drawn image | `results[0].save()` | `model("image.jpg", save=True)` |
| Tracker | a YAML file, `tracker="bytetrack.yaml"` | a name, `tracker="bytetrack"` |
| Hyperparameter search | `model.tune()` | not available |
| `show()` `save_txt()` `save_crop()` | available | not available |

`boxes`, `masks`, `keypoints`, `obb` and `probs` keep their attribute names (`xyxy`, `xywh`, `conf`, `cls` and so on).

LibreYOLO predicts with `conf=0.25` and `iou=0.45`. If you tuned your thresholds, pass them explicitly.

In `train()` the arguments `data`, `epochs`, `imgsz`, `batch`, `device`, `workers`, `optimizer`, `lr0`, `patience` and `resume` keep their names. Others don't: the closest to `lrf` is `min_lr_ratio` and the closest to `close_mosaic` is `no_aug_epochs`. LibreYOLO warns about arguments it doesn't know and ignores them, so read the first lines of the training log.

## Your dataset

There is nothing to convert. LibreYOLO reads the same dataset YAML (`path`, `train`, `val`, `test`, `names`) and the same label files for detection, segmentation, pose and oriented boxes. It also reads COCO JSON directly through an `annotations:` key.

Two differences. Datasets are looked up under `~/datasets` or `LIBREYOLO_DATASETS_DIR`. A Python `download:` script inside a YAML runs only if you pass `allow_download_scripts=True`.

## Your weights

LibreYOLO has no port of Ultralytics YOLOv5, YOLOv8, YOLO11 or YOLO26 and does not load checkpoints trained with the Ultralytics package. Ultralytics says the models you train with its code [fall under AGPL-3.0 by default](https://www.ultralytics.com/license), and converting a file does not change its terms. So we don't ship a converter.

The clean route is to retrain on your own data, starting from weights with a permissive licence:

| Task | Families | Pretrained weights |
| --- | --- | --- |
| Detection | YOLOv9, RF-DETR, YOLOX, RT-DETR, D-FINE | MIT for YOLOv9, Apache-2.0 for the rest |
| Segmentation | RF-DETR, D-FINE, RTMDet | Apache-2.0 |
| Pose | RF-DETR, EdgeCrafter | Apache-2.0 |
| Oriented boxes | RF-DETR | CC BY 4.0 |
| Classification | ResNet, ConvNeXt, MobileNetV4, EfficientNetV2 | Apache-2.0 |

A few files carry other terms (the YOLO-NAS and SegFormer weights are non-commercial, for example). The licence on each [Hugging Face model card](https://huggingface.co/LibreYOLO) is the one that counts.

Then compare the old and the new model on your own validation set before you change anything in production.

## When to stay with Ultralytics

If AGPL-3.0 suits your project, or you hold an Enterprise Licence and it works for you, there is no reason to move. Ultralytics has things LibreYOLO doesn't: the YOLO11 and YOLO26 models themselves, one model family that covers every task, built-in hyperparameter search and a hosted platform.

## The licence

LibreYOLO's code is MIT. Ultralytics publishes its package under [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) or its Enterprise Licence, and its [licence page](https://www.ultralytics.com/license) lists the uses for which it asks for the Enterprise one. That page is Ultralytics' guidance on how it applies AGPL-3.0. Read it next to the licence text and ask a lawyer about your own case. This article is not legal advice.

*Disclosure: LibreYOLO is our project. The facts about Ultralytics were checked on 11 November 2026 against their public pages and `ultralytics` 8.4.210. Ultralytics is a trademark of Ultralytics Inc. LibreYOLO is not affiliated with or endorsed by Ultralytics.*
