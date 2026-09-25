---
title: "Die besten YOLOv8-Alternativen 2026: RF-DETR und YOLO-NAS"
description: "RF100-VL misst, wie gut sich ein Detektor mit neuen Daten nachtrainieren lässt. RF-DETR und YOLO-NAS schneiden besser ab als YOLOv8. So führst du beide in LibreYOLO aus."
date: 2026-08-20
author: Xuban
tags: [LibreYOLO, YOLOv8 alternative, YOLO8 alternative, RF-DETR, YOLO-NAS, object detection, RF100-VL]
faq:
  - q: "Was ist 2026 die beste YOLOv8-Alternative?"
    a: "RF-DETR-S für eine GPU. Das Modell erzielte in LibreYOLOs RF100-VL-Kampagne 60.41 mAP50-95. Das RF100-VL-Paper gibt für YOLOv8m den Wert 56.9 an. YOLO-NAS-S ist die konvolutionale Option mit 58.00."
  - q: "Überträgt RF-DETR besser als YOLOv8?"
    a: "Auf RF100-VL: ja. LibreYOLO maß für RF-DETR-S einen Wert von 60.41 und für RF-DETR-M 61.13. Roboflow gibt 60.2 und 61.2 an. Das Paper nennt für YOLOv8m den Wert 56.9."
  - q: "Kann ich YOLO-NAS kommerziell nutzen?"
    a: "Die vortrainierten Gewichte von Deci sind nicht kommerziell nutzbar. Die Architektur steht unter Apache-2.0. LibreYOLO hostet keine dieser Gewichte. Details findest du im Artikel zu YOLO-NAS."
---

COCO ist die übliche Grundlage für die Wahl eines Detektors. Ein anderer Blickwinkel ist, wie gut sich das Modell mit anderen Daten nachtrainieren lässt. [RF100-VL](https://arxiv.org/abs/2505.20612) misst genau das: Ein auf COCO vortrainierter Checkpoint wird auf jedem von 100 Datensätzen 100 Epochen lang trainiert und anschließend auf jeder Testaufteilung bewertet. Der Wert ist der mittlere mAP50-95.

YOLOv8m erreicht dort laut [Anhang des Papers](https://papers.neurips.cc/paper_files/paper/2025/file/1013f8ff40a194f3f12a6bcc5221bb34-Paper-Datasets_and_Benchmarks_Track.pdf) 56.9. Zwei von uns ausgeführte Modelle schneiden besser ab: RF-DETR und YOLO-NAS. Die unveränderten Ergebnisse findest du unter [huggingface.co/datasets/LibreYOLO/rf100-vl-results](https://huggingface.co/datasets/LibreYOLO/rf100-vl-results).

## RF-DETR

RF-DETR-S erreicht in [LibreYOLOs Kampagne](/articles/rf100vl-benchmark) 60.41 und RF-DETR-M 61.13. [Roboflow gibt](https://rfdetr.roboflow.com/latest/learn/benchmarks/) 60.2 und 61.2 an. Nano bis Large stehen unter Apache-2.0.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRFDETRs.pt")
model.train(data="my-dataset.yaml", epochs=100, imgsz=512, batch=8)
```

[Dokumentation](/docs/models/rf-detr) | [Gewichte](https://huggingface.co/LibreYOLO/LibreRFDETRs) | [Lauf `20260814-rfdetr-s-1b1190ee`](https://huggingface.co/datasets/LibreYOLO/rf100-vl-results/tree/main/rfdetr-s/20260814-rfdetr-s-1b1190ee)

## YOLO-NAS

YOLO-NAS-S erreicht 58.00 und ist die konvolutionale Option, wenn du ein YOLO-artiges CNN möchtest. Die veröffentlichten Gewichte von Deci sind nicht kommerziell nutzbar. LibreYOLO hostet keine davon. Details: [YOLO-NAS wird weiterhin gepflegt](/articles/yolo-nas-with-libreyolo).

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLONASs.pt")
model.train(data="my-dataset.yaml", epochs=100, imgsz=640, batch=16)
```

[Dokumentation](/docs/models/yolo-nas) | [Lauf `20260809-yolonas-s-02926964`](https://huggingface.co/datasets/LibreYOLO/rf100-vl-results/tree/main/yolonas-s/20260809-yolonas-s-02926964)

Installiere RF-DETR mit `pip install "libreyolo[rfdetr]"` oder YOLO-NAS mit `pip install libreyolo`. Kampagnenseite: [RF100-VL](/articles/rf100vl-benchmark).

[GitHub](https://github.com/LibreYOLO/libreyolo) | [Dokumentation](https://www.libreyolo.com/docs)
