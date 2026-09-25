---
title: "Альтернативи YOLOv8 у 2026 році: RF-DETR і YOLO-NAS"
description: "RF100-VL оцінює, наскільки добре детектор донавчається поза COCO. RF-DETR і YOLO-NAS набирають більше за YOLOv8. Як запускати обидві моделі в LibreYOLO."
date: 2026-08-20
author: Xuban
tags: [LibreYOLO, YOLOv8 alternative, YOLO8 alternative, RF-DETR, YOLO-NAS, object detection, RF100-VL]
faq:
  - q: "Яка найкраща альтернатива YOLOv8 у 2026 році?"
    a: "RF-DETR-S для GPU. У кампанії LibreYOLO на RF100-VL модель набрала 60.41 mAP50-95. У статті про RF100-VL для YOLOv8m наведено 56.9. Варіант із згортковою мережею, YOLO-NAS-S, набирає 58.00."
  - q: "Чи краще RF-DETR переносить знання на нові датасети, ніж YOLOv8?"
    a: "За результатами RF100-VL, так. LibreYOLO виміряла 60.41 для RF-DETR-S і 61.13 для RF-DETR-M. Roboflow наводить 60.2 і 61.2. У статті для YOLOv8m зазначено 56.9."
  - q: "Чи можна використовувати YOLO-NAS комерційно?"
    a: "Попередньо навчені ваги Deci не можна використовувати комерційно. Архітектура поширюється за Apache-2.0. Деталі ліцензії наведено у статті про YOLO-NAS."
---

Зазвичай детектор обирають за результатами на COCO. Інший підхід полягає в тому, щоб оцінити, наскільки добре модель донавчається на інших даних. Саме це вимірює [RF100-VL](https://arxiv.org/abs/2505.20612): контрольну точку, попередньо навчену на COCO, навчають 100 епох на кожному зі 100 датасетів, а потім оцінюють на кожному тестовому розбитті. Показник дорівнює середньому mAP50-95.

На RF100-VL модель YOLOv8m набирає 56.9, згідно з [додатком до статті](https://papers.neurips.cc/paper_files/paper/2025/file/1013f8ff40a194f3f12a6bcc5221bb34-Paper-Datasets_and_Benchmarks_Track.pdf). Дві протестовані нами моделі набирають більше: RF-DETR і YOLO-NAS. Необроблені результати запусків доступні на [huggingface.co/datasets/LibreYOLO/rf100-vl-results](https://huggingface.co/datasets/LibreYOLO/rf100-vl-results).

## RF-DETR

У [кампанії LibreYOLO](/articles/rf100vl-benchmark) RF-DETR-S набирає 60.41, а RF-DETR-M, 61.13. [Roboflow наводить](https://rfdetr.roboflow.com/latest/learn/benchmarks/) результати 60.2 і 61.2. Моделі від Nano до Large поширюються за Apache-2.0.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRFDETRs.pt")
model.train(data="my-dataset.yaml", epochs=100, imgsz=512, batch=8)
```

[Документація](/docs/models/rf-detr) | [Ваги](https://huggingface.co/LibreYOLO/LibreRFDETRs) | [Запуск `20260814-rfdetr-s-1b1190ee`](https://huggingface.co/datasets/LibreYOLO/rf100-vl-results/tree/main/rfdetr-s/20260814-rfdetr-s-1b1190ee)

## YOLO-NAS

YOLO-NAS-S набирає 58.00. Це варіант зі згортковою мережею, якщо потрібна CNN у стилі YOLO. Опубліковані Deci ваги не можна використовувати комерційно. LibreYOLO не розміщує їх. Деталі: [YOLO-NAS досі підтримується](/articles/yolo-nas-with-libreyolo).

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLONASs.pt")
model.train(data="my-dataset.yaml", epochs=100, imgsz=640, batch=16)
```

[Документація](/docs/models/yolo-nas) | [Запуск `20260809-yolonas-s-02926964`](https://huggingface.co/datasets/LibreYOLO/rf100-vl-results/tree/main/yolonas-s/20260809-yolonas-s-02926964)

Установіть RF-DETR командою `pip install "libreyolo[rfdetr]"`, а YOLO-NAS командою `pip install libreyolo`. Сторінка кампанії: [RF100-VL](/articles/rf100vl-benchmark).

[GitHub](https://github.com/LibreYOLO/libreyolo) | [Документація](https://www.libreyolo.com/docs)
