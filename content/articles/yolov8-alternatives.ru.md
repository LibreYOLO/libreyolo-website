---
title: "Лучшие альтернативы YOLOv8 в 2026 году: RF-DETR и YOLO-NAS"
description: "RF100-VL показывает, насколько хорошо детектор дообучается на данных вне COCO. RF-DETR и YOLO-NAS набрали больше, чем YOLOv8. Как запустить обе модели в LibreYOLO."
date: 2026-08-20
author: Xuban
tags: [LibreYOLO, YOLOv8 alternative, YOLO8 alternative, RF-DETR, YOLO-NAS, object detection, RF100-VL]
faq:
  - q: "Какая альтернатива YOLOv8 лучше всего подходит в 2026 году?"
    a: "Для GPU — RF-DETR-S. В кампании LibreYOLO на RF100-VL модель набрала 60.41 mAP50-95. В статье о RF100-VL для YOLOv8m указано 56.9. YOLO-NAS-S — вариант на свёрточной архитектуре, его результат — 58.00."
  - q: "RF-DETR лучше переносит обучение на новые данные, чем YOLOv8?"
    a: "На RF100-VL — да. По измерениям LibreYOLO, RF-DETR-S набрал 60.41, а RF-DETR-M — 61.13. Roboflow приводит значения 60.2 и 61.2. В статье для YOLOv8m указано 56.9."
  - q: "Можно ли использовать YOLO-NAS в коммерческих целях?"
    a: "Предобученные веса Deci нельзя использовать в коммерческих целях. Архитектура распространяется под Apache-2.0. LibreYOLO не размещает эти веса. Подробности о лицензии — в статье о YOLO-NAS."
---

Для выбора детектора обычно смотрят на результаты COCO. Есть и другой критерий: насколько хорошо модель дообучается на других данных. Это измеряет [RF100-VL](https://arxiv.org/abs/2505.20612): чекпойнт, предобученный на COCO, обучают 100 эпох на каждом из 100 датасетов, а затем оценивают на тестовой выборке каждого из них. Итоговое значение — среднее mAP50-95.

Для YOLOv8m оно составляет 56.9 — это число из [приложения к статье](https://papers.neurips.cc/paper_files/paper/2025/file/1013f8ff40a194f3f12a6bcc5221bb34-Paper-Datasets_and_Benchmarks_Track.pdf). В запусках LibreYOLO две модели набрали больше: RF-DETR и YOLO-NAS. Результаты запусков доступны в [huggingface.co/datasets/LibreYOLO/rf100-vl-results](https://huggingface.co/datasets/LibreYOLO/rf100-vl-results).

## RF-DETR

В [кампании LibreYOLO](/articles/rf100vl-benchmark) RF-DETR-S набрал 60.41, а RF-DETR-M — 61.13. [По данным Roboflow](https://rfdetr.roboflow.com/latest/learn/benchmarks/), результаты составляют 60.2 и 61.2. Варианты от Nano до Large распространяются под Apache-2.0.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRFDETRs.pt")
model.train(data="my-dataset.yaml", epochs=100, imgsz=512, batch=8)
```

[Документация](/docs/models/rf-detr) | [Веса](https://huggingface.co/LibreYOLO/LibreRFDETRs) | [Запуск `20260814-rfdetr-s-1b1190ee`](https://huggingface.co/datasets/LibreYOLO/rf100-vl-results/tree/main/rfdetr-s/20260814-rfdetr-s-1b1190ee)

## YOLO-NAS

Результат YOLO-NAS-S — 58.00. Это свёрточная нейросеть (CNN) для тех, кому нужна архитектура в стиле YOLO. Опубликованные Deci веса нельзя использовать в коммерческих целях. LibreYOLO их не размещает. Подробности: [YOLO-NAS по-прежнему поддерживается](/articles/yolo-nas-with-libreyolo).

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLONASs.pt")
model.train(data="my-dataset.yaml", epochs=100, imgsz=640, batch=16)
```

[Документация](/docs/models/yolo-nas) | [Запуск `20260809-yolonas-s-02926964`](https://huggingface.co/datasets/LibreYOLO/rf100-vl-results/tree/main/yolonas-s/20260809-yolonas-s-02926964)

Для RF-DETR установите пакет командой `pip install "libreyolo[rfdetr]"`, а для YOLO-NAS — командой `pip install libreyolo`. Страница кампании: [RF100-VL](/articles/rf100vl-benchmark).

[GitHub](https://github.com/LibreYOLO/libreyolo) | [Документация](https://www.libreyolo.com/docs)
