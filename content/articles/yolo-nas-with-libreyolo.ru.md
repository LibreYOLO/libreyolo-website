---
title: "YOLO-NAS в LibreYOLO: обучение, валидация и экспорт"
description: "Запускайте, обучайте, валидируйте и экспортируйте YOLO-NAS в LibreYOLO. Последний релиз SuperGradients вышел в апреле 2024 года. Планируется обучить новые веса с нуля, чтобы отказаться от лицензии Deci."
date: 2026-08-20
author: Xuban
tags: [LibreYOLO, yolo-nas, yolo-nas-maintained, object-detection]
faq:
  - q: "YOLO-NAS забросили?"
    a: "Последний релиз SuperGradients, версия 3.7.1, вышел в апреле 2024 года. LibreYOLO поддерживает предсказание, обучение, валидацию и экспорт YOLO-NAS на актуальной версии PyTorch."
  - q: "Ultralytics поддерживает YOLO-NAS?"
    a: "Ultralytics поддерживает инференс, валидацию и экспорт. Обучение не поддерживается."
  - q: "Можно ли использовать предобученные веса YOLO-NAS в коммерческих целях?"
    a: "На предобученные веса Deci распространяются исходные некоммерческие условия вне зависимости от того, какая библиотека их загружает. Обучение со случайной инициализацией позволяет обойтись без этих чекпойнтов. Также планируется опубликовать веса COCO, обученные с нуля."
---

YOLO-NAS по-прежнему остаётся полезным детектором. В исходном репозитории модели, [SuperGradients](https://github.com/Deci-AI/super-gradients), последний релиз 3.7.1 вышел в апреле 2024 года. LibreYOLO поддерживает для неё предсказание, обучение, валидацию и экспорт.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLONASs.pt")
result = model("image.jpg", save=True)
print(result.boxes.xyxy)
```

LibreYOLO поддерживает модели S, M и L для детекции, а также оценку позы. Чекпойнты скачиваются из общедоступной CDN Deci под именами вроде `LibreYOLONASs.pt`. LibreYOLO их не размещает, и на них по-прежнему распространяются некоммерческие условия Deci. SuperGradients ограничивает версию PyTorch диапазоном от 1.9 до 1.13, а LibreYOLO требует версию 2.4 или новее. Для детекции доступны экспорт в ONNX, TorchScript, OpenVINO, NCNN, TFLite, Paddle, MNN, ExecuTorch и Core AI. CoreML не поддерживается.

Чтобы обучить модель без чекпойнта Deci:

```python
from libreyolo import LibreYOLONAS

model = LibreYOLONAS(None, size="s")
model.train(data="my-dataset.yaml", imgsz=640, batch=16)
```

Также планируется обучить YOLO-NAS на COCO с нуля и опубликовать эти веса.

Установите библиотеку командой `pip install libreyolo`. Подробности о модели приведены в [документации YOLO-NAS](https://www.libreyolo.com/docs/models/yolo-nas). Также есть более подробная статья об [альтернативе SuperGradients](/articles/supergradients-alternative).

[GitHub](https://github.com/LibreYOLO/libreyolo) | [Документация](https://www.libreyolo.com/docs)
