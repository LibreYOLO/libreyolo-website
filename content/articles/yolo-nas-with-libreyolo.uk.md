---
title: "YOLO-NAS у LibreYOLO: підтримка, навчання та експорт"
description: "Запускайте, навчайте, валідуйте й експортуйте YOLO-NAS у LibreYOLO. SuperGradients востаннє випускали у квітні 2024 року. Ми плануємо навчити нові ваги з нуля, щоб відмовитися від ліцензії Deci."
date: 2026-08-20
author: Xuban
tags: [LibreYOLO, yolo-nas, yolo-nas-maintained, object-detection]
faq:
  - q: "Чи покинуто YOLO-NAS?"
    a: "Останній випуск SuperGradients, версії 3.7.1, вийшов у квітні 2024 року. LibreYOLO підтримує передбачення, навчання, валідацію та експорт YOLO-NAS у сучасній версії PyTorch."
  - q: "Чи підтримує Ultralytics YOLO-NAS?"
    a: "Ultralytics підтримує інференс, валідацію та експорт. Навчання не підтримується."
  - q: "Чи можна комерційно використовувати попередньо навчені ваги YOLO-NAS?"
    a: "На попередньо навчені ваги Deci поширюються початкові умови некомерційного використання незалежно від того, яка бібліотека їх завантажує. Навчання з випадкової ініціалізації дає змогу не використовувати ці контрольні точки. Ми також плануємо опублікувати ваги COCO, навчені з нуля."
---

YOLO-NAS досі залишається корисним детектором. У рідному для YOLO-NAS проєкті [SuperGradients](https://github.com/Deci-AI/super-gradients) останній випуск, версії 3.7.1, вийшов у квітні 2024 року. LibreYOLO підтримує передбачення, навчання, валідацію та експорт цієї моделі.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLONASs.pt")
result = model("image.jpg", save=True)
print(result.boxes.xyxy)
```

LibreYOLO підтримує моделі S, M і L для виявлення об'єктів, а також оцінювання пози. Контрольні точки завантажуються з публічного CDN Deci під назвами на кшталт `LibreYOLONASs.pt`. LibreYOLO не розміщує жодної з них, а умови Deci щодо некомерційного використання залишаються чинними. SuperGradients фіксує залежність від PyTorch у діапазоні версій від 1.9 до 1.13, тоді як LibreYOLO потребує версії 2.4 або новішої. Експорт моделей для виявлення об'єктів підтримує ONNX, TorchScript, OpenVINO, NCNN, TFLite, Paddle, MNN, ExecuTorch і Core AI. CoreML не підтримується.

Щоб навчати модель без контрольної точки Deci:

```python
from libreyolo import LibreYOLONAS

model = LibreYOLONAS(None, size="s")
model.train(data="my-dataset.yaml", imgsz=640, batch=16)
```

Ми також плануємо навчити YOLO-NAS з нуля на COCO й опублікувати ці ваги.

Встановіть пакет командою `pip install libreyolo`. У [документації YOLO-NAS](https://www.libreyolo.com/docs/models/yolo-nas) наведено відомості про модель. Докладніше альтернативу описано у статті [альтернатива SuperGradients](/articles/supergradients-alternative).

[GitHub](https://github.com/LibreYOLO/libreyolo) | [Документація](https://www.libreyolo.com/docs)
