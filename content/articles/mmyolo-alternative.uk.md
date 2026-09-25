---
title: "Альтернатива MMYOLO у 2026 році: запуск RTMDet через LibreYOLO"
description: "Останній випуск MMYOLO датовано 2023 роком. Запускайте підтримувані детектори, зокрема RTMDet, через прямий API LibreYOLO."
date: 2026-08-15
author: Xuban
tags: [LibreYOLO, mmyolo-alternative, rtmdet, object-detection]
faq:
  - q: "Розробку MMYOLO припинено?"
    a: "MMYOLO офіційно не архівовано, але його останній випуск на GitHub, версія 0.6.0, датовано серпнем 2023 року."
  - q: "Чи може LibreYOLO завантажити будь-яку контрольну точку MMYOLO?"
    a: "Ні. LibreYOLO підтримує визначені сімейства моделей і не гарантує пряме завантаження довільних контрольних точок MMYOLO."
---

[MMYOLO](https://github.com/open-mmlab/mmyolo) не архівовано, але його останній випуск, версія 0.6.0, датується серпнем 2023 року. Якщо вам потрібен лише підтримуваний детектор, наприклад RTMDet, повний стек OpenMMLab може бути зайвим.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRTMDets.pt")
result = model("image.jpg", save=True)
print(result.boxes.xyxy)
```

LibreYOLO не є повною заміною MMYOLO. Він не відтворює кожну конфігурацію, рецепт чи контрольну точку. Натомість підтримувані сімейства, як-от RTMDet і YOLOX, мають прямий API для передбачення, навчання, валідації та експорту.

Почніть з `pip install libreyolo`. У [документації RTMDet](https://www.libreyolo.com/docs/models/rtmdet) перелічено підтримувані операції. У [короткому посібнику з RTMDet](/articles/rtmdet-without-mmdetection) описано перехід.

[GitHub](https://github.com/LibreYOLO/libreyolo) | [Документація](https://www.libreyolo.com/docs)
