---
title: "Альтернатива MMYOLO в 2026 году: запуск RTMDet через LibreYOLO"
description: "Последний релиз MMYOLO вышел в 2023 году. Запускайте поддерживаемые детекторы, например RTMDet, через прямой API LibreYOLO."
date: 2026-08-15
author: Xuban
tags: [LibreYOLO, mmyolo-alternative, rtmdet, object-detection]
faq:
  - q: "MMYOLO заброшен?"
    a: "MMYOLO официально не архивирован, но его последний релиз на GitHub — версия 0.6.0 от августа 2023 года."
  - q: "Может ли LibreYOLO загружать любые чекпойнты MMYOLO?"
    a: "Нет. LibreYOLO поддерживает определённые семейства моделей и не гарантирует загрузку произвольных чекпойнтов MMYOLO без дополнительных изменений."
---

[MMYOLO](https://github.com/open-mmlab/mmyolo) не архивирован, но его последний релиз, версия 0.6.0, вышел в августе 2023 года. Если нужен только поддерживаемый детектор, например RTMDet, весь стек OpenMMLab может не понадобиться.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRTMDets.pt")
result = model("image.jpg", save=True)
print(result.boxes.xyxy)
```

LibreYOLO — не полная замена MMYOLO. Он не воспроизводит все конфигурации, рецепты и чекпойнты. Зато поддерживаемые семейства, например RTMDet и YOLOX, получают прямой API для предсказания, обучения, валидации и экспорта.

Начните с `pip install libreyolo`. В [документации RTMDet](https://www.libreyolo.com/docs/models/rtmdet) перечислены поддерживаемые операции. В [кратком руководстве по RTMDet](/articles/rtmdet-without-mmdetection) описан переход.

[GitHub](https://github.com/LibreYOLO/libreyolo) | [Документация](https://www.libreyolo.com/docs)
