---
title: "YOLOX в 2026 году: запуск, обучение и экспорт через LibreYOLO"
description: "Последний релиз YOLOX вышел в 2022 году. Запускайте и обучайте YOLOX, проводите валидацию и экспортируйте модель через API LibreYOLO."
date: 2026-08-15
author: Xuban
tags: [LibreYOLO, yolox-alternative, object-detection, tutorial]
faq:
  - q: "YOLOX забросили?"
    a: "Официальный репозиторий не архивирован, но последний релиз — YOLOX 0.3.0 — вышел в апреле 2022 года. Точнее сказать, что разработка затихла, а не что проект официально забросили."
  - q: "Можно ли обучать YOLOX в LibreYOLO?"
    a: "Да. LibreYOLO поддерживает предсказание, обучение, валидацию и экспорт YOLOX."
---

YOLOX по-прежнему остаётся полезным детектором под Apache-2.0. Его [официальный репозиторий](https://github.com/Megvii-BaseDetection/YOLOX) не архивирован, но последний релиз, 0.3.0, вышел в апреле 2022 года. Проблема с поддержкой реальна. Сама модель всё ещё работает.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLOXs.pt")
result = model("image.jpg", save=True)
print(result.boxes.xyxy)
```

LibreYOLO поддерживает шесть размеров YOLOX — от Nano до X. Через единый API доступны предсказание, обучение, валидация и экспорт. Используйте исходный проект, если вам нужен его точный рабочий процесс с конфигурациями `Exp`.

Для начала установите пакет командой `pip install libreyolo`. В [документации YOLOX](https://www.libreyolo.com/docs/models/yolox) описан полный API. Также есть короткая заметка о том, что [YOLO-NAS по-прежнему поддерживается](/articles/yolo-nas-with-libreyolo).

[GitHub](https://github.com/LibreYOLO/libreyolo) | [Документация](https://www.libreyolo.com/docs)
