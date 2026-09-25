---
title: "Альтернатива MiDaS в 2026 году: оценка глубины с LibreYOLO"
description: "Официальный репозиторий MiDaS архивирован. Запускайте модели оценки глубины Small и DPT-Large с помощью LibreYOLO."
date: 2026-08-15
author: Xuban
tags: [LibreYOLO, midas-alternative, depth-estimation, tutorial]
faq:
  - q: "Заброшен ли MiDaS?"
    a: "Официальный репозиторий isl-org/MiDaS был архивирован 25 августа 2025 года."
  - q: "Возвращает ли MiDaS глубину в метрических единицах?"
    a: "Нет. MiDaS возвращает относительную обратную глубину без метрических единиц и фиксированного масштаба между изображениями."
---

[Официальный репозиторий MiDaS](https://github.com/isl-org/MiDaS) был архивирован 25 августа 2025 года. LibreYOLO предоставляет поддерживаемый способ запускать его чекпойнты Small и DPT-Large.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreMiDaSl-depth.pt")
result = model("image.jpg", save=True)
print(result.depth_map.data.shape)
```

Предсказание, валидация в режиме zero-shot и экспорт с фиксированным разрешением поддерживаются. Обучение не поддерживается. На выходе получается относительная обратная глубина: чем выше значение, тем ближе объект, но у неё нет метрических единиц и фиксированного масштаба между изображениями.

Достаточно базовой установки: `pip install libreyolo`. Продолжите с [документации MiDaS](https://www.libreyolo.com/docs/models/midas) или прочитайте про [альтернативу Real-ESRGAN](/articles/real-esrgan-alternative).

[GitHub](https://github.com/LibreYOLO/libreyolo) | [Документация](https://www.libreyolo.com/docs)
