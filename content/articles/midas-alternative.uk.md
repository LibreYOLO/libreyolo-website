---
title: "Альтернатива MiDaS у 2026: оцінювання глибини в LibreYOLO"
description: "Офіційний репозиторій MiDaS архівовано. Запускайте моделі оцінювання глибини Small і DPT-Large через LibreYOLO."
date: 2026-08-15
author: Xuban
tags: [LibreYOLO, midas-alternative, depth-estimation, tutorial]
faq:
  - q: "Чи припинили підтримувати MiDaS?"
    a: "Офіційний репозиторій isl-org/MiDaS архівовано 25 серпня 2025 року."
  - q: "Чи повертає MiDaS глибину в метричних одиницях?"
    a: "Ні. MiDaS повертає відносну обернену глибину без метричної одиниці та фіксованого масштабу між зображеннями."
---

[Офіційний репозиторій MiDaS](https://github.com/isl-org/MiDaS) архівовано 25 серпня 2025 року. LibreYOLO надає актуальний спосіб запускати контрольні точки Small і DPT-Large.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreMiDaSl-depth.pt")
result = model("image.jpg", save=True)
print(result.depth_map.data.shape)
```

Підтримуються передбачення, zero-shot валідація та експорт із фіксованою роздільною здатністю. Навчання не підтримується. Результат є відносною оберненою глибиною: що вище значення, то ближче об'єкт, але між зображеннями немає метричної одиниці чи фіксованого масштабу.

Для початку достатньо базового встановлення: `pip install libreyolo`. Продовжіть із [документацією MiDaS](https://www.libreyolo.com/docs/models/midas) або перегляньте [альтернативу Real-ESRGAN](/articles/real-esrgan-alternative).

[GitHub](https://github.com/LibreYOLO/libreyolo) | [Документація](https://www.libreyolo.com/docs)
