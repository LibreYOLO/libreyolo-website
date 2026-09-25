---
title: "Альтернатива Facebook DETR у 2026 році: запуск через LibreYOLO"
description: "Оригінальний репозиторій Facebook Research DETR заархівовано. Запускайте чотири офіційні варіанти DETR через LibreYOLO."
date: 2026-08-15
author: Xuban
tags: [LibreYOLO, facebook-detr-alternative, object-detection, transformer]
faq:
  - q: "Facebook DETR покинули?"
    a: "Оригінальний репозиторій facebookresearch/detr заархівували 12 березня 2024 року."
  - q: "Чи може LibreYOLO навчати оригінальну DETR?"
    a: "Ні. LibreYOLO підтримує передбачення, валідацію та експорт DETR, але не її рецепт навчання протягом 500 епох."
---

Оригінальний [репозиторій Facebook Research DETR](https://github.com/facebookresearch/detr) заархівували 12 березня 2024 року. LibreYOLO і далі дає змогу використовувати початковий детектор через той самий API, що й інші сімейства моделей.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreDETRr50.pt")
result = model("image.jpg", save=True)
print(result.boxes.xyxy)
```

Доступні чотири варіанти з ResNet-50 і ResNet-101 для передбачення, валідації та експорту в ONNX і TorchScript. Навчання не реалізовано. LibreYOLO також використовує фіксований вхідний розмір 800 на 800 замість відтворення оригінального конвеєра оцінювання зі збереженням співвідношення сторін.

Спробуйте після встановлення за допомогою `pip install libreyolo`. У [документації DETR](https://www.libreyolo.com/docs/models/detr) перелічено всі чотири контрольні точки. Для порівняння з CNN дивіться [альтернативу EfficientDet](/articles/efficientdet-alternative).

[GitHub](https://github.com/LibreYOLO/libreyolo) | [Документація](https://www.libreyolo.com/docs)
