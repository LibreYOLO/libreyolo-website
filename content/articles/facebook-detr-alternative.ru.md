---
title: "Альтернатива Facebook DETR в 2026 году: запуск через LibreYOLO"
description: "Исходный репозиторий Facebook Research DETR архивирован. Запускайте четыре официальных варианта DETR через LibreYOLO."
date: 2026-08-15
author: Xuban
tags: [LibreYOLO, facebook-detr-alternative, object-detection, transformer]
faq:
  - q: "Facebook DETR заброшен?"
    a: "Исходный репозиторий facebookresearch/detr архивировали 12 марта 2024 года."
  - q: "Может ли LibreYOLO обучать исходную модель DETR?"
    a: "Нет. LibreYOLO поддерживает предсказание, валидацию и экспорт DETR, но не его рецепт обучения на 500 эпох."
---

Исходный [репозиторий Facebook Research DETR](https://github.com/facebookresearch/detr) архивировали 12 марта 2024 года. LibreYOLO позволяет продолжать использовать исходный детектор через тот же API, что и другие семейства моделей.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreDETRr50.pt")
result = model("image.jpg", save=True)
print(result.boxes.xyxy)
```

Доступны четыре варианта с ResNet-50 и ResNet-101 для предсказания, валидации, экспорта в ONNX и TorchScript. Обучение не реализовано. Кроме того, LibreYOLO использует фиксированный входной размер 800 на 800, а не воспроизводит исходный конвейер оценки с сохранением пропорций изображения.

Попробуйте модель после установки `pip install libreyolo`. В [документации по DETR](https://www.libreyolo.com/docs/models/detr) перечислены все четыре чекпойнта. Для сравнения с CNN см. [альтернативу EfficientDet](/articles/efficientdet-alternative).

[GitHub](https://github.com/LibreYOLO/libreyolo) | [Документация](https://www.libreyolo.com/docs)
