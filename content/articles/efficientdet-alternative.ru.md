---
title: "Альтернатива EfficientDet в 2026 году: запуск D0–D4 через LibreYOLO"
description: "Запускайте предсказание и валидацию EfficientDet D0–D4, а также экспортируйте модели через API LibreYOLO для детекции объектов."
date: 2026-08-15
author: Xuban
tags: [LibreYOLO, efficientdet-alternative, object-detection, tutorial]
faq:
  - q: "EfficientDet заброшен?"
    a: "Репозиторий PyTorch от rwightman официально не архивирован. Точнее сказать, что он затих, а не заброшен."
  - q: "Можно ли обучать EfficientDet в LibreYOLO?"
    a: "Нет. LibreYOLO поддерживает инференс, валидацию и экспорт EfficientDet, но не обучение."
---

Популярный [репозиторий EfficientDet PyTorch](https://github.com/rwightman/efficientdet-pytorch) не архивирован, поэтому неверно называть его заброшенным. Он затих. LibreYOLO предлагает ещё один путь к развёртыванию.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreEfficientDetd0.pt")
result = model("image.jpg", save=True)
print(result.boxes.xyxy)
```

LibreYOLO поддерживает предсказание и валидацию EfficientDet от D0 до D4. Для экспорта в ONNX, TorchScript, OpenVINO и TensorRT проверено соответствие результатов. Обучение не реализовано, и для каждого размера сохраняется фиксированное исходное разрешение входных данных.

Установите библиотеку командой `pip install libreyolo`. Сведения о каждом размере приведены в [документации EfficientDet](https://www.libreyolo.com/docs/models/efficientdet); также можно сравнить модель с [альтернативой Facebook DETR](/articles/facebook-detr-alternative).

[GitHub](https://github.com/LibreYOLO/libreyolo) | [Документация](https://www.libreyolo.com/docs)
