---
title: "Альтернатива EfficientDet у 2026: запуск D0-D4 через LibreYOLO"
description: "Виконуйте передбачення, валідацію та експорт EfficientDet D0-D4 через API бібліотеки LibreYOLO для виявлення об'єктів."
date: 2026-08-15
author: Xuban
tags: [LibreYOLO, efficientdet-alternative, object-detection, tutorial]
faq:
  - q: "EfficientDet покинули?"
    a: "Репозиторій EfficientDet для PyTorch від rwightman офіційно не архівовано. Його доречніше назвати малоактивним, ніж покинутим."
  - q: "Чи може LibreYOLO навчати EfficientDet?"
    a: "Ні. LibreYOLO підтримує інференс, валідацію та експорт EfficientDet, але не навчання."
---

Популярний [репозиторій EfficientDet для PyTorch](https://github.com/rwightman/efficientdet-pytorch) не архівовано, тож називати його покинутим було б неправильно. Проєкт радше затих. LibreYOLO пропонує інший шлях до розгортання.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreEfficientDetd0.pt")
result = model("image.jpg", save=True)
print(result.boxes.xyxy)
```

LibreYOLO підтримує EfficientDet від D0 до D4 для передбачення та валідації. Експорт у ONNX, TorchScript, OpenVINO і TensorRT покрито перевірками на паритетність. Навчання не реалізовано, а кожен розмір має власну фіксовану нативну роздільну здатність входу.

Встановіть бібліотеку командою `pip install libreyolo`. Відомості про кожен розмір наведено в [документації EfficientDet](https://www.libreyolo.com/docs/models/efficientdet), або порівняйте цю модель з [альтернативою Facebook DETR](/articles/facebook-detr-alternative).

[GitHub](https://github.com/LibreYOLO/libreyolo) | [Документація](https://www.libreyolo.com/docs)
