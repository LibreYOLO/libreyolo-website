---
title: "Альтернатива YOLOX у 2026 році: запуск і навчання в LibreYOLO"
description: "Останній реліз YOLOX вийшов у 2022 році. Запускайте, навчайте, валідуйте й експортуйте YOLOX через API LibreYOLO."
date: 2026-08-15
author: Xuban
tags: [LibreYOLO, yolox-alternative, object-detection, tutorial]
faq:
  - q: "YOLOX покинули?"
    a: "Офіційний репозиторій не архівовано, але останній реліз, YOLOX 0.3.0, вийшов у квітні 2022 року. Точніше сказати, що активність проєкту низька, а не що його офіційно покинули."
  - q: "Чи може LibreYOLO навчати YOLOX?"
    a: "Так. LibreYOLO підтримує передбачення, навчання, валідацію та експорт YOLOX."
---

YOLOX і досі залишається корисним детектором під ліцензією Apache-2.0. Його [офіційний репозиторій](https://github.com/Megvii-BaseDetection/YOLOX) не архівовано, але останній реліз, 0.3.0, датовано квітнем 2022 року. Проблема з підтримкою реальна. Сама модель досі працює.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLOXs.pt")
result = model("image.jpg", save=True)
print(result.boxes.xyxy)
```

LibreYOLO підтримує шість розмірів YOLOX, від Nano до X, і надає через єдиний API передбачення, навчання, валідацію та експорт. Використовуйте оригінальний проєкт, якщо потрібен його точний робочий процес із конфігурацією `Exp`.

Почніть із команди `pip install libreyolo`. У [документації YOLOX](https://www.libreyolo.com/docs/models/yolox) описано повний API. Також є коротка примітка про те, що [YOLO-NAS досі підтримується](/articles/yolo-nas-with-libreyolo).

[GitHub](https://github.com/LibreYOLO/libreyolo) | [Документація](https://www.libreyolo.com/docs)
