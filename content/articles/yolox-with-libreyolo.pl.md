---
title: "Alternatywa dla YOLOX w 2026: uruchom YOLOX z LibreYOLO"
description: "Najnowsze wydanie YOLOX pochodzi z 2022 roku. Uruchamiaj, trenuj, waliduj i eksportuj YOLOX przez API LibreYOLO."
date: 2026-08-15
author: Xuban
tags: [LibreYOLO, yolox-alternative, object-detection, tutorial]
faq:
  - q: "Czy YOLOX został porzucony?"
    a: "Oficjalne repozytorium nie jest zarchiwizowane, ale jego najnowsze wydanie, YOLOX 0.3.0, pochodzi z kwietnia 2022 roku. Trafniej mówić o braku aktywności niż o formalnym porzuceniu. Model nadal działa."
  - q: "Czy LibreYOLO umożliwia trenowanie YOLOX?"
    a: "Tak. LibreYOLO obsługuje predykcję, trenowanie, walidację i eksport modeli YOLOX."
---

YOLOX nadal jest użytecznym detektorem na licencji Apache-2.0. Jego [oficjalne repozytorium](https://github.com/Megvii-BaseDetection/YOLOX) nie jest zarchiwizowane, ale najnowsze wydanie, 0.3.0, pochodzi z kwietnia 2022 roku. Problem z utrzymaniem jest realny. Model nadal działa.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLOXs.pt")
result = model("image.jpg", save=True)
print(result.boxes.xyxy)
```

LibreYOLO obsługuje sześć rozmiarów YOLOX, od Nano do X, a predykcję, trenowanie, walidację i eksport udostępnia przez jedno API. Jeśli potrzebny jest dokładny workflow konfiguracji `Exp`, należy użyć oryginalnego projektu.

Zacznij od `pip install libreyolo`. [Dokumentacja YOLOX](https://www.libreyolo.com/docs/models/yolox) opisuje pełne API. Dostępna jest też krótka informacja, że [YOLO-NAS nadal jest utrzymywany](/articles/yolo-nas-with-libreyolo).

[GitHub](https://github.com/LibreYOLO/libreyolo) | [Dokumentacja](https://www.libreyolo.com/docs)
