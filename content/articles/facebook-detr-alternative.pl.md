---
title: "Alternatywa dla Facebook DETR: uruchom DETR w LibreYOLO (2026)"
description: "Oryginalne repozytorium Facebook Research DETR zostało zarchiwizowane. Uruchom cztery oficjalne warianty DETR w LibreYOLO."
date: 2026-08-15
author: Xuban
tags: [LibreYOLO, facebook-detr-alternative, object-detection, transformer]
faq:
  - q: "Czy Facebook DETR został porzucony?"
    a: "Oryginalne repozytorium facebookresearch/detr zostało zarchiwizowane 12 marca 2024 roku."
  - q: "Czy LibreYOLO może trenować oryginalny DETR?"
    a: "Nie. LibreYOLO obsługuje predykcję, walidację i eksport DETR, ale nie obsługuje oryginalnej procedury trenowania trwającej 500 epok."
---

Oryginalne [repozytorium Facebook Research DETR](https://github.com/facebookresearch/detr) zostało zarchiwizowane 12 marca 2024 roku. LibreYOLO pozwala nadal korzystać z oryginalnego detektora przez to samo API co z innych rodzin modeli.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreDETRr50.pt")
result = model("image.jpg", save=True)
print(result.boxes.xyxy)
```

Dostępne są cztery warianty z ResNet-50 i ResNet-101 do predykcji, walidacji oraz eksportu do ONNX i TorchScript. Trenowanie nie jest zaimplementowane. LibreYOLO używa też stałego wejścia o rozmiarze 800 na 800 zamiast odtwarzać oryginalny pipeline ewaluacji zachowujący proporcje obrazu.

Wypróbuj po instalacji poleceniem `pip install libreyolo`. [Dokumentacja DETR](https://www.libreyolo.com/docs/models/detr) zawiera listę wszystkich czterech checkpointów. Porównanie z modelem CNN znajdziesz w artykule [Alternatywa dla EfficientDet](/articles/efficientdet-alternative).

[GitHub](https://github.com/LibreYOLO/libreyolo) | [Dokumentacja](https://www.libreyolo.com/docs)
