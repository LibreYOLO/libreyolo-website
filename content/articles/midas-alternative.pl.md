---
title: "Alternatywa dla MiDaS w 2026: estymacja głębi w LibreYOLO"
description: "Oficjalne repozytorium MiDaS zostało zarchiwizowane. Uruchamiaj modele Small i DPT-Large do estymacji głębi przez LibreYOLO."
date: 2026-08-15
author: Xuban
tags: [LibreYOLO, midas-alternative, depth-estimation, tutorial]
faq:
  - q: "Czy MiDaS zostało porzucone?"
    a: "Oficjalne repozytorium isl-org/MiDaS zostało zarchiwizowane 25 sierpnia 2025 roku."
  - q: "Czy MiDaS zwraca głębię metryczną?"
    a: "Nie. MiDaS zwraca względną odwrotną głębię bez jednostki metrycznej i bez stałej skali między obrazami."
---

[Oficjalne repozytorium MiDaS](https://github.com/isl-org/MiDaS) zostało zarchiwizowane 25 sierpnia 2025 roku. LibreYOLO zapewnia utrzymywaną metodę uruchamiania checkpointów Small i DPT-Large.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreMiDaSl-depth.pt")
result = model("image.jpg", save=True)
print(result.depth_map.data.shape)
```

Obsługiwane są predykcja, walidacja zero-shot i eksport do stałej rozdzielczości. Trenowanie nie jest obsługiwane. Wynikiem jest względna odwrotna głębia: wyższe wartości oznaczają bliższe obiekty, ale nie ma jednostki metrycznej ani stałej skali między obrazami.

Wystarczy podstawowa instalacja: `pip install libreyolo`. Dalej można przejść do [dokumentacji MiDaS](https://www.libreyolo.com/docs/models/midas) lub artykułu o [alternatywie dla Real-ESRGAN](/articles/real-esrgan-alternative).

[GitHub](https://github.com/LibreYOLO/libreyolo) | [Dokumentacja](https://www.libreyolo.com/docs)
