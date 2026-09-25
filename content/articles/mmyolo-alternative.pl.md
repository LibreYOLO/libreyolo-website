---
title: "Alternatywa dla MMYOLO w 2026: RTMDet w LibreYOLO"
description: "Najnowsze wydanie MMYOLO pochodzi z 2023 roku. Uruchom obsługiwane detektory, takie jak RTMDet, bezpośrednio przez API LibreYOLO."
date: 2026-08-15
author: Xuban
tags: [LibreYOLO, mmyolo-alternative, rtmdet, object-detection]
faq:
  - q: "Czy MMYOLO zostało porzucone?"
    a: "MMYOLO nie zostało formalnie zarchiwizowane, ale jego najnowsze wydanie w serwisie GitHub to wersja 0.6.0 z sierpnia 2023 roku."
  - q: "Czy LibreYOLO może wczytać dowolny checkpoint MMYOLO?"
    a: "Nie. LibreYOLO obsługuje nazwane rodziny modeli i nie gwarantuje bezpośredniego wczytywania dowolnych checkpointów MMYOLO."
---

[MMYOLO](https://github.com/open-mmlab/mmyolo) nie zostało zarchiwizowane, ale jego najnowsze wydanie, wersja 0.6.0, pochodzi z sierpnia 2023 roku. Jeśli potrzebny jest tylko obsługiwany detektor, taki jak RTMDet, pełny stos OpenMMLab może okazać się zbędny.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRTMDets.pt")
result = model("image.jpg", save=True)
print(result.boxes.xyxy)
```

LibreYOLO nie jest pełnym zamiennikiem MMYOLO. Nie odtwarza każdej konfiguracji, receptury ani checkpointu. Udostępnia za to obsługiwanym rodzinom, takim jak RTMDet i YOLOX, bezpośrednie API do predykcji, trenowania, walidacji i eksportu.

Zacznij od `pip install libreyolo`. [Dokumentacja RTMDet](https://www.libreyolo.com/docs/models/rtmdet) zawiera listę obsługiwanych operacji. [Krótki przewodnik po RTMDet](/articles/rtmdet-without-mmdetection) opisuje migrację.

[GitHub](https://github.com/LibreYOLO/libreyolo) | [Dokumentacja](https://www.libreyolo.com/docs)
