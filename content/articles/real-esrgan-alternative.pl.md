---
title: "Alternatywa dla Real-ESRGAN w 2026: powiększanie obrazów z LibreYOLO"
description: "Wykonuj 2-krotne i 4-krotne powiększanie obrazów Real-ESRGAN przez LibreYOLO, także z inferencją kafelkową dla dużych obrazów."
date: 2026-08-15
author: Xuban
tags: [LibreYOLO, real-esrgan-alternative, super-resolution, tutorial]
faq:
  - q: "Czy Real-ESRGAN został porzucony?"
    a: "Repozytorium nie zostało zarchiwizowane, ale jego najnowsze wydanie w GitHubie to wersja 0.3.0 z września 2022 roku."
  - q: "Czy LibreYOLO umożliwia trenowanie Real-ESRGAN?"
    a: "Nie. LibreYOLO obsługuje predykcję, walidację i eksport Real-ESRGAN, ale nie trenowanie."
---

[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) nie został formalnie zarchiwizowany, ale jego najnowsze wydanie w GitHubie, wersja 0.3.0, pochodzi z września 2022 roku. LibreYOLO uruchamia checkpointy 2x i 4x przez standardowe API predykcji.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRealESRGANx4-restore.pt")
result = model("large-photo.jpg", tile=512, save=True)
print(result.restored.array.shape)
```

LibreYOLO obsługuje checkpointy 2x, 4x i fast 4x, predykcję kafelkową, walidację PSNR/SSIM oraz eksport. Trenowanie nie jest zaimplementowane. Funkcja łączenia z regulacją siły odszumiania z projektu upstreamowego również nie jest dostępna w tym porcie.

Aby wypróbować, użyj `pip install libreyolo`. Pełny zestaw opcji odtwarzania obrazu opisano w [dokumentacji Real-ESRGAN](https://www.libreyolo.com/docs/models/real-esrgan). Więcej informacji zawiera artykuł [Alternatywa dla MiDaS](/articles/midas-alternative).

[GitHub](https://github.com/LibreYOLO/libreyolo) | [Dokumentacja](https://www.libreyolo.com/docs)
