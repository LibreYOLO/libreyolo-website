---
title: "Real-ESRGAN-Alternative 2026: Bilder mit LibreYOLO hochskalieren"
description: "Führe das 2x- und 4x-Upscaling von Real-ESRGAN mit LibreYOLO aus, einschließlich gekachelter Inferenz für große Bilder."
date: 2026-08-15
author: Xuban
tags: [LibreYOLO, real-esrgan-alternative, super-resolution, tutorial]
faq:
  - q: "Wird Real-ESRGAN nicht mehr weiterentwickelt?"
    a: "Das Repository ist nicht archiviert, aber das letzte GitHub-Release ist Version 0.3.0 vom September 2022."
  - q: "Kann LibreYOLO Real-ESRGAN trainieren?"
    a: "Nein. LibreYOLO unterstützt für Real-ESRGAN Vorhersage, Validierung und Export, aber kein Training."
---

[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) ist nicht offiziell archiviert, aber sein letztes GitHub-Release, Version 0.3.0, stammt aus dem September 2022. LibreYOLO führt seine 2x- und 4x-Checkpoints über die reguläre Vorhersage-API aus.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRealESRGANx4-restore.pt")
result = model("large-photo.jpg", tile=512, save=True)
print(result.restored.array.shape)
```

LibreYOLO unterstützt 2x-, 4x- und schnelle 4x-Checkpoints sowie gekachelte Vorhersage, PSNR/SSIM-Validierung und Export. Training ist nicht implementiert. Auch die Upstream-Mischung mit einstellbarer Entrauschungsstärke ist in diesem Port nicht enthalten.

Installiere es mit `pip install libreyolo`. Alle Optionen für die Wiederherstellung findest du in der [Real-ESRGAN-Dokumentation](https://www.libreyolo.com/docs/models/real-esrgan). Für Tiefenschätzung siehe die [MiDaS-Alternative](/articles/midas-alternative).

[GitHub](https://github.com/LibreYOLO/libreyolo) | [Dokumentation](https://www.libreyolo.com/docs)
