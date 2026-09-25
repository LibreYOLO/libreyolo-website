---
title: "YOLOX-Alternative: YOLOX 0.3.0 mit LibreYOLO ausführen"
description: "Das neueste YOLOX-Release stammt aus dem Jahr 2022. Führe YOLOX mit der LibreYOLO-API aus, trainiere und validiere es und exportiere es."
date: 2026-08-15
author: Xuban
tags: [LibreYOLO, yolox-alternative, object-detection, tutorial]
faq:
  - q: "Wird YOLOX nicht mehr weiterentwickelt?"
    a: "Das offizielle Repository ist nicht archiviert, aber das neueste Release ist YOLOX 0.3.0 vom April 2022. „Wenig aktiv“ trifft es besser als „offiziell aufgegeben“."
  - q: "Kann LibreYOLO YOLOX trainieren?"
    a: "Ja. LibreYOLO unterstützt Vorhersagen, Training, Validierung und Export mit YOLOX."
---

YOLOX ist weiterhin ein nützlicher Detektor unter Apache-2.0. Das [offizielle Repository](https://github.com/Megvii-BaseDetection/YOLOX) ist nicht archiviert, aber das neueste Release, 0.3.0, stammt aus dem April 2022. Das Wartungsproblem ist real. Das Modell funktioniert weiterhin.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLOXs.pt")
result = model("image.jpg", save=True)
print(result.boxes.xyxy)
```

LibreYOLO unterstützt sechs YOLOX-Größen von Nano bis X. Vorhersage, Training, Validierung und Export laufen über eine gemeinsame API. Verwende das Originalprojekt, wenn du den exakten `Exp`-Konfigurationsworkflow brauchst.

Starte mit `pip install libreyolo`. Die [YOLOX-Doku](https://www.libreyolo.com/docs/models/yolox) beschreibt die vollständige API. Außerdem gibt es einen kurzen Hinweis dazu, dass [YOLO-NAS weiterhin gepflegt wird](/articles/yolo-nas-with-libreyolo).

[GitHub](https://github.com/LibreYOLO/libreyolo) | [Dokumentation](https://www.libreyolo.com/docs)
