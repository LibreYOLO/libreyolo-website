---
title: "MMYOLO-Alternative 2026: RTMDet mit LibreYOLO ausführen"
description: "Das letzte MMYOLO-Release stammt aus dem Jahr 2023. Führe unterstützte Detektoren wie RTMDet direkt über die LibreYOLO-API aus."
date: 2026-08-15
author: Xuban
tags: [LibreYOLO, mmyolo-alternative, rtmdet, object-detection]
faq:
  - q: "Wird MMYOLO nicht mehr weiterentwickelt?"
    a: "MMYOLO wurde nicht offiziell archiviert, aber das letzte GitHub-Release ist Version 0.6.0 vom August 2023."
  - q: "Kann LibreYOLO beliebige MMYOLO-Checkpoints laden?"
    a: "Nein. LibreYOLO unterstützt benannte Modellfamilien und verspricht nicht, dass beliebige MMYOLO-Checkpoints direkt geladen werden können."
---

[MMYOLO](https://github.com/open-mmlab/mmyolo) wurde nicht archiviert, aber sein letztes Release, Version 0.6.0, stammt aus dem August 2023. Wenn du nur einen unterstützten Detektor wie RTMDet brauchst, benötigst du möglicherweise nicht den vollständigen OpenMMLab-Stack.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRTMDets.pt")
result = model("image.jpg", save=True)
print(result.boxes.xyxy)
```

LibreYOLO ist kein vollständiger Ersatz für MMYOLO. Es bildet nicht jede Konfiguration, jedes Trainingsrezept und jeden Checkpoint nach. Unterstützte Modellfamilien wie RTMDet und YOLOX erhalten aber eine direkte API für Vorhersage, Training, Validierung und Export.

Starte mit `pip install libreyolo`. Die [RTMDet-Doku](https://www.libreyolo.com/docs/models/rtmdet) listet die unterstützten Funktionen auf. Der [kurze RTMDet-Leitfaden](/articles/rtmdet-without-mmdetection) beschreibt die Migration.

[GitHub](https://github.com/LibreYOLO/libreyolo) | [Dokumentation](https://www.libreyolo.com/docs)
