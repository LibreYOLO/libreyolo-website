---
title: "Facebook DETR Alternative 2026: Mit LibreYOLO ausführen"
description: "Das ursprüngliche Facebook-Research-DETR-Repository ist archiviert. Führe vier offizielle DETR-Varianten mit LibreYOLO aus."
date: 2026-08-15
author: Xuban
tags: [LibreYOLO, facebook-detr-alternative, object-detection, transformer]
faq:
  - q: "Ist Facebook DETR aufgegeben worden?"
    a: "Das ursprüngliche Repository facebookresearch/detr wurde am March 12, 2024 archiviert."
  - q: "Kann LibreYOLO das ursprüngliche DETR trainieren?"
    a: "Nein. LibreYOLO unterstützt Vorhersage, Validierung und Export für DETR, aber nicht dessen 500-Epochen-Trainingsrezept."
---

Das ursprüngliche [Facebook-Research-DETR-Repository](https://github.com/facebookresearch/detr) wurde am March 12, 2024 archiviert. LibreYOLO hält den ursprünglichen Detektor über dieselbe API wie seine anderen Modellfamilien nutzbar.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreDETRr50.pt")
result = model("image.jpg", save=True)
print(result.boxes.xyxy)
```

Vier Varianten mit ResNet-50 und ResNet-101 stehen für Vorhersage, Validierung sowie den Export nach ONNX und TorchScript zur Verfügung. Training ist nicht implementiert. LibreYOLO verwendet außerdem eine feste Eingabegröße von 800 × 800, statt die ursprüngliche Evaluationspipeline mit beibehaltenem Seitenverhältnis nachzubilden.

Probiere es nach `pip install libreyolo` aus. Die [DETR-Dokumentation](https://www.libreyolo.com/docs/models/detr) führt alle vier Checkpoints auf. Einen Vergleich mit einem CNN findest du unter [EfficientDet-Alternative](/articles/efficientdet-alternative).

[GitHub](https://github.com/LibreYOLO/libreyolo) | [Dokumentation](https://www.libreyolo.com/docs)
