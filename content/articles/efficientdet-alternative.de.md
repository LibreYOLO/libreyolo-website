---
title: "EfficientDet-Alternative 2026: D0-D4 mit LibreYOLO ausführen"
description: "EfficientDet D0-D4 mit der Objekterkennungs-API von LibreYOLO für Vorhersagen und Validierung ausführen und exportieren."
date: 2026-08-15
author: Xuban
tags: [LibreYOLO, efficientdet-alternative, object-detection, tutorial]
faq:
  - q: "Wird EfficientDet nicht mehr weiterentwickelt?"
    a: "Das PyTorch-Repository von rwightman wurde nicht offiziell archiviert. Es ist treffender, es als wenig aktiv statt als aufgegeben zu bezeichnen."
  - q: "Kann LibreYOLO EfficientDet trainieren?"
    a: "Nein. LibreYOLO unterstützt Inferenz, Validierung und Export für EfficientDet, aber kein Training."
---

Das weit verbreitete [EfficientDet-PyTorch-Repository](https://github.com/rwightman/efficientdet-pytorch) wurde nicht archiviert. Es wäre daher falsch, es als aufgegeben zu bezeichnen. Es ist wenig aktiv. LibreYOLO bietet einen weiteren Weg für das Deployment.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreEfficientDetd0.pt")
result = model("image.jpg", save=True)
print(result.boxes.xyxy)
```

LibreYOLO unterstützt EfficientDet D0 bis D4 für Vorhersagen und Validierung. ONNX-, TorchScript-, OpenVINO- und TensorRT-Exporte sind auf Parität geprüft. Training ist nicht implementiert, und jede Modellgröße behält ihre feste native Eingabeauflösung.

Installiere die Bibliothek mit `pip install libreyolo`. Informationen zu jeder Modellgröße findest du in der [EfficientDet-Dokumentation](https://www.libreyolo.com/docs/models/efficientdet). Oder vergleiche EfficientDet mit der [Facebook-DETR-Alternative](/articles/facebook-detr-alternative).

[GitHub](https://github.com/LibreYOLO/libreyolo) | [Dokumentation](https://www.libreyolo.com/docs)
