---
title: "YOLO-NAS wird weiter gepflegt: Vorhersage, Training und Export mit LibreYOLO"
description: "Führe YOLO-NAS mit LibreYOLO aus, trainiere, validiere und exportiere Modelle. Das letzte SuperGradients-Release erschien im April 2024. Wir planen, neue Gewichte von Grund auf neu zu trainieren, die nicht Decis Lizenzbedingungen unterliegen."
date: 2026-08-20
author: Xuban
tags: [LibreYOLO, yolo-nas, yolo-nas-maintained, object-detection]
faq:
  - q: "Wird YOLO-NAS nicht mehr weiterentwickelt?"
    a: "SuperGradients veröffentlichte zuletzt im April 2024 die Version 3.7.1. LibreYOLO unterstützt Vorhersage, Training, Validierung und Export mit YOLO-NAS auf aktuellen PyTorch-Versionen."
  - q: "Pflegt Ultralytics YOLO-NAS?"
    a: "Ultralytics unterstützt Inferenz, Validierung und Export. Training wird nicht unterstützt."
  - q: "Dürfen vortrainierte YOLO-NAS-Gewichte kommerziell genutzt werden?"
    a: "Für Decis vortrainierte Gewichte gelten weiterhin die nicht-kommerziellen Bedingungen des Upstream-Projekts, unabhängig davon, welche Bibliothek sie lädt. Beim Training mit zufälliger Initialisierung werden diese Checkpoints nicht verwendet. Wir planen außerdem, YOLO-NAS-Gewichte für COCO von Grund auf neu zu trainieren und zu veröffentlichen."
---

YOLO-NAS ist weiterhin ein nützlicher Detektor. Das ursprüngliche Zuhause des Modells, [SuperGradients](https://github.com/Deci-AI/super-gradients), veröffentlichte zuletzt im April 2024 die Version 3.7.1. LibreYOLO pflegt die Unterstützung für Vorhersage, Training, Validierung und Export.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLONASs.pt")
result = model("image.jpg", save=True)
print(result.boxes.xyxy)
```

LibreYOLO unterstützt S, M und L für die Objekterkennung sowie Pose-Schätzung. Checkpoints werden unter Namen wie `LibreYOLONASs.pt` von Decis öffentlichem CDN heruntergeladen. LibreYOLO hostet sie nicht, und Decis nicht-kommerzielle Bedingungen gelten weiterhin. SuperGradients legt PyTorch auf Versionen zwischen 1.9 und 1.13 fest; LibreYOLO erfordert Version 2.4 oder neuer. Der Export für die Objekterkennung unterstützt ONNX, TorchScript, OpenVINO, NCNN, TFLite, Paddle, MNN, ExecuTorch und Core AI. CoreML wird nicht unterstützt.

So trainierst du ohne einen Deci-Checkpoint:

```python
from libreyolo import LibreYOLONAS

model = LibreYOLONAS(None, size="s")
model.train(data="my-dataset.yaml", imgsz=640, batch=16)
```

Wir planen außerdem, YOLO-NAS von Grund auf neu mit COCO zu trainieren und diese Gewichte zu veröffentlichen.

Installiere die Bibliothek mit `pip install libreyolo`. In der [YOLO-NAS-Dokumentation](https://www.libreyolo.com/docs/models/yolo-nas) findest du die Modelldetails. Außerdem gibt es einen ausführlicheren Artikel zur [SuperGradients-Alternative](/articles/supergradients-alternative).

[GitHub](https://github.com/LibreYOLO/libreyolo) | [Dokumentation](https://www.libreyolo.com/docs)
