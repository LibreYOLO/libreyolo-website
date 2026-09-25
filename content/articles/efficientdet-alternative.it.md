---
title: "Alternativa a EfficientDet nel 2026: esegui D0-D4 con LibreYOLO"
description: "Esegui predizioni, validazione ed esportazione di EfficientDet D0-D4 tramite l'API di LibreYOLO per il rilevamento di oggetti."
date: 2026-08-15
author: Xuban
tags: [LibreYOLO, efficientdet-alternative, object-detection, tutorial]
faq:
  - q: "EfficientDet è stato abbandonato?"
    a: "Il repository PyTorch di rwightman non è formalmente archiviato. È più corretto dire che è poco attivo, non abbandonato."
  - q: "LibreYOLO può addestrare EfficientDet?"
    a: "No. LibreYOLO supporta l'inferenza, la validazione e l'esportazione di EfficientDet, ma non l'addestramento."
---

Il diffuso [repository PyTorch di EfficientDet](https://github.com/rwightman/efficientdet-pytorch) non è archiviato, quindi sarebbe sbagliato definirlo abbandonato. È poco attivo. LibreYOLO offre un altro percorso per il deployment.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreEfficientDetd0.pt")
result = model("image.jpg", save=True)
print(result.boxes.xyxy)
```

LibreYOLO supporta EfficientDet da D0 a D4 per le predizioni e la validazione. Le esportazioni in ONNX, TorchScript, OpenVINO e TensorRT hanno una copertura di parità. L'addestramento non è implementato e ogni dimensione mantiene la propria risoluzione di input nativa fissa.

Installa con `pip install libreyolo`. Consulta la [documentazione di EfficientDet](https://www.libreyolo.com/docs/models/efficientdet) per ogni dimensione, oppure confrontalo con l'[alternativa a Facebook DETR](/articles/facebook-detr-alternative).

[GitHub](https://github.com/LibreYOLO/libreyolo) | [Documentazione](https://www.libreyolo.com/docs)
