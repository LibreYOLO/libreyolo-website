---
title: "Alternativa a Facebook DETR nel 2026: eseguilo con LibreYOLO"
description: "Il repository originale Facebook Research DETR è archiviato. Esegui quattro varianti ufficiali di DETR con LibreYOLO."
date: 2026-08-15
author: Xuban
tags: [LibreYOLO, facebook-detr-alternative, object-detection, transformer]
faq:
  - q: "Facebook DETR è abbandonato?"
    a: "Il repository originale facebookresearch/detr è stato archiviato il March 12, 2024."
  - q: "LibreYOLO può addestrare il DETR originale?"
    a: "No. LibreYOLO supporta la predizione, la validazione e l'esportazione di DETR, ma non la sua ricetta di addestramento di 500 epoche."
---

Il [repository Facebook Research DETR](https://github.com/facebookresearch/detr) originale è stato archiviato il March 12, 2024. LibreYOLO mantiene utilizzabile il detector originale tramite la stessa API delle altre famiglie di modelli.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreDETRr50.pt")
result = model("image.jpg", save=True)
print(result.boxes.xyxy)
```

Sono disponibili quattro varianti ResNet-50 e ResNet-101 per la predizione, la validazione e l'esportazione in ONNX e TorchScript. L'addestramento non è implementato. LibreYOLO usa inoltre un input fisso di 800 per 800, anziché riprodurre la pipeline di valutazione originale che conserva le proporzioni.

Provalo dopo `pip install libreyolo`. La [documentazione di DETR](https://www.libreyolo.com/docs/models/detr) elenca tutti e quattro i checkpoint. Per un confronto con una CNN, consulta l'[alternativa a EfficientDet](/articles/efficientdet-alternative).

[GitHub](https://github.com/LibreYOLO/libreyolo) | [Documentazione](https://www.libreyolo.com/docs)
