---
title: "Alternativa a YOLOX nel 2026: usalo con LibreYOLO"
description: "L'ultima release di YOLOX risale al 2022. Esegui, addestra, valida ed esporta YOLOX con l'API di LibreYOLO."
date: 2026-08-15
author: Xuban
tags: [LibreYOLO, yolox-alternative, object-detection, tutorial]
faq:
  - q: "YOLOX è abbandonato?"
    a: "Il repository ufficiale non è archiviato, ma la più recente, YOLOX 0.3.0, risale ad aprile 2022. È più corretto definirlo inattivo che formalmente abbandonato."
  - q: "LibreYOLO può addestrare YOLOX?"
    a: "Sì. LibreYOLO supporta YOLOX per fare predizioni, addestrare, validare ed esportare."
---

YOLOX è ancora un utile modello di rilevamento con licenza Apache-2.0. Il suo [repository ufficiale](https://github.com/Megvii-BaseDetection/YOLOX) non è archiviato, ma l'ultima release, la 0.3.0, risale ad aprile 2022. Il problema della manutenzione è reale. Il modello funziona ancora.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLOXs.pt")
result = model("image.jpg", save=True)
print(result.boxes.xyxy)
```

LibreYOLO supporta sei dimensioni di YOLOX, da Nano a X, e offre predizione, addestramento, validazione ed esportazione tramite un'unica API. Usa il progetto originale se ti serve il suo workflow di configurazione `Exp`.

Inizia con `pip install libreyolo`. La [documentazione di YOLOX](https://www.libreyolo.com/docs/models/yolox) descrive l'API completa. C'è anche una breve nota sul fatto che [YOLO-NAS è ancora mantenuto](/articles/yolo-nas-with-libreyolo).

[GitHub](https://github.com/LibreYOLO/libreyolo) | [Documentazione](https://www.libreyolo.com/docs)
