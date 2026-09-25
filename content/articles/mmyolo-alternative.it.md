---
title: "Alternativa a MMYOLO nel 2026: esegui RTMDet con LibreYOLO"
description: "L'ultima release di MMYOLO risale al 2023. Esegui detector supportati come RTMDet con l'API diretta di LibreYOLO."
date: 2026-08-15
author: Xuban
tags: [LibreYOLO, mmyolo-alternative, rtmdet, object-detection]
faq:
  - q: "MMYOLO è abbandonato?"
    a: "MMYOLO non è stato archiviato ufficialmente, ma l'ultima release su GitHub è la versione 0.6.0 di agosto 2023."
  - q: "LibreYOLO può caricare qualsiasi checkpoint di MMYOLO?"
    a: "No. LibreYOLO supporta famiglie di modelli specifiche e non garantisce il caricamento diretto di checkpoint MMYOLO arbitrari."
---

[MMYOLO](https://github.com/open-mmlab/mmyolo) non è stato archiviato, ma la sua ultima release, la versione 0.6.0, risale ad agosto 2023. Se ti serve solo un detector supportato come RTMDet, potresti non aver bisogno dell'intero stack OpenMMLab.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRTMDets.pt")
result = model("image.jpg", save=True)
print(result.boxes.xyxy)
```

LibreYOLO non è un sostituto completo di MMYOLO. Non riproduce ogni configurazione, ricetta o checkpoint. Offre però alle famiglie supportate, come RTMDet e YOLOX, un'API diretta per fare predizioni, addestrare, validare ed esportare.

Inizia con `pip install libreyolo`. La [documentazione di RTMDet](https://www.libreyolo.com/docs/models/rtmdet) elenca le operazioni supportate. La [guida rapida a RTMDet](/articles/rtmdet-without-mmdetection) spiega come migrare.

[GitHub](https://github.com/LibreYOLO/libreyolo) | [Documentazione](https://www.libreyolo.com/docs)
