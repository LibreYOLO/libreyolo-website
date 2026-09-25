---
title: "Alternativa a MiDaS nel 2026: stima della profondità con LibreYOLO"
description: "Il repository ufficiale di MiDaS è archiviato. Esegui i suoi modelli di profondità Small e DPT-Large con LibreYOLO."
date: 2026-08-15
author: Xuban
tags: [LibreYOLO, midas-alternative, depth-estimation, tutorial]
faq:
  - q: "MiDaS è abbandonato?"
    a: "Il repository ufficiale isl-org/MiDaS è stato archiviato il 25 agosto 2025."
  - q: "MiDaS restituisce una profondità metrica?"
    a: "No. MiDaS restituisce una profondità inversa relativa, senza unità metriche né una scala fissa tra immagini."
---

Il [repository ufficiale di MiDaS](https://github.com/isl-org/MiDaS) è stato archiviato il 25 agosto 2025. LibreYOLO offre un modo mantenuto per eseguire i suoi checkpoint Small e DPT-Large.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreMiDaSl-depth.pt")
result = model("image.jpg", save=True)
print(result.depth_map.data.shape)
```

Sono supportate la predizione, la validazione zero-shot e l'esportazione a risoluzione fissa. L'addestramento non è supportato. L'output è una profondità inversa relativa: valori più alti indicano elementi più vicini, ma non esistono un'unità metrica né una scala fissa tra immagini.

È sufficiente l'installazione di base: `pip install libreyolo`. Continua con la [documentazione di MiDaS](https://www.libreyolo.com/docs/models/midas) o con l'[alternativa a Real-ESRGAN](/articles/real-esrgan-alternative).

[GitHub](https://github.com/LibreYOLO/libreyolo) | [Documentazione](https://www.libreyolo.com/docs)
