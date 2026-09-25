---
title: "Alternativa a Real-ESRGAN nel 2026: upscale delle immagini con LibreYOLO"
description: "Esegui l'upscaling delle immagini 2x e 4x di Real-ESRGAN con LibreYOLO, inclusa l'inferenza a tasselli per immagini grandi."
date: 2026-08-15
author: Xuban
tags: [LibreYOLO, real-esrgan-alternative, super-resolution, tutorial]
faq:
  - q: "Real-ESRGAN è abbandonato?"
    a: "Il repository non è archiviato, ma l'ultima release su GitHub è la versione 0.3.0, di settembre 2022."
  - q: "LibreYOLO può addestrare Real-ESRGAN?"
    a: "No. LibreYOLO supporta la predizione, la validazione e l'esportazione di Real-ESRGAN, ma non l'addestramento."
---

[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) non è formalmente archiviato, ma la sua ultima release su GitHub, la versione 0.3.0, risale a settembre 2022. LibreYOLO esegue i suoi checkpoint 2x e 4x tramite la normale API di predizione.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRealESRGANx4-restore.pt")
result = model("large-photo.jpg", tile=512, save=True)
print(result.restored.array.shape)
```

LibreYOLO supporta i checkpoint 2x, 4x e fast 4x, la predizione a tasselli, la validazione PSNR/SSIM e l'esportazione. L'addestramento non è implementato. Anche la fusione upstream basata sull'intensità di denoising non è inclusa in questo port.

Usa `pip install libreyolo` per provarlo. Trovi tutte le opzioni di ripristino nella [documentazione di Real-ESRGAN](https://www.libreyolo.com/docs/models/real-esrgan). Per approfondire, consulta l'[alternativa a MiDaS](/articles/midas-alternative).

[GitHub](https://github.com/LibreYOLO/libreyolo) | [Documentazione](https://www.libreyolo.com/docs)
