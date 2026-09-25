---
title: "Alternative à YOLOX en 2026 : exécuter YOLOX avec LibreYOLO"
description: "La dernière version de YOLOX date de 2022. Exécutez, entraînez, validez et exportez YOLOX avec l'API LibreYOLO."
date: 2026-08-15
author: Xuban
tags: [LibreYOLO, yolox-alternative, object-detection, tutorial]
faq:
  - q: "YOLOX est-il abandonné ?"
    a: "Le dépôt officiel n'est pas archivé, mais sa dernière version, YOLOX 0.3.0, date d'avril 2022. Il est plus juste de parler d'un projet peu actif que d'un projet officiellement abandonné."
  - q: "LibreYOLO peut-il entraîner YOLOX ?"
    a: "Oui. LibreYOLO prend en charge la prédiction, l'entraînement, la validation et l'export de YOLOX."
---

YOLOX reste un détecteur utile sous licence Apache-2.0. Son [dépôt officiel](https://github.com/Megvii-BaseDetection/YOLOX) n'est pas archivé, mais sa dernière version, 0.3.0, date d'avril 2022. Le problème de maintenance est réel. Le modèle fonctionne toujours.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLOXs.pt")
result = model("image.jpg", save=True)
print(result.boxes.xyxy)
```

LibreYOLO prend en charge six tailles de YOLOX, de Nano à X, avec la prédiction, l'entraînement, la validation et l'export dans une seule API. Utilisez le projet d'origine si vous avez besoin de son workflow de configuration `Exp` exact.

Commencez par `pip install libreyolo`. La [documentation YOLOX](https://www.libreyolo.com/docs/models/yolox) couvre l'API complète. Vous trouverez aussi une courte note indiquant que [YOLO-NAS est toujours maintenu](/articles/yolo-nas-with-libreyolo).

[GitHub](https://github.com/LibreYOLO/libreyolo) | [Documentation](https://www.libreyolo.com/docs)
