---
title: "Alternative à MMYOLO en 2026 : utiliser RTMDet avec LibreYOLO"
description: "La dernière version de MMYOLO date de 2023. Exécutez des détecteurs pris en charge comme RTMDet via l'API directe de LibreYOLO."
date: 2026-08-15
author: Xuban
tags: [LibreYOLO, mmyolo-alternative, rtmdet, object-detection]
faq:
  - q: "MMYOLO est-il abandonné ?"
    a: "MMYOLO n'est pas officiellement archivé, mais sa dernière version sur GitHub est la 0.6.0, publiée en août 2023."
  - q: "LibreYOLO peut-il charger n'importe quel checkpoint MMYOLO ?"
    a: "Non. LibreYOLO prend en charge des familles de modèles nommées et ne garantit pas le chargement direct de checkpoints MMYOLO arbitraires."
---

[MMYOLO](https://github.com/open-mmlab/mmyolo) n'est pas archivé, mais sa dernière version, la 0.6.0, date d'août 2023. Si vous avez seulement besoin d'un détecteur pris en charge comme RTMDet, toute la pile OpenMMLab ne vous est peut-être pas nécessaire.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRTMDets.pt")
result = model("image.jpg", save=True)
print(result.boxes.xyxy)
```

LibreYOLO ne remplace pas complètement MMYOLO. Il ne reproduit pas toutes les configs, recettes ou checkpoints. Il fournit toutefois aux familles prises en charge, comme RTMDet et YOLOX, une API directe pour la prédiction, l'entraînement, la validation et l'export.

Commencez par `pip install libreyolo`. La [documentation RTMDet](https://www.libreyolo.com/docs/models/rtmdet) répertorie les opérations prises en charge. Le [guide rapide RTMDet](/articles/rtmdet-without-mmdetection) explique la migration.

[GitHub](https://github.com/LibreYOLO/libreyolo) | [Documentation](https://www.libreyolo.com/docs)
