---
title: "Alternative à Facebook DETR en 2026 : utiliser DETR avec LibreYOLO"
description: "Le dépôt original Facebook Research DETR est archivé. Exécutez quatre variantes officielles de DETR avec LibreYOLO."
date: 2026-08-15
author: Xuban
tags: [LibreYOLO, facebook-detr-alternative, object-detection, transformer]
faq:
  - q: "Facebook DETR est-il abandonné ?"
    a: "Le dépôt original facebookresearch/detr a été archivé le 12 mars 2024."
  - q: "LibreYOLO peut-il entraîner le DETR original ?"
    a: "Non. LibreYOLO prend en charge la prédiction, la validation et l'export de DETR, mais pas sa recette d'entraînement sur 500 époques."
---

Le [dépôt Facebook Research DETR](https://github.com/facebookresearch/detr) original a été archivé le 12 mars 2024. LibreYOLO permet de continuer à utiliser le détecteur d'origine avec la même API que ses autres familles de modèles.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreDETRr50.pt")
result = model("image.jpg", save=True)
print(result.boxes.xyxy)
```

Quatre variantes basées sur ResNet-50 et ResNet-101 sont disponibles pour la prédiction, la validation et l'export ONNX et TorchScript. L'entraînement n'est pas implémenté. LibreYOLO utilise également une entrée fixe de 800 par 800, au lieu de reproduire le pipeline d'évaluation d'origine qui préserve le rapport hauteur/largeur.

Essayez-le après `pip install libreyolo`. La [documentation DETR](https://www.libreyolo.com/docs/models/detr) répertorie les quatre checkpoints. Pour une comparaison avec un CNN, consultez l'[alternative à EfficientDet](/articles/efficientdet-alternative).

[GitHub](https://github.com/LibreYOLO/libreyolo) | [Documentation](https://www.libreyolo.com/docs)
