---
title: "Alternative à EfficientDet en 2026 : exécuter D0-D4 avec LibreYOLO"
description: "Exécutez la prédiction, la validation et l'export d'EfficientDet D0-D4 avec l'API de détection d'objets de LibreYOLO."
date: 2026-08-15
author: Xuban
tags: [LibreYOLO, efficientdet-alternative, object-detection, tutorial]
faq:
  - q: "EfficientDet est-il abandonné ?"
    a: "Le dépôt PyTorch de rwightman n'est pas officiellement archivé. Il est plus juste de le qualifier d'inactif que d'abandonné."
  - q: "LibreYOLO peut-il entraîner EfficientDet ?"
    a: "Non. LibreYOLO prend en charge l'inférence, la validation et l'export d'EfficientDet, mais pas son entraînement."
---

Le [dépôt PyTorch EfficientDet, largement utilisé](https://github.com/rwightman/efficientdet-pytorch) n'est pas archivé : il serait donc faux de le qualifier d'abandonné. Il est inactif. LibreYOLO propose une autre voie de déploiement.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreEfficientDetd0.pt")
result = model("image.jpg", save=True)
print(result.boxes.xyxy)
```

LibreYOLO prend en charge EfficientDet D0 à D4 pour la prédiction et la validation. Les exports ONNX, TorchScript, OpenVINO et TensorRT bénéficient d'une couverture de parité. L'entraînement n'est pas implémenté, et chaque taille conserve sa résolution d'entrée native fixe.

Installez le paquet avec `pip install libreyolo`. Consultez la [documentation EfficientDet](https://www.libreyolo.com/docs/models/efficientdet) pour chaque taille, ou comparez-le à l'[alternative à Facebook DETR](/articles/facebook-detr-alternative).

[GitHub](https://github.com/LibreYOLO/libreyolo) | [Documentation](https://www.libreyolo.com/docs)
