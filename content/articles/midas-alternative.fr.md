---
title: "Alternative à MiDaS en 2026 : estimation de profondeur avec LibreYOLO"
description: "Le dépôt officiel de MiDaS est archivé. Exécutez ses modèles de profondeur Small et DPT-Large avec LibreYOLO."
date: 2026-08-15
author: Xuban
tags: [LibreYOLO, midas-alternative, depth-estimation, tutorial]
faq:
  - q: "MiDaS est-il abandonné ?"
    a: "Le dépôt officiel isl-org/MiDaS a été archivé le 25 août 2025."
  - q: "MiDaS renvoie-t-il une profondeur métrique ?"
    a: "Non. MiDaS renvoie une profondeur inverse relative, sans unité métrique ni échelle fixe entre les images."
---

Le [dépôt officiel de MiDaS](https://github.com/isl-org/MiDaS) a été archivé le 25 août 2025. LibreYOLO offre une solution maintenue pour exécuter ses checkpoints Small et DPT-Large.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreMiDaSl-depth.pt")
result = model("image.jpg", save=True)
print(result.depth_map.data.shape)
```

La prédiction, la validation zero-shot et l'export à résolution fixe sont pris en charge. L'entraînement ne l'est pas. La sortie est une profondeur inverse relative : plus la valeur est élevée, plus l'objet correspondant est proche, mais il n'y a ni unité métrique ni échelle fixe entre les images.

L'installation de base suffit : `pip install libreyolo`. Consultez ensuite la [documentation MiDaS](https://www.libreyolo.com/docs/models/midas) ou l'[alternative à Real-ESRGAN](/articles/real-esrgan-alternative).

[GitHub](https://github.com/LibreYOLO/libreyolo) | [Documentation](https://www.libreyolo.com/docs)
