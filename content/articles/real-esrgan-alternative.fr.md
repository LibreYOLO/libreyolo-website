---
title: "Alternative à Real-ESRGAN en 2026 : agrandissez vos images avec LibreYOLO"
description: "Effectuez l'upscaling d'images avec Real-ESRGAN en 2x et 4x via LibreYOLO, y compris l'inférence par tuiles pour les grandes images."
date: 2026-08-15
author: Xuban
tags: [LibreYOLO, real-esrgan-alternative, super-resolution, tutorial]
faq:
  - q: "Real-ESRGAN est-il abandonné ?"
    a: "Le dépôt n'est pas archivé, mais sa dernière version sur GitHub est la 0.3.0, publiée en septembre 2022."
  - q: "LibreYOLO peut-il entraîner Real-ESRGAN ?"
    a: "Non. LibreYOLO prend en charge la prédiction, la validation et l'export de Real-ESRGAN, mais pas son entraînement."
---

[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) n'est pas officiellement archivé, mais sa dernière version sur GitHub, la 0.3.0, remonte à septembre 2022. LibreYOLO exécute ses checkpoints 2x et 4x via l'API de prédiction habituelle.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRealESRGANx4-restore.pt")
result = model("large-photo.jpg", tile=512, save=True)
print(result.restored.array.shape)
```

LibreYOLO prend en charge les checkpoints 2x, 4x et fast 4x, avec la prédiction par tuiles, la validation PSNR/SSIM et l'export. L'entraînement n'est pas implémenté. Le mélange de débruitage avec intensité réglable de la version amont n'est pas non plus pris en charge par ce portage.

Installez-le avec `pip install libreyolo` pour l'essayer. Vous trouverez toutes les options de restauration dans la [documentation Real-ESRGAN](https://www.libreyolo.com/docs/models/real-esrgan). Pour l'estimation de profondeur, consultez l'[alternative à MiDaS](/articles/midas-alternative).

[GitHub](https://github.com/LibreYOLO/libreyolo) | [Documentation](https://www.libreyolo.com/docs)
