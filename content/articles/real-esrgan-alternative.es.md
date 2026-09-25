---
title: "Alternativa a Real-ESRGAN en 2026: amplía imágenes con LibreYOLO"
description: "Ejecuta el escalado de imágenes 2x y 4x de Real-ESRGAN con LibreYOLO, incluida la inferencia por teselas para imágenes grandes."
date: 2026-08-15
author: Xuban
tags: [LibreYOLO, real-esrgan-alternative, super-resolution, tutorial]
faq:
  - q: "¿Está abandonado Real-ESRGAN?"
    a: "El repositorio no está archivado, pero su último lanzamiento en GitHub es la versión 0.3.0, de septiembre de 2022."
  - q: "¿Puede LibreYOLO entrenar Real-ESRGAN?"
    a: "No. LibreYOLO admite la predicción, la validación y la exportación de Real-ESRGAN, pero no el entrenamiento."
---

[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) no está archivado oficialmente, pero su último lanzamiento en GitHub, la versión 0.3.0, data de septiembre de 2022. LibreYOLO ejecuta sus checkpoints 2x y 4x mediante la API habitual de predicción.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRealESRGANx4-restore.pt")
result = model("large-photo.jpg", tile=512, save=True)
print(result.restored.array.shape)
```

LibreYOLO admite checkpoints 2x, 4x y fast 4x, predicción por teselas, validación con PSNR/SSIM y exportación. El entrenamiento no está implementado. La mezcla controlada por la intensidad de eliminación de ruido de la implementación original tampoco está incluida en este port.

Usa `pip install libreyolo` para probarlo. Encontrarás todas las opciones de restauración en la [documentación de Real-ESRGAN](https://www.libreyolo.com/docs/models/real-esrgan). Para profundizar, consulta la [alternativa a MiDaS](/articles/midas-alternative).

[GitHub](https://github.com/LibreYOLO/libreyolo) | [Documentación](https://www.libreyolo.com/docs)
