---
title: "Alternativa a Facebook DETR en 2026: ejecútalo con LibreYOLO"
description: "El repositorio original de Facebook Research DETR está archivado. Ejecuta cuatro variantes oficiales de DETR con LibreYOLO."
date: 2026-08-15
author: Xuban
tags: [LibreYOLO, facebook-detr-alternative, object-detection, transformer]
faq:
  - q: "¿Está abandonado Facebook DETR?"
    a: "El repositorio original facebookresearch/detr se archivó el 12 de marzo de 2024."
  - q: "¿Puede LibreYOLO entrenar el DETR original?"
    a: "No. LibreYOLO admite la predicción, validación y exportación de DETR, pero no su receta de entrenamiento de 500 épocas."
---

El [repositorio original de Facebook Research DETR](https://github.com/facebookresearch/detr) se archivó el 12 de marzo de 2024. LibreYOLO permite seguir usando el detector original mediante la misma API que sus otras familias de modelos.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreDETRr50.pt")
result = model("image.jpg", save=True)
print(result.boxes.xyxy)
```

Hay cuatro variantes ResNet-50 y ResNet-101 disponibles para predicción, validación y exportación a ONNX y TorchScript. El entrenamiento no está implementado. LibreYOLO también usa una entrada fija de 800 por 800, en lugar de reproducir el pipeline de evaluación original que conserva la relación de aspecto.

Pruébalo después de ejecutar `pip install libreyolo`. La [documentación de DETR](https://www.libreyolo.com/docs/models/detr) enumera los cuatro checkpoints. Para compararlo con una CNN, consulta la [alternativa a EfficientDet](/articles/efficientdet-alternative).

[GitHub](https://github.com/LibreYOLO/libreyolo) | [Documentación](https://www.libreyolo.com/docs)
