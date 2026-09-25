---
title: "Alternativa a EfficientDet en 2026: ejecutar D0-D4 con LibreYOLO"
description: "Ejecuta predicciones, validaciones y exportaciones de EfficientDet D0-D4 con la API de detección de objetos de LibreYOLO."
date: 2026-08-15
author: Xuban
tags: [LibreYOLO, efficientdet-alternative, object-detection, tutorial]
faq:
  - q: "¿Está abandonado EfficientDet?"
    a: "El repositorio de PyTorch de rwightman no está archivado oficialmente. Es más preciso decir que está inactivo que abandonado."
  - q: "¿Puede LibreYOLO entrenar EfficientDet?"
    a: "No. LibreYOLO admite la inferencia, la validación y la exportación de EfficientDet, pero no el entrenamiento."
---

El popular [repositorio de EfficientDet para PyTorch](https://github.com/rwightman/efficientdet-pytorch) no está archivado, así que decir que está abandonado sería incorrecto. Está inactivo. LibreYOLO ofrece otra vía de despliegue.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreEfficientDetd0.pt")
result = model("image.jpg", save=True)
print(result.boxes.xyxy)
```

LibreYOLO admite EfficientDet D0 a D4 para predicción y validación. Las exportaciones a ONNX, TorchScript, OpenVINO y TensorRT tienen cobertura de paridad. El entrenamiento no está implementado y cada tamaño mantiene su resolución de entrada nativa fija.

Instala el paquete con `pip install libreyolo`. Consulta la [documentación de EfficientDet](https://www.libreyolo.com/docs/models/efficientdet) para ver cada tamaño, o compáralo con la [alternativa a Facebook DETR](/articles/facebook-detr-alternative).

[GitHub](https://github.com/LibreYOLO/libreyolo) | [Documentación](https://www.libreyolo.com/docs)
