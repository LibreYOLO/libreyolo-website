---
title: "Alternativa a MMYOLO en 2026: ejecuta RTMDet con LibreYOLO"
description: "La última versión de MMYOLO es de 2023. Ejecuta detectores compatibles como RTMDet mediante la API directa de LibreYOLO."
date: 2026-08-15
author: Xuban
tags: [LibreYOLO, mmyolo-alternative, rtmdet, object-detection]
faq:
  - q: "¿MMYOLO está abandonado?"
    a: "MMYOLO no está archivado formalmente, pero su última versión de GitHub es la 0.6.0, de agosto de 2023."
  - q: "¿Puede LibreYOLO cargar cualquier checkpoint de MMYOLO?"
    a: "No. LibreYOLO admite familias de modelos concretas y no garantiza la carga directa de checkpoints arbitrarios de MMYOLO."
---

[MMYOLO](https://github.com/open-mmlab/mmyolo) no está archivado, pero su última versión, la 0.6.0, es de agosto de 2023. Si solo necesitas un detector compatible como RTMDet, quizá no necesites todo el stack de OpenMMLab.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRTMDets.pt")
result = model("image.jpg", save=True)
print(result.boxes.xyxy)
```

LibreYOLO no es un reemplazo completo de MMYOLO. No reproduce todas las configuraciones, recetas ni checkpoints. Sí ofrece una API directa para familias compatibles como RTMDet y YOLOX, con la que puedes hacer predicciones, entrenamiento, validación y exportación.

Empieza con `pip install libreyolo`. La [documentación de RTMDet](https://www.libreyolo.com/docs/models/rtmdet) enumera las operaciones compatibles. La [guía breve de RTMDet](/articles/rtmdet-without-mmdetection) explica la migración.

[GitHub](https://github.com/LibreYOLO/libreyolo) | [Documentación](https://www.libreyolo.com/docs)
