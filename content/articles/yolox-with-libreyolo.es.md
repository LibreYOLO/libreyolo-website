---
title: "Alternativa a YOLOX en 2026: ejecútalo con LibreYOLO"
description: "La última versión de YOLOX es de 2022. Ejecuta, entrena, valida y exporta YOLOX con la API de LibreYOLO."
date: 2026-08-15
author: Xuban
tags: [LibreYOLO, yolox-alternative, object-detection, tutorial]
faq:
  - q: "¿Está abandonado YOLOX?"
    a: "El repositorio oficial no está archivado, pero la última versión es YOLOX 0.3.0, de abril de 2022. Es más preciso decir que el proyecto está inactivo que afirmar que se ha abandonado formalmente."
  - q: "¿Puede LibreYOLO entrenar YOLOX?"
    a: "Sí. LibreYOLO admite la predicción, el entrenamiento, la validación y la exportación de YOLOX."
---

YOLOX sigue siendo un detector útil con licencia Apache-2.0. Su [repositorio oficial](https://github.com/Megvii-BaseDetection/YOLOX) no está archivado, pero la última versión, la 0.3.0, es de abril de 2022. El problema de mantenimiento es real. El modelo sigue funcionando.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLOXs.pt")
result = model("image.jpg", save=True)
print(result.boxes.xyxy)
```

LibreYOLO admite seis tamaños de YOLOX, de Nano a X, con predicción, entrenamiento, validación y exportación en una sola API. Usa el proyecto original si necesitas su flujo de configuración `Exp` exacto.

Empieza con `pip install libreyolo`. La [documentación de YOLOX](https://www.libreyolo.com/docs/models/yolox) cubre toda la API. También hay una nota breve que explica que [YOLO-NAS sigue teniendo mantenimiento](/articles/yolo-nas-with-libreyolo).

[GitHub](https://github.com/LibreYOLO/libreyolo) | [Documentación](https://www.libreyolo.com/docs)
