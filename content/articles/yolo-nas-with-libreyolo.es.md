---
title: "YOLO-NAS sigue activo: entrenar, validar y exportar con LibreYOLO"
description: "Ejecuta, entrena, valida y exporta YOLO-NAS con LibreYOLO. SuperGradients publicó su última versión en abril de 2024. Planeamos entrenar nuevos pesos desde cero para prescindir de la licencia de Deci."
date: 2026-08-20
author: Xuban
tags: [LibreYOLO, yolo-nas, yolo-nas-maintained, object-detection]
faq:
  - q: "¿Está abandonado YOLO-NAS?"
    a: "SuperGradients publicó la versión 3.7.1 en abril de 2024. LibreYOLO admite la predicción, el entrenamiento, la validación y la exportación de YOLO-NAS con versiones actuales de PyTorch."
  - q: "¿Ultralytics mantiene YOLO-NAS?"
    a: "Ultralytics admite la inferencia, la validación y la exportación. No admite el entrenamiento."
  - q: "¿Se pueden usar comercialmente los pesos preentrenados de YOLO-NAS?"
    a: "Los pesos preentrenados de Deci conservan las condiciones no comerciales de su proyecto original, independientemente de la biblioteca que los cargue. Entrenar desde una inicialización aleatoria evita usar esos checkpoints. También planeamos publicar pesos de COCO entrenados desde cero."
---

YOLO-NAS sigue siendo un detector útil. Su proyecto original, [SuperGradients](https://github.com/Deci-AI/super-gradients), publicó por última vez la versión 3.7.1 en abril de 2024. LibreYOLO mantiene la predicción, el entrenamiento, la validación y la exportación del modelo.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLONASs.pt")
result = model("image.jpg", save=True)
print(result.boxes.xyxy)
```

LibreYOLO admite los tamaños S, M y L para detección, además de pose. Los checkpoints se descargan de la CDN pública de Deci con nombres como `LibreYOLONASs.pt`. LibreYOLO no aloja ninguno de ellos, y siguen aplicándose las condiciones no comerciales de Deci. SuperGradients fija la versión de PyTorch entre la 1.9 y la 1.13; LibreYOLO requiere la 2.4 o una posterior. La exportación para detección incluye ONNX, TorchScript, OpenVINO, NCNN, TFLite, Paddle, MNN, ExecuTorch y Core AI. CoreML no es compatible.

Para entrenar sin un checkpoint de Deci:

```python
from libreyolo import LibreYOLONAS

model = LibreYOLONAS(None, size="s")
model.train(data="my-dataset.yaml", imgsz=640, batch=16)
```

También planeamos entrenar YOLO-NAS desde cero con COCO y publicar esos pesos.

Instálalo con `pip install libreyolo`. La [documentación de YOLO-NAS](https://www.libreyolo.com/docs/models/yolo-nas) incluye los detalles del modelo. También puedes leer el artículo más extenso sobre la [alternativa a SuperGradients](/articles/supergradients-alternative).

[GitHub](https://github.com/LibreYOLO/libreyolo) | [Documentación](https://www.libreyolo.com/docs)
