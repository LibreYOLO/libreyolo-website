---
title: "LibreYOLO en CVPR 2026: inferencia edge con Hailo-8L y Snapdragon"
description: "En CVPR 2026, en Denver, un equipo de Jabra y la IT University of Copenhagen usó LibreYOLOXs como modelo de ejemplo en su tutorial «Edge AI in Action» y lo ejecutó en Hailo-8L y Snapdragon."
date: 2026-06-30
author: Xuban
tags: [LibreYOLO, yolox, edge-ai, cvpr]
faq:
  - q: "¿A qué velocidad se ejecuta LibreYOLOXs en hardware edge?"
    a: "En el tutorial de CVPR 2026, LibreYOLOXs se ejecutó a 19.0 ms por frame (52.6 FPS) en una Raspberry Pi 5 con Hailo-8L y a 6.69 ms en el HTP/DSP de un Qualcomm QCS6490 tras la cuantización INT8."
  - q: "¿Cómo despliego un modelo de LibreYOLO en un acelerador edge?"
    a: "Expórtalo a ONNX con la llamada de exportación de LibreYOLO y después compílalo para el hardware de destino: Hailo Dataflow Compiler genera un HEF para los chips Hailo, y la pila SNPE / QAIRT / AI Hub sirve para Qualcomm Snapdragon. Ese es el flujo completo que recorrió el tutorial de CVPR."
---

![CVPR 2026, Denver, Colorado, del 3 al 7 de junio](/articles/libreyolo-at-cvpr-2026/cvpr-denver-banner.png)

LibreYOLO estuvo presente en CVPR 2026.

CVPR está ampliamente reconocida como la conferencia de visión artificial más prestigiosa del mundo. Este año se celebró en Denver, donde un equipo de Jabra y la IT University of Copenhagen impartió un tutorial llamado «Edge AI in Action: Mastering On-Device Inference». Eligieron LibreYOLOXs como modelo de ejemplo para llevar la detección de objetos a chips edge.

## Qué desarrollaron

El tutorial recorrió el flujo completo de edge de principio a fin. Primero, un modelo LibreYOLOXs. Después, su exportación a ONNX. Por último, la compilación de ese ONNX para dos tipos de hardware muy distintos: un acelerador Hailo-8L y Qualcomm Snapdragon.

## Tiempo real en Hailo-8L

Compilaron el ONNX a HEF con Hailo Dataflow Compiler y después lo ejecutaron en una Raspberry Pi 5 con Hailo-8L AI HAT.

![LibreYOLOXs se ejecuta en una Raspberry Pi 5 con Hailo-8L y detecta personas y bolsos en una calle concurrida](/articles/libreyolo-at-cvpr-2026/live-detection-hailo.png)

Se ejecutó a 19.0 ms por frame, 52.6 FPS, en una Raspberry Pi.

## Tiempo real en Snapdragon

En el lado de Qualcomm, cuantizaron LibreYOLOXs a INT8 y lo ejecutaron con la pila SNPE, QAIRT y AI Hub; las pruebas se realizaron en un QCS6490.

<img src="/articles/libreyolo-at-cvpr-2026/live-detection-qualcomm.png" alt="LibreYOLOXs detecta en directo una televisión, un portátil, un teclado, un ratón y un teléfono móvil en la aplicación EdgeVision AI de un teléfono Snapdragon" style="display:block;margin:1.5rem auto;max-width:360px;width:100%" />

El resultado fue de 6.69 ms en el HTP/DSP.

## Gracias

Muchas gracias a Sai Narsi Reddy Donthi Reddy, Fabricio Batista Narcizo, Elizabete Munzlingera y Shan Ahmed Shaffi por presentar el proyecto.

- Tutorial y diapositivas: [Edge AI in Action: Mastering On-Device Inference](https://www.fabricionarcizo.com/cvpr2026-edge-ai-in-action/)
- Modelos LibreYOLOXs cuantizados para Qualcomm: [en Hugging Face](https://huggingface.co/fabricionarcizo/LibreYOLOXs)

## Pruébalo

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLOXs.pt")
model.export(format="onnx")   # then compile for your edge target
```

LibreYOLO tiene licencia MIT, funciona en Linux, Mac y Windows, y se ejecuta en GPU, Apple Silicon y CPU sin cambiar el código. Una sola API abarca YOLOX, RF-DETR, D-FINE, DEIM, YOLO-NAS, segmentación, pose, profundidad y más.

Dale una estrella en GitHub: [github.com/LibreYOLO/libreyolo](https://github.com/LibreYOLO/libreyolo) | Documentación: [libreyolo.com/docs](https://www.libreyolo.com/docs/v1.3.1)
