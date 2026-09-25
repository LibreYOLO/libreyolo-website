---
title: "Alternativa a SuperGradients para visión artificial en tiempo real"
description: "SuperGradients dejó de publicar versiones tras la adquisición de Deci por NVIDIA en 2024. LibreYOLO es una alternativa mantenida que ejecuta los mismos pesos de YOLO-NAS: predice, entrena y exporta con una API bajo licencia MIT."
date: 2026-08-15
author: Xuban
tags: [LibreYOLO, supergradients-alternative, yolo-nas, object-detection]
faq:
  - q: "¿Se sigue manteniendo SuperGradients?"
    a: "No. La última versión fue la 3.7.1 en abril de 2024, justo antes de que NVIDIA adquiriera Deci. Desde entonces, el repositorio no ha publicado versiones, las incidencias quedan sin respuesta y el sitio de documentación dejó de estar disponible. No está archivado formalmente, pero ya no hay nadie al cargo."
  - q: "¿Cuál es la mejor alternativa a SuperGradients?"
    a: "Para ejecutar YOLO-NAS, LibreYOLO carga los mismos pesos de Deci mediante una API mantenida y con licencia MIT, y añade entrenamiento, validación y exportación. Si dependes de las recetas de entrenamiento de SuperGradients para otras arquitecturas, el repositorio antiguo sigue funcionando si fijas sus dependencias."
  - q: "¿Puedo usar YOLO-NAS comercialmente?"
    a: "Los pesos preentrenados de Deci son para uso no comercial, independientemente de la biblioteca que los cargue. La arquitectura tiene una licencia permisiva, así que entrenar un modelo YOLO-NAS desde cero con tus propios datos te da un modelo que no incluye ningún checkpoint de Deci en su historial."
  - q: "¿LibreYOLO sustituye todo lo que hacía SuperGradients?"
    a: "No todo. LibreYOLO no exporta YOLO-NAS a CoreML y su ruta de TensorRT para esta familia no está probada. Además, las recetas de entrenamiento con cuantización consciente de SuperGradients no tienen un equivalente directo. Para esos casos, conserva el repositorio antiguo."
---

SuperGradients era la biblioteca de entrenamiento de código abierto de Deci y el hogar de YOLO-NAS, uno de los detectores en tiempo real más precisos que se han publicado. En mayo de 2024, NVIDIA adquirió Deci y SuperGradients dejó de avanzar. La última versión, la 3.7.1, se publicó el 8 de abril de 2024. La documentación de supergradients.com dejó de estar disponible. Hay unas 120 incidencias abiertas, y la titulada [«¿Está muerto este proyecto?»](https://github.com/Deci-AI/super-gradients/issues/2062) no tiene respuesta de ningún mantenedor, lo que ya es una respuesta en sí misma.

El repositorio no está archivado, así que a primera vista parece activo. En la práctica, notas que está abandonado en cuanto intentas instalarlo: `super-gradients` fija `torchmetrics==0.8`, que entra en conflicto con las pilas actuales de PyTorch, y además incorpora hydra, omegaconf, boto3 y tensorboard. Cada mes que pasa, esas versiones fijadas entran más en conflicto con el resto de tu entorno, y no llegará ninguna solución.

Nada de esto hace que YOLO-NAS sea un modelo peor. La variante grande sigue alcanzando 52.2 mAP en COCO a velocidad de tiempo real. El modelo está bien; lo que se quemó fue su casa.

<iframe
  src="https://visionanalysis.org/embed/scatter?highlight=yolonas-s%2Cyolonas-m%2Cyolonas-l"
  width="100%"
  height="420"
  style="border:0;border-radius:12px;overflow:hidden"
  loading="lazy"
  title="YOLO-NAS accuracy vs parameters - visionanalysis.org">
</iframe>

## Ejecuta los mismos pesos con LibreYOLO

LibreYOLO carga los checkpoints de YOLO-NAS directamente desde la CDN de Deci, mediante la misma API que utiliza para todas las demás familias de modelos. No tienes que escribir configuraciones de hydra ni desenvolver un wrapper de predicción:

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLONASs.pt")  # auto-downloads on first run
results = model("image.jpg", save=True)
print(results[0].boxes.xyxy)
print(results[0].boxes.conf)
```

Funcionan todas las variantes de detección S, M y L, y también la variante de pose: sustituye el modelo por `LibreYOLONASs-pose.pt` y obtendrás los puntos clave de COCO. Como todas las familias devuelven el mismo objeto `Results`, comparar YOLO-NAS con RF-DETR o D-FINE en tus propios datos requiere cambiar una línea, en vez de usar otra base de código.

## Entrenamiento y exportación, no solo inferencia

SuperGradients era ante todo una biblioteca de entrenamiento, así que una alternativa real también debe permitir entrenar. LibreYOLO ajusta YOLO-NAS a tu dataset con la misma llamada que utiliza para cualquier otro modelo:

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLONASs.pt")
model.train(data="my-dataset.yaml", epochs=100, imgsz=640, batch=16)
```

También puedes entrenar desde una inicialización aleatoria, algo relevante para las licencias (más información abajo). La validación devuelve métricas mAP para cualquier dataset en tu formato, y la exportación admite ONNX, TorchScript, OpenVINO, NCNN y TFLite. Es una matriz de formatos de exportación más amplia que la que llegó a ofrecer la biblioteca original para YOLO-NAS. Encontrarás todos los detalles en la [página de documentación de YOLO-NAS](https://www.libreyolo.com/docs/models/yolo-nas), y una nota más breve sobre cómo [YOLO-NAS sigue teniendo mantenimiento](/articles/yolo-nas-with-libreyolo).

## Cuándo el repositorio antiguo sigue siendo la opción adecuada

Seamos claros. Hay tres casos en los que conviene conservar SuperGradients instalado:

**CoreML.** LibreYOLO no exporta YOLO-NAS a CoreML. Si publicas en dispositivos Apple y necesitas un `.mlpackage`, SuperGradients todavía tiene una ruta funcional.

**TensorRT.** SuperGradients documentaba y probaba su flujo de TensorRT para YOLO-NAS, incluidos los detalles particulares del tamaño de batch. La compatibilidad de LibreYOLO con TensorRT para esta familia no está probada.

**Recetas de entrenamiento con cuantización consciente.** YOLO-NAS se diseñó para INT8, y SuperGradients publicó las recetas QAT que permitían aprovecharlo al máximo. LibreYOLO exporta con cuantización INT8 y FP16, pero no tiene un equivalente a esas recetas de entrenamiento.

El repositorio todavía funciona si fijas un entorno de la época de 2024. Simplemente, no deberías construir nada nuevo sobre una base que nadie mantiene.

## Una nota sobre los pesos

Los pesos preentrenados de YOLO-NAS son de Deci y se publicaron bajo una licencia para uso no comercial. Esa licencia acompaña a los pesos, sea cual sea la biblioteca que los cargue. LibreYOLO no aloja ni refleja ninguno de esos pesos: la descarga procede de la CDN pública de Deci y muestra sus condiciones antes de comenzar. Para investigación y trabajo no comercial, ambas opciones son válidas.

Si necesitas un YOLO-NAS para uso comercial, ahora hay una vía clara: la arquitectura tiene una licencia permisiva, así que entrenar desde cero con tus propios datos produce un modelo que no deriva de ningún checkpoint de Deci. LibreYOLO lo permite con `LibreYOLONAS(None, size="s")`.

## Pruébalo

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLONASl.pt")
results = model("image.jpg", save=True)
```

LibreYOLO tiene licencia MIT, funciona en Linux, Mac y Windows, y funciona en GPU, Apple Silicon y CPU sin cambiar el código. Una sola API abarca YOLO-NAS, RF-DETR, D-FINE, DEIM, YOLOX, RTMDet y muchos más, en detección, segmentación, pose, clasificación, profundidad y seguimiento.

Dale una estrella en GitHub: [github.com/LibreYOLO/libreyolo](https://github.com/LibreYOLO/libreyolo) | Documentación: [libreyolo.com/docs](https://www.libreyolo.com/docs)
