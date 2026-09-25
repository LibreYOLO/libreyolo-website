---
title: Cómo ejecutar Depth Anything V2 con LibreYOLO
description: Depth Anything V2 ofrece estimación de profundidad monocular de última generación. Así puedes ejecutarlo en dos líneas con LibreYOLO.
date: 2026-06-26
author: Xuban
tags: [LibreYOLO, depth-anything-v2, depth-estimation, tutorial]
faq:
  - q: "¿Cuál es la forma más sencilla de ejecutar Depth Anything V2?"
    a: "Son dos líneas con LibreYOLO: carga LibreDepthAnythingV2l-depth.pt por nombre y llama a predict con save=True. Los pesos se descargan automáticamente la primera vez, y la normalización, el mapa de color y la asignación del dispositivo se gestionan por ti en CUDA, Apple Silicon o CPU estándar."
  - q: "¿Puedo usar Depth Anything V2 con fines comerciales?"
    a: "Solo el encoder Small tiene licencia Apache 2.0. Base, Large y Giant tienen licencia CC-BY-NC-4.0, así que los checkpoints más potentes son para uso no comercial. La licencia original se aplica independientemente de la biblioteca que cargue los pesos."
  - q: "¿LibreYOLO permite entrenar Depth Anything V2?"
    a: "No. El entrenamiento se mantiene en el repositorio original; LibreYOLO cubre la inferencia, el vídeo y la validación para la tarea de profundidad."
---

![El Guggenheim de Bilbao junto a su mapa de profundidad](/articles/simplest-way-to-run-depth-anything-v2/guggenheim-bilbao-input-vs-depth.jpg)

Depth Anything V2 genera algunos de los mejores mapas de profundidad monocular disponibles actualmente.

Si vienes del mundo de YOLO, el repositorio oficial no es el más fácil para empezar. Obtener un único mapa de profundidad requiere varios pasos manuales:

* descargar a mano el checkpoint adecuado y colocarlo en la carpeta correcta,

* hacer coincidir el encoder con su configuración (`encoder="vitl"`, `features=256`, `out_channels=...`),

* escribir por tu cuenta los pasos de normalización y aplicación del color,

* gestionar la asignación del dispositivo para que funcione en un Mac o en CPU, y no solo en CUDA,

* y aprender una API de inferencia que no se parece en nada al resto de tu stack.

Nada de esto es difícil. Solo añade fricción, y puedes evitarlo todo.

LibreYOLO carga los mismos pesos de Depth Anything V2 y ofrece la tarea de profundidad mediante una sola llamada a `predict()`. Indica el nombre del modelo y se descarga la primera vez:

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreDepthAnythingV2l-depth.pt")  # auto-downloads on first run
model.predict("image.jpg", save=True)     # writes a colorized depth map to disk
```

Eso es todo. Si has usado la API estándar de YOLO, la estructura te resultará familiar: carga un modelo por nombre, llama a `predict` y deja que `save=True` guarde la visualización. Es exactamente la misma llamada que usarías para la detección de objetos, salvo que el resultado contiene un mapa de profundidad en lugar de bounding boxes. El mapa de color, la normalización y la asignación del dispositivo se gestionan por ti. Funciona igual en Linux con CUDA, en un Mac con Apple Silicon o en una CPU estándar. No tienes que cambiar el código.

A partir de ahí, la misma llamada permite hacer más cosas: puedes obtener directamente los valores de profundidad sin procesar. El entrenamiento de Depth Anything V2 se mantiene en el repositorio original; LibreYOLO cubre la inferencia, el vídeo y la validación.

La misma llamada de una línea, en escenas muy distintas:

![La imagen de muestra de parkour junto a su mapa de profundidad: quienes saltan en primer plano destacan frente a los muros de hormigón del fondo.](/articles/simplest-way-to-run-depth-anything-v2/parkour-input-vs-depth.jpg)

![Vista aérea de la bahía de La Concha, en Donostia, junto a su mapa de profundidad: las embarcaciones y la costa se perciben cerca, y el mar abierto se aleja.](/articles/simplest-way-to-run-depth-anything-v2/donostia-bay-input-vs-depth.jpg)

![El patio columnado de la Casa de Juntas de Gernika junto a su mapa de profundidad, que muestra cómo la arquitectura se aleja.](/articles/simplest-way-to-run-depth-anything-v2/gernika-casa-juntas-input-vs-depth.jpg)

![Una multitud en un festival nocturno en la Plaza de la Constitución de Donostia junto a su mapa de profundidad; las filas de personas cercanas destacan frente a la fachada iluminada del fondo.](/articles/simplest-way-to-run-depth-anything-v2/donostia-plaza-crowd-input-vs-depth.jpg)

Una nota sobre los pesos: LibreYOLO aloja checkpoints de Depth Anything V2 convertidos y los descarga la primera vez, así que no hace falta descargarlos a mano. La licencia original sigue aplicándose: el encoder Small tiene licencia Apache-2.0, mientras que Base, Large y Giant tienen licencia CC-BY-NC-4.0 (no comercial), por lo que los checkpoints más potentes son para uso no comercial. ¿Quieres trabajar sin conexión o convertir tus propios pesos? El script de conversión de una sola vez sigue disponible:

```bash
# optional: convert an official checkpoint yourself instead of auto-downloading
python weights/convert_depth_anything_v2_weights.py \
  depth_anything_v2_vitl.pth weights/LibreDepthAnythingV2l-depth.pt
```

## Pruébalo

```bash
pip install libreyolo
```

LibreYOLO es la biblioteca de visión por computador más completa que puedes instalar con pip. Una API familiar abarca un catálogo creciente de modelos de última generación: detección de objetos (RF-DETR, D-FINE, DEIM), segmentación, pose, bounding boxes orientados y ahora profundidad monocular con Depth Anything V2, además de entrenamiento y validación para las tareas compatibles. Funciona en los principales sistemas operativos, con GPU o CPU, y cuenta con licencia MIT, así que puedes distribuirla comercialmente sin restricciones.

Dale una estrella en GitHub: [github.com/LibreYOLO/libreyolo](https://github.com/LibreYOLO/libreyolo) | Documentación: [libreyolo.com/docs](https://www.libreyolo.com/docs/v1.3.1)
