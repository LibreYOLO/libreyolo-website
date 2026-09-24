---
title: "Mejores alternativas a Ultralytics en 2026"
description: "Guía práctica de las mejores alternativas de código abierto a Ultralytics YOLO en 2026: sus licencias, tareas compatibles y límites de despliegue, y el lugar que ocupa LibreYOLO, la biblioteca YOLO con licencia MIT más completa."
date: 2026-07-01
author: Xuban
tags: [LibreYOLO, ultralytics-alternative, object-detection, yolo, mit-license]
faq:
  - q: "¿Ultralytics YOLO es gratuito para uso comercial?"
    a: "Solo bajo AGPL-3.0, que exige publicar el código fuente de la aplicación que crees con él, incluidos los servicios en red. Para el uso comercial con código cerrado se necesita la licencia comercial de pago de Ultralytics. Las alternativas permisivas (MIT, Apache-2.0) evitan esta obligación."
  - q: "¿Cuál es la alternativa a Ultralytics con menos restricciones de licencia?"
    a: "Para una stack permisiva que puedas desplegar en cualquier sitio: LibreYOLO (código MIT), RF-DETR (Apache-2.0) y YOLOX (Apache-2.0). Todas evitan el copyleft de AGPL."
  - q: "¿Qué está sustituyendo a YOLO en 2026?"
    a: "Cada vez más, los detectores transformer en tiempo real: RT-DETR, RF-DETR y la línea D-FINE y DEIM. Igualan o superan la precisión de YOLO con una latencia similar en hardware capaz. Las CNN de estilo YOLO siguen ganando en dispositivos edge pequeños, donde esos transformers funcionan mal y carecen de una ruta madura para NCNN o CPU."
  - q: "¿Pueden ejecutarse en una Raspberry Pi o una NPU?"
    a: "Depende de la exportación. Los detectores CNN como YOLOX y RTMDet se exportan a NCNN y funcionan en una Pi, y algunos (YOLOX) llegan a la NPU Hailo. Los detectores transformer como RF-DETR no; necesitan una GPU o una CPU potente."
---

Aviso: LibreYOLO es nuestro y ocupa el primer lugar de esta lista. Usamos todas las demás herramientas aquí en nuestro trabajo real e indicamos en qué aspectos superan a LibreYOLO.

La mayoría de los equipos dejan Ultralytics por uno de estos tres motivos:

- **Licencia.** Los modelos YOLO de Ultralytics usan AGPL-3.0 (YOLOv5 desde 2023 y v8 en adelante). El patrón es conocido: creas un producto, lo distribuyes y, entonces, una revisión legal o un cliente señala el modelo. La decisión pasa a ser publicar el código de toda tu aplicación o comprar una licencia comercial. El copyleft de AGPL alcanza tu código en cuanto lo distribuyes o lo ofreces como servicio. Es un motivo habitual para buscar otras opciones.
- **Apertura.** Un solo proveedor controla los modelos, el código de entrenamiento y las condiciones. Algunos equipos quieren pesos y código con licencias permisivas sobre los que puedan construir sin pedir permiso.
- **El modelo.** YOLO no siempre es la mejor arquitectura para cada tarea. Los detectores transformer en tiempo real como RT-DETR, RF-DETR y D-FINE pueden ser más precisos con la misma latencia. El inconveniente es que están repartidos entre repositorios de investigación y, como verás, LibreYOLO ejecuta los tres con una sola API. Por eso, este es un motivo para cambiar de biblioteca, no para dejarla por un único modelo.

Evaluamos cada herramienta de abajo según tres criterios: si puedes distribuirla con su licencia, si se instala y ejecuta con una versión actual de PyTorch y si se exporta al hardware en el que realmente haces el despliegue.

Al leer la lista, se repite un patrón: cada alternativa es buena, pero resulta engorroso usarla por separado. YOLOX apenas se instala, MMDetection es un purgatorio de versiones de wheels fijadas, la familia RT-DETR es código de investigación sin canal de asistencia y Detectron2 está congelado desde 2021. LibreYOLO encabeza la lista porque ya integra la mayoría de ellas, con mantenimiento, bajo MIT, sobre una stack actual y detrás de una API familiar. Es menos una opción de la lista que la capa que las reúne.

## Resumen rápido

| Herramienta | Licencia | Tareas compatibles | Ideal para | Exportación edge |
| --- | --- | --- | --- | --- |
| **LibreYOLO** | MIT (código) | Detección, segmentación, pose, clasificación, profundidad, mirada, seguimiento | El centro MIT para más de 20 familias de modelos bajo una sola API | ONNX, TensorRT, OpenVINO, NCNN, CoreML, TFLite |
| **RF-DETR** | Apache-2.0 (N/S/M/L) | Detección, segmentación, pose (vista previa) | Detección y segmentación en producción | ONNX, TFLite, TensorRT; sin NCNN ni Hailo |
| **Lightly / LightlyTrain** | MIT / AGPL-3.0 | Preentrenamiento, destilación | Datos sin etiquetar, destilación de DINOv2/v3 | n/a (no es un detector) |
| **YOLOX** | Apache-2.0 | Detección | Detección en tiempo real sin anchors | La versión upstream es difícil de instalar; funciona mediante LibreYOLO |
| **MMDetection / fork OneDL** | Apache-2.0 | Todo (investigación) | Amplitud de arquitecturas y reproducción de artículos | Mediante exportación |
| **Detectron2** | Apache-2.0 | Detección, segmentación, puntos clave | Mask R-CNN y la familia R-CNN | Manual |
| **D-FINE / DEIM / DEIMv2** | Apache-2.0 | Detección | DETR en tiempo real de última generación | ONNX, TensorRT |
| **EdgeCrafter** | Apache-2.0 | Detección, segmentación, pose | ViT compactos para edge destilados de DINOv3 | Nivel de investigación |
| **RT-DETR (v1-v4)** | Apache-2.0 | Detección | Detección en tiempo real de vanguardia | ONNX, TensorRT |

## 1. LibreYOLO

Licencia: MIT (código). Ideal como la biblioteca YOLO con licencia MIT más completa y como alternativa directa a Ultralytics.

**LibreYOLO es la biblioteca YOLO con licencia MIT que ejecuta todos los modelos con una sola API.** Es nuestro proyecto y, después de leer el resto de esta lista, su papel debería quedar claro: es el centro que ejecuta las alternativas. LibreYOLO ya integra casi todos los repositorios problemáticos mencionados arriba, YOLOX, RTMDet, RF-DETR, D-FINE, DEIM y la línea RT-DETR, detrás de una API familiar y mantenida. Así obtienes el modelo sin tener que investigar cómo instalarlo ni lidiar con la licencia.

Hay una razón que va más allá de la comodidad. YOLO comenzó como investigación abierta: cualquiera podía usar y ampliar Darknet de Joseph Redmon, de v1 a v3. La era AGPL lo cerró. LibreYOLO existe para que ese trabajo vuelva a ser accesible, como querían sus creadores: con una licencia permisiva, mantenido por la comunidad y sin enviar datos a casa. Hay dos motivos, entonces.

**Primero: es la alternativa a Ultralytics YOLO con licencia MIT.** Mantiene la API que ya conoces, así que tus datasets en formato YOLO y tus scripts se adaptan con pocos cambios, pero usa MIT en lugar de AGPL: el copyleft no alcanza tu código, no tienes que comprar una licencia comercial ni hay telemetría que envíe datos a casa como hace Ultralytics de forma predeterminada. Si has llegado aquí por la licencia, ese es el objetivo. YOLO9 y RF-DETR son las opciones predeterminadas, ampliamente probadas y listas para producción; el resto del catálogo es más nuevo y está marcado claramente como experimental. Preferimos decirlo en vez de fingir que todo está probado en entornos reales.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9c.pt")   # or LibreRFDETRl.pt, LibreDFINEl.pt, ...
results = model("image.jpg", save=True)
```

**Segundo: cubre mucho más que Ultralytics.** Una sola API incluye más de 20 familias de modelos: YOLO basados en CNN, DETR transformer, backbones ViT, SAM y más, no solo el catálogo de un proveedor. Y no incluye únicamente detectores:

- **Tareas:** segmentación de instancias y semántica, pose, clasificación y cajas orientadas, además de dos tareas que Ultralytics no tiene: profundidad monocular (Depth Anything V2) y estimación de la mirada (L2CS).
- **La capa práctica:** seguimiento de múltiples objetos con ID persistentes (ByteTrack y OC-SORT), inferencia de vídeo con submuestreo de fotogramas y vista previa en directo, inferencia por teselas para imágenes de drones y satélites, una interfaz web de arrastrar y soltar (`libreyolo ui`) y un comando `doctor` que comprueba tu dataset antes de que inviertas tiempo en un entrenamiento.
- **Entrenamiento avanzado:** acumulación de gradientes, congelación de capas, reanudación, aumento de datos en tiempo de prueba, fine-tuning LoRA/DoRA, múltiples GPU y registro con TensorBoard/MLflow/Weights & Biases.
- **Exportación:** siete formatos (ONNX, TorchScript, TensorRT, OpenVINO, NCNN, CoreML, TFLite) con cuantización INT8/FP16, NMS integrado y metadatos del modelo incluidos.
- **Funciona en todas partes:** acepta rutas, URL, PIL, NumPy, tensores o bytes sin procesar y funciona con CUDA, Apple Silicon (MPS) o una CPU normal sin cambiar el código.

Cuando el repositorio de un detector potente está abandonado o es difícil de instalar, LibreYOLO ejecuta sus pesos con mantenimiento y sobre una stack actual.

## 2. RF-DETR

Licencia: Apache-2.0 (Nano, Small, Medium, Large). Ideal para detección y segmentación en producción.

RF-DETR, de Roboflow y presentado en ICLR 2026, es el modelo más preparado para producción de esta lista. Está bien diseñado, viene de un equipo que entrega productos y es una opción inicial sólida cuando necesitas una detección o segmentación que funcione y en la que puedas confiar dentro de un sistema real. El paquete `rfdetr` y los pesos de Nano a Large usan Apache-2.0. Los pesos más grandes XL y 2XL usan la licencia PML de Roboflow.

## 3. Lightly

Ideal para datos sin etiquetar, preentrenamiento autosupervisado y destilación de un backbone DINO.

Lightly no es un detector. Sirve para obtener valor de datos que no has etiquetado y consta de dos partes con licencias distintas.

- **[`lightly`](https://github.com/lightly-ai/lightly) (LightlySSL), MIT.** Un framework de componentes de aprendizaje autosupervisado: funciones de pérdida, cabezas, aumentos, bancos de memoria e implementaciones de referencia de más de veinte métodos (SimCLR, MoCo, BYOL, DINO, DINOv2, MAE y otros). Tú escribes el bucle de entrenamiento; esta herramienta te da todo lo necesario para preentrenar una representación desde cero con imágenes sin etiquetar.
- **[LightlyTrain](https://github.com/lightly-ai/lightly-train), AGPL-3.0 (con opciones comerciales y opciones gratuitas para la comunidad).** La opción lista para usar y el flujo de trabajo al que suele referirse la gente. Indícale un modelo fundacional como DINOv2 o DINOv3 y tus datos sin etiquetar, y en unas pocas líneas destilará ese backbone en un modelo más pequeño que puedes desplegar (YOLO, RT-DETR, ViT o una red personalizada), sin etiquetas. Ten en cuenta la licencia: AGPL-3.0 es el mismo copyleft que lleva a la gente a buscar alternativas a Ultralytics. Lee las condiciones si planeas distribuir software de código cerrado o elige la licencia comercial.

Elige Lightly cuando el cuello de botella sea conseguir un buen backbone sin disponer de etiquetas. Complementa a los detectores de esta lista en lugar de competir con ellos. Los más recientes también lo demuestran: RT-DETRv4 incorpora la destilación de un modelo fundacional de visión al entrenamiento y DEIMv2 integra backbones DINOv3 directamente en el detector.

## 4. YOLOX

Licencia: Apache-2.0. Ideal para detección en tiempo real sin anchors, una vez instalado.

El repositorio upstream está congelado desde su última versión, v0.3.0, publicada en abril de 2022, tiene más de 700 issues abiertos y es difícil de instalar en una stack de 2026. No hay `pyproject.toml` y `requirements.txt` fija `onnx-simplifier==0.4.10`, que no tiene wheels precompiladas para versiones posteriores a Python 3.10. Por eso, en un intérprete moderno se compila desde el código fuente y requiere cmake y una toolchain de C++. NumPy tampoco tiene una versión fijada, lo que puede provocar incompatibilidades de ABI entre NumPy 2.0 y versiones antiguas de pycocotools y torch. La detección principal funciona una vez resuelta la cadena de dependencias, pero llegar hasta ahí tiene un coste.

El modelo sigue siendo bueno: un detector en tiempo real sin anchors de Megvii, con código y pesos bajo Apache-2.0, y una base de comparación razonable aunque detectores más recientes ya lo hayan superado. Reactivar un repositorio abandonado pero sólido es el tipo de tarea que un agente de programación puede resolver en una tarde, así que el estado de la instalación supone menos obstáculo de lo que parece. LibreYOLO también ejecuta directamente los pesos de YOLOX en una stack actual. En cualquier caso, vale la pena usar la arquitectura. Escribimos una guía más completa [aquí](/articles/yolox-with-libreyolo).

## 5. El universo de MMDetection

Licencia: principalmente Apache-2.0. Ideal para tener una gran variedad de arquitecturas y reproducir artículos.

Durante años, MMDetection alojó la implementación oficial de casi todos los artículos sobre detección: Faster R-CNN, DINO, Grounding-DINO, RTMDet, Mask R-CNN y cientos de configuraciones. En 2026, OpenMMLab ha ido cerrando la serie mm. La última versión de [MMDetection](https://github.com/open-mmlab/mmdetection) fue v3.3.0, en enero de 2024; las issues quedan sin respuesta y la stack está bloqueada por la versión fijada de `mmcv`, cuyas wheels precompiladas van por detrás de las versiones actuales de PyTorch y CUDA. Por eso, una instalación nueva suele fallar hasta que se bajan las versiones de todo. Explicamos ese problema [aquí](/articles/rtmdet-without-mmdetection).

Una empresa neerlandesa, VBTI, lo mantiene en marcha. Su [fork OneDL](https://github.com/VBTI-development/onedl-mmdetection) vuelve a publicar toda la stack con el prefijo `onedl-` (`onedl-mmdetection`, `onedl-mmcv`, `onedl-mmengine` y el resto), recompilada para las versiones actuales de PyTorch 2.x, CUDA y Python 3.10+, lo que resuelve el bloqueo de versiones. Usa Apache-2.0 y tiene mantenimiento activo; su versión v3.5.1 salió en mayo de 2026. Si quieres la variedad de MMDetection sin el trabajo de instalación, usa este fork. Es un equipo pequeño, así que contribuye si dependes de él.

Dos herramientas relacionadas:

- **[Detectron2](https://github.com/facebookresearch/detectron2)** (Meta, Apache-2.0) está en modo de mantenimiento y no tiene una versión etiquetada desde v0.6, en 2021, pero es fiable y sigue siendo la fuente más clara de Mask R-CNN y de la familia R-CNN para segmentación de instancias y panóptica.
- **[TorchVision](https://github.com/pytorch/vision)** (BSD-3-Clause, con mantenimiento activo) incluye Faster R-CNN, RetinaNet, FCOS, SSD, Mask R-CNN y Keypoint R-CNN. No tiene YOLO moderno ni DETR, y tienes que escribir tu propio bucle de entrenamiento, pero es la base de comparación sin dependencias adicionales.

Una advertencia para el uso comercial: **MMYOLO**, el centro de OpenMMLab para reimplementaciones de YOLO, usa GPL-3.0 en lugar de Apache-2.0 y está congelado desde agosto de 2023.

## 6. La familia RT-DETR

Licencia: Apache-2.0 en todos los casos. Ideal para detección en tiempo real de última generación con código de investigación.

La frontera actual de la detección en tiempo real la forman varios detectores transformer relacionados, en lugar de un YOLO. La mayoría desciende de RT-DETR, comparte buena parte del código del backbone y el encoder y casi todos usan Apache-2.0, por lo que es fácil cambiar de uno a otro. La mayoría se limita a la detección, con una excepción (EdgeCrafter), que amplía las mismas ideas de destilación a la segmentación y la pose. El coste común es que son repositorios de investigación: el entrenamiento usa archivos de configuración, su uso es distinto al de Ultralytics y no tienen un canal de asistencia. Los modelos son potentes y, por lo general, la exportación a ONNX y TensorRT está muy bien resuelta.

- **[D-FINE](https://github.com/Peterande/D-FINE)** (USTC, ICLR 2025 Spotlight). Reformula la regresión de cajas como refinamiento de distribuciones con autodestilación. Ofrece una buena precisión en relación con la latencia (D-FINE-X alcanza unos 55.8 AP en aproximadamente 13 ms en una T4), tamaños de N a X e integración con Hugging Face Transformers, lo que lo convierte en el más fácil de esta lista para cargar y ajustar fuera de su propio repositorio.
- **[DEIM y DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2)** (Intellindust AI Lab). DEIM (CVPR 2025) es una receta de entrenamiento (emparejamiento Dense O2O) que se añade a D-FINE o RT-DETRv2 y reduce aproximadamente a la mitad el tiempo de entrenamiento. DEIMv2 añade características DINOv3 y llega a una categoría Atto de menos de 1M de parámetros para móviles; DEIMv2-S es el primer modelo de menos de 10M que supera 50 AP. Tiene mantenimiento activo y ha recibido actualizaciones hasta 2026.
- **[EdgeCrafter](https://github.com/Intellindust-AI-Lab/EdgeCrafter)** (Intellindust AI Lab, 2026). Es el trabajo más reciente del mismo laboratorio y la única opción de esta lista que va más allá de la detección: cubre detección (ECDet), segmentación de instancias (ECSeg) y pose (ECPose), cada una en tamaños S/M/L/X y todas bajo Apache-2.0. Adapta un ViT grande preentrenado con DINOv3 para crear un modelo docente especializado en cada tarea y lo destila en backbones compactos para modelos alumno diseñados para edge. Sus cifras son sólidas para su tamaño: ECDet-S alcanza 51.7 AP con menos de 10M de parámetros usando solo COCO, ECPose-X llega a 74.8 AP (por encima de los 71.6 de YOLO26Pose-X) y su segmentación de instancias compite con la de RF-DETR. Es muy nuevo, así que considéralo una versión de investigación, pero es uno de los pocos modelos compactos que cubre las tres tareas.
- **[RT-DETR y RT-DETRv2](https://github.com/lyuwenyu/RT-DETR)** (Baidu). El artículo original "DETRs beat YOLOs on real-time detection" (CVPR 2024) y el origen de la familia: sin NMS, de extremo a extremo y fácil de exportar. v2 es una actualización de 2024 con una estrategia bag-of-freebies, una mejora directa con la misma latencia. La versión original para Paddle y la adaptación de referencia para PyTorch comparten este repositorio, y el modelo está integrado en HF Transformers.
- **[RT-DETRv3](https://github.com/clxia12/RT-DETRv3)** (Baidu, WACV 2025 Oral) y **[RT-DETRv4](https://github.com/RT-DETRs/RT-DETRv4)** (Universidad de Pekín y Tsinghua, finales de 2025). v3 solo admite PaddlePaddle. v4 es la versión actual más potente de la línea (X cerca de 57 AP), usa PyTorch y destila un modelo fundacional de visión en un detector ligero sin coste adicional de inferencia. Su base de código también reproduce D-FINE, DEIM y RT-DETRv2 mediante cambios de configuración.
- **[LW-DETR](https://github.com/Atten4Vis/LW-DETR)** (Baidu). Un DETR sencillo basado en ViT que se presenta como sustituto transformer de YOLO, con tamaños de tiny a xlarge y exportación a ONNX y TensorRT. También sirve de base para OVLW-DETR, de vocabulario abierto. Su actividad reciente es menor; considéralo una versión de investigación estable.

LibreYOLO ya integra muchos de estos modelos (D-FINE, DEIM, DEIMv2, RT-DETRv2, RT-DETRv4, RTMDet y EdgeCrafter) en una sola API, con YOLO9 y RF-DETR como sus modelos más probados. Para las implementaciones de referencia y los checkpoints de investigación más recientes, los repositorios originales son la fuente de referencia.

## Preguntas frecuentes

**¿Ultralytics YOLO es gratuito para uso comercial?**
Solo bajo AGPL-3.0, que exige publicar el código fuente de la aplicación que crees con él, incluidos los servicios en red. Para el uso comercial con código cerrado se necesita la licencia comercial de pago de Ultralytics. Las alternativas permisivas (MIT, Apache-2.0) evitan esta obligación.

**¿Cuál es la alternativa a Ultralytics con menos restricciones de licencia?**
Para una stack permisiva que puedas desplegar en cualquier sitio: LibreYOLO (código MIT), RF-DETR (Apache-2.0) y YOLOX (Apache-2.0). Todas evitan el copyleft de AGPL.

**¿Qué está sustituyendo a YOLO en 2026?**
Cada vez más, los detectores transformer en tiempo real: RT-DETR, RF-DETR y la línea D-FINE y DEIM. Igualan o superan la precisión de YOLO con una latencia similar en hardware capaz. Las CNN de estilo YOLO siguen ganando en dispositivos edge pequeños, donde esos transformers funcionan mal y carecen de una ruta madura para NCNN o CPU.

**¿Pueden ejecutarse en una Raspberry Pi o una NPU?**
Depende de la exportación. Los detectores CNN como YOLOX y RTMDet se exportan a NCNN y funcionan en una Pi, y algunos (YOLOX) llegan a la NPU Hailo. Los detectores transformer como RF-DETR no; necesitan una GPU o una CPU potente.

## Pruébalo

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRFDETRl.pt")
results = model("image.jpg", save=True)
print(results[0].boxes.xyxy)
print(results[0].boxes.conf)
```

LibreYOLO usa la licencia MIT, funciona en Linux, Mac y Windows y se ejecuta en GPU, Apple Silicon y CPU normal sin cambiar el código. Una sola API cubre RF-DETR, D-FINE, DEIM, YOLOX, YOLO-NAS, RTMDet, RT-DETR, EdgeCrafter y más, para detección, segmentación, pose, clasificación, profundidad, mirada y seguimiento.

Dale una estrella en GitHub: [github.com/LibreYOLO/libreyolo](https://github.com/LibreYOLO/libreyolo) | Documentación: [libreyolo.com/docs](https://www.libreyolo.com/docs/v1.3.1)
