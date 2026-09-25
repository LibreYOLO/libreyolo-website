---
title: "RT-DETR sin Ultralytics: RT-DETR, v2 y v4 con licencia Apache-2.0"
description: "Ejecuta y ajusta RT-DETR, RT-DETRv2 y RT-DETRv4 con pesos Apache-2.0 en una biblioteca MIT: una API para predict, train, val y export."
date: 2026-09-24
author: Xuban
tags: [LibreYOLO, rt-detr, rtdetr, rt-detrv4, object-detection, license, tutorial]
faq:
  - q: "¿RT-DETR es gratis para uso comercial?"
    a: "Sí, si usas una implementación permisiva. El código original de RT-DETR y RT-DETRv2 (lyuwenyu/RT-DETR) y el de RT-DETRv4 (RT-DETRs/RT-DETRv4) tienen licencia Apache-2.0, y LibreYOLO ejecuta las tres versiones con pesos Apache-2.0 en una biblioteca MIT. La implementación de RT-DETR dentro del paquete ultralytics tiene licencia AGPL-3.0, con una licencia Enterprise de pago como alternativa para el uso con código cerrado. Esto es información general, no asesoramiento legal."
  - q: "¿Qué licencia tiene RT-DETRv4?"
    a: "Apache-2.0. El archivo LICENSE de github.com/RT-DETRs/RT-DETRv4 indica Apache-2.0. RT-DETRv4 usa un teacher DINOv3 solo durante el entrenamiento; los pesos publicados del student no contienen parámetros de DINOv3, por lo que la licencia de DINOv3 de Meta no se extiende a ellos."
  - q: "¿Puedo hacer fine-tuning de RT-DETR sin Ultralytics?"
    a: "Sí. LibreYOLO hace fine-tuning de los checkpoints de detección de RT-DETR, RT-DETRv2 y RT-DETRv4 con model.train(), o con libreyolo train en la línea de comandos, a partir de los pesos COCO publicados. Los modelos de cajas orientadas de RT-DETRv2 solo permiten inferencia."
  - q: "¿Qué pesos de RT-DETR incluye LibreYOLO?"
    a: "RT-DETR en r18, r34, r50, r50m, r101, l y x; RT-DETRv2 en r18, r34, r50, r50m y r101; RT-DETRv4 en s, m, l y x, todos para detección en COCO a 640 px. RT-DETRv2 también tiene modelos de cajas orientadas n, s, m, l y x, entrenados con DOTA v1.0 a 1024 px. Todos los archivos tienen licencia Apache-2.0."
---

RT-DETR es un transformer de detección en tiempo real de Baidu. Decodifica un conjunto fijo de consultas en lugar de una cuadrícula densa, así que no hay que ajustar ningún paso de NMS. Si lo buscas, una de las primeras implementaciones que encontrarás está dentro del paquete de Python `ultralytics`. Esto plantea una cuestión de licencia que el modelo en sí nunca tuvo.

## La licencia de RT-DETR depende del repositorio

La licencia corresponde a un repositorio, no a un modelo. RT-DETR tiene varios repositorios y sus licencias no coinciden:

| Repositorio | Incluye | Licencia |
|:---|:---|:---|
| [lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR) | RT-DETR y RT-DETRv2, PyTorch y Paddle | Apache-2.0 |
| [RT-DETRs/RT-DETRv4](https://github.com/RT-DETRs/RT-DETRv4) | RT-DETRv4 | Apache-2.0 |
| [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) | RT-DETR dentro del paquete `ultralytics` | AGPL-3.0 o licencia Enterprise |
| [LibreYOLO](https://github.com/LibreYOLO/libreyolo) | RT-DETR, RT-DETRv2 y RT-DETRv4 | Código MIT, pesos Apache-2.0 |

La implementación de Ultralytics es sólida. Su [página de RT-DETR](https://docs.ultralytics.com/models/rtdetr/) incluye los modelos preentrenados `rtdetr-l.pt` y `rtdetr-x.pt`, con entrenamiento, validación, inferencia y exportación. Se distribuye dentro de un paquete con licencia AGPL-3.0, y Ultralytics vende una licencia Enterprise para los equipos que no quieren publicar el código fuente de todo su proyecto. Si ninguna opción te sirve, sus propios autores ofrecen la arquitectura con licencia Apache-2.0. La [explicación de las licencias de YOLO](/articles/yolo-licenses-explained) cubre el mismo patrón en toda la familia YOLO, y [la guía de licencias comerciales](/articles/yolo-commercial-license) explica qué implica AGPL-3.0 para un producto de código cerrado.

## Ejecuta RT-DETR con LibreYOLO

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO, SAMPLE_IMAGE

model = LibreYOLO("LibreRTDETRr18.pt")  # downloads from Hugging Face on first use
result = model(SAMPLE_IMAGE, save=True)

for box in result.boxes:
    print(box.cls, box.conf, box.xyxy)
```

La misma llamada funciona desde la línea de comandos:

```bash
libreyolo predict model=LibreRTDETRr18.pt source=image.jpg save=True
```

`conf` y `max_det` filtran una decodificación top-k sobre consultas y clases. Se acepta `iou`, pero no se usa porque no hay NMS. El objeto `Results` que se devuelve es el que retorna cada modelo de LibreYOLO, así que cambiar de detector requiere modificar una sola línea.

## RT-DETR, RT-DETRv2 y RT-DETRv4 en un solo cargador

La versión forma parte del nombre del archivo y `LibreYOLO()` selecciona el modelo a partir del checkpoint, así que las tres versiones se cargan de la misma forma:

* **RT-DETR:** `LibreRTDETR` en r18, r34, r50, r50m y r101 (backbones ResNet), y en l y x (HGNetv2).
* **RT-DETRv2:** `LibreRTDETRv2` en r18, r34, r50, r50m y r101. Mantiene la arquitectura de la versión 1 y cambia cómo se muestrea la atención deformable.
* **RT-DETRv4:** `LibreRTDETRv4` en s, m, l y x. Reutiliza la arquitectura de D-FINE, y sus pesos se obtienen destilando un teacher DINOv3 en un student HGNetv2.

Todos los pesos de detección se entrenan con COCO y funcionan a 640 px. Como referencia, los README de los proyectos originales indican 46.5 AP para RT-DETR-R18 y 53.0 AP para RT-DETR-L en COCO, y de 49.8 AP para RT-DETRv4-S a 57.0 AP para RT-DETRv4-X. La [página de documentación de RT-DETR](/docs/models/rt-detr) incluye la tabla de benchmarks de LibreYOLO para cada checkpoint.

RT-DETRv2 también admite cajas orientadas. `LibreRTDETRv2n-obb.pt` hasta `LibreRTDETRv2x-obb.pt` son los checkpoints oficiales de DOTA v1.0, con 15 clases aéreas a 1024 px, convertidos al formato de LibreYOLO:

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRTDETRv2n-obb.pt")
result = model("aerial.png", save=True)
print(result.obb.xywhr)  # (N, 5): cx, cy, w, h, radians
```

La [detección orientada](/docs/tasks/oriented-detection) explica el formato de etiquetas y las métricas.

## Haz fine-tuning de RT-DETR con tus propios datos

El entrenamiento parte de un checkpoint publicado:

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRTDETRr18.pt")
model.train(data="coco128.yaml", epochs=50, batch=4, lr0=1e-4)
```

Para una ejecución real, indica en `data` el YAML de tu propio dataset. Cada versión tiene su propio learning rate predeterminado, y las versiones 1 y 2 fijan el learning rate del backbone en una vigésima parte de `lr0`, siguiendo la receta original. En la CLI se configura Multi-GPU con `device=0,1`, y el fine-tuning de adaptadores [LoRA](/docs/train/lora) se activa con `lora=True`, tras instalar el extra `lora`. Los modelos de cajas orientadas de RT-DETRv2 solo permiten inferencia. Consulta [entrenamiento](/docs/train) para obtener información sobre datasets, aumento de datos y loggers.

## Validación y exportación

```python
metrics = model.val(data="coco128.yaml")
print(metrics["metrics/mAP50-95"])

path = model.export(format="onnx")  # needs: pip install "libreyolo[onnx]"
```

`val()` devuelve un diccionario simple. Los formatos de exportación incluyen ONNX, TorchScript, ExecuTorch, TensorRT y OpenVINO; la matriz de exportación de la [página del modelo](/docs/models/rt-detr) muestra cuáles se han validado para cada tarea. Un archivo `.onnx` o `.engine` exportado se carga de nuevo con `LibreYOLO()` y devuelve el mismo objeto `Results`. Consulta [exportación](/docs/export) para ver las guías de cada formato.

## Cuándo conviene seguir usando los repositorios originales

Si necesitas reproducir un resultado de un artículo con las configuraciones exactas de sus autores, quieres la versión de Paddle o quieres ejecutar por tu cuenta la destilación del teacher DINOv3 de RT-DETRv4, usa los repositorios originales. LibreYOLO hace fine-tuning del student v4 publicado, no ejecuta el teacher. Y si tu proyecto ya tiene licencia AGPL-3.0 o está cubierto por una licencia Enterprise de Ultralytics, no hay ningún motivo relacionado con la licencia para cambiar.

## Una nota sobre la licencia

El código de LibreYOLO tiene licencia MIT. Todos los archivos de pesos de RT-DETR tienen licencia Apache-2.0, y cada repositorio de pesos de Hugging Face incluye el LICENSE y el NOTICE originales, que Apache-2.0 exige conservar al redistribuir los pesos. Los pesos que entrenes con tus propios datos son tuyos. RT-DETRv4 usa DINOv3 solo como teacher durante el entrenamiento, y los modelos student publicados no contienen parámetros de DINOv3. Esto es información general, no asesoramiento legal. Comprueba los archivos LICENSE vigentes antes de distribuir.

## Pruébalo

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRTDETRv4s.pt")
result = model("image.jpg", save=True)
print(result.boxes.xyxy)
```

LibreYOLO tiene licencia MIT, funciona en Linux, Mac y Windows, y se ejecuta en GPU, Apple Silicon y CPU sin cambiar el código.

[GitHub](https://github.com/LibreYOLO/libreyolo) | [Documentación de RT-DETR](/docs/models/rt-detr)
