---
title: Detección de objetos
seo_title: Detección de objetos en LibreYOLO
description: >-
  Detecta objetos como bounding boxes alineados a los ejes en LibreYOLO: las
  familias que cubren la tarea, el formato de etiquetas y las llamadas de
  predicción, entrenamiento, validación y exportación.
lead: >-
  La detección de objetos localiza cada instancia de objeto en una imagen y
  devuelve un rectángulo alineado a los ejes, una etiqueta de clase y una
  puntuación para cada una. La clave de la tarea es detect.
keywords:
  - detección de objetos python
  - detectar objetos en imagen
  - bounding box python
  - librería detección de objetos licencia MIT
  - alternativa a YOLO
  - entrenar detector de objetos dataset propio
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9t.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(result.names[int(box.cls)], float(box.conf), box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9t.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: 'Otra familia, la misma llamada'
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # La factory decide según el checkpoint, y todos los detectores
        devuelven

        # el mismo objeto Results, así que cambiar de familia es un cambio de
        una línea.

        model = LibreYOLO("LibreDFINEn.pt")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy.shape)
    - label: Vídeo y streams
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Cualquier fuente que acepte la librería: archivo, carpeta, URL,
        # índice de webcam, stream RTSP o una lista .streams.
        for result in model.predict("clip.mp4", stream=True, save=True):
            print(len(result.boxes))
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # coco128.yaml descarga una muestra de 128 imágenes en el primer uso.
        # Apunta data al YAML de tu propio dataset para un entrenamiento real.
        model.train(data="coco128.yaml", epochs=50, imgsz=640, batch=8)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9t.pt data=coco128.yaml \
          epochs=50 imgsz=640 batch=8
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreYOLO9t.pt data=coco128.yaml \
          epochs=50 device=0,1 batch=-1
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # val() devuelve un dict plano, no un objeto.
        metrics = model.val(data="coco128.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"], metrics["metrics/mAP75"])
        print(metrics["metrics/AR100"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLO9t.pt data=coco128.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")
        model.export(format="onnx", imgsz=640)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreYOLO9t.pt format=onnx imgsz=640
    - label: Usar el archivo exportado
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # La factory decide según la extensión del archivo, así que un artefacto

        # exportado se carga como un checkpoint y devuelve el mismo objeto
        Results.

        model = LibreYOLO("LibreYOLO9t.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: c735b6e3de78dd2b
---

## Definición

La detección de objetos responde dónde está cada objeto y qué es. Entra una
imagen y sale una fila por instancia: cuatro números para el rectángulo, un
índice de clase y una puntuación. No incluye nada sobre la forma a nivel de
píxel, la orientación ni las partes, que es lo que la separa de la
[segmentación de instancias](/docs/tasks/instance-segmentation),
los [bounding boxes orientados](/docs/tasks/oriented-detection) y
la [pose](/docs/tasks/pose-estimation).

`detect` es la clave canónica de la tarea y la opción por defecto: un checkpoint
cuyo nombre de archivo no lleva sufijo de tarea se carga como detector.

`predict()` rellena `result.boxes`. `.xyxy` da las esquinas en píxeles sobre el
lienzo de la imagen original, `.conf` la puntuación y `.cls` el índice de clase
dentro de `result.names`. `.xywh`, `.xyxyn` y `.xywhn` son vistas derivadas de
las mismas filas, y `.id` lleva un id de seguimiento cuando hay un tracker
acoplado. Iterar un objeto `Boxes` produce slices de una fila, así que
`box.cls`, `box.conf` y `box.xyxy` funcionan por detección.

## Modelos

Doce familias entrenan y predicen: [YOLOv9](/docs/models/yolov9),
[RF-DETR](/docs/models/rf-detr), [EdgeCrafter](/docs/models/edgecrafter),
[RT-DETR](/docs/models/rt-detr), [D-FINE](/docs/models/d-fine),
[DEIM](/docs/models/deim), [Dome-DETR](/docs/models/dome-detr),
[YOLO-NAS](/docs/models/yolo-nas),
[YOLOX](/docs/models/yolox), [YOLOv7](/docs/models/yolov7),
[RTMDet](/docs/models/rtmdet) y [PicoDet](/docs/models/picodet). YOLOv9 y
RF-DETR son las dos familias insignia, y las novedades llegan primero a ellas.
RF-DETR necesita su propio extra, `pip install "libreyolo[rfdetr]"`; el resto
funciona con el paquete base.

Once más predicen, validan y exportan, pero su `train()` lanza
`NotImplementedError`: [LW-DETR](/docs/models/lw-detr),
[DETR](/docs/models/detr), [Deformable DETR](/docs/models/deformable-detr),
[DINO-DETR](/docs/models/dino-detr), [Faster R-CNN](/docs/models/faster-rcnn),
[Mask R-CNN](/docs/models/mask-rcnn), [FCOS](/docs/models/fcos),
[RetinaNet](/docs/models/retinanet), [SSD](/docs/models/ssd),
[CenterNet](/docs/models/centernet) y
[EfficientDet](/docs/models/efficientdet).

El linaje Darknet, [YOLOv1](/docs/models/yolov1),
[YOLOv2](/docs/models/yolov2), [YOLOv3](/docs/models/yolov3) y
[YOLOv4](/docs/models/yolov4), se conserva como una pieza de museo congelada:
predecir, validar y exportar funcionan; entrenar, no.

Un grupo aparte toma su lista de clases en tiempo de ejecución en lugar de
tomarla del checkpoint, de modo que detecta nombres nunca vistos durante el
entrenamiento:
[Grounding DINO](/docs/models/grounding-dino), [OWLv2](/docs/models/owlv2),
[OMDet-Turbo](/docs/models/omdet-turbo) y [OV-DEIM](/docs/models/ov-deim),
más las familias de visión y lenguaje
[Florence-2](/docs/models/florence-2), [Kosmos-2](/docs/models/kosmos-2),
[Qwen3-VL](/docs/models/qwen3-vl), [SmolVLM2](/docs/models/smolvlm2),
[InternVL3](/docs/models/internvl3), [LFM2-VL](/docs/models/lfm2-vl),
[LocateAnything](/docs/models/locate-anything),
[SenseNova-Vision](/docs/models/sensenova-vision) y
[LibreMODUS](/docs/models/libremodus). Estas se cargan a través de su propia
factory y sus extras; cada página de modelo lleva la llamada exacta.

## Predicción

Los pesos se descargan de Hugging Face en el primer uso y se cachean localmente.

<code-tabs name="predict" />

`conf` fija el umbral de confianza y `max_det` limita el número de filas.
`iou` es el umbral de NMS, así que solo tiene efecto en una familia que ejecute
NMS; RF-DETR y la cabeza end-to-end de YOLOv9 decodifican un conjunto fijo de
predicciones y lo ignoran. Consulta la [predicción](/docs/predict) para las
fuentes, el streaming y el manejo de resultados.

## Formato del dataset

Un archivo de etiquetas `.txt` por imagen, que se localiza sustituyendo
`images` por `labels` en la ruta de la imagen y cambiando la extensión.

```text
dataset/
  data.yaml
  images/
    train/000001.jpg
    val/000101.jpg
  labels/
    train/000001.txt
    val/000101.txt
```

Cada fila tiene exactamente cinco campos, un índice de clase seguido de un
bounding box normalizado de centro y tamaño:

```text
<class_id> <cx> <cy> <w> <h>
```

Las coordenadas son floats en `[0, 1]`, relativas al ancho y alto de la imagen
original. `w` y `h` deben ser positivos. Un archivo de etiquetas ausente o
vacío significa que la imagen no tiene objetos. Las filas no llevan confianza
ni id de seguimiento.

El YAML nombra los splits y las clases:

```yaml
path: dataset
train: images/train
val: images/val
names:
  0: person
  1: bicycle
```

`train` y `val` pueden ser directorios de imágenes, archivos `.txt` con listas
de imágenes, o listas de cualquiera de los dos. `nc` es opcional y debe
coincidir con `names` cuando está presente. El JSON nativo de COCO también
funciona: añade un mapeo `annotations` de nombre de split a archivo JSON, y la
ruta del split pasa entonces a indicar la raíz de las imágenes. Cuando `names`
está presente define los ids de las etiquetas, así que los nombres de categoría
del JSON tienen que coincidir con él.

## Entrenamiento

<code-tabs name="train" />

`epochs`, `imgsz`, `batch` y `lr0` son los argumentos que se tocan primero.
`lr0` es el que no se traslada entre familias: un valor que un detector
convolucional tolera hará divergir a uno transformer, así que toma el valor de
la página del modelo y no del ejemplo de otra familia. Una familia también
puede ignorar un argumento por completo, y su página lista cuáles. Consulta el
[entrenamiento](/docs/train) para los datasets, el aumento de datos, el
multi-GPU y los loggers.

## Validación

`val()` devuelve un diccionario plano de claves `metrics/`, calculadas con la
evaluación COCO sobre el split que nombra `val` en el YAML del dataset.

<code-tabs name="val" />

`metrics/mAP50-95` es la precisión media (mean average precision) promediada
sobre umbrales de IoU de 0.50 a 0.95, y es la cifra principal. `metrics/mAP50`
y `metrics/mAP75` son las versiones de umbral único. `metrics/mAP_small`,
`metrics/mAP_medium` y `metrics/mAP_large` desglosan el mismo promedio por área
del objeto, y `metrics/AR1`, `metrics/AR10`, `metrics/AR100`,
`metrics/AR_small`, `metrics/AR_medium` y `metrics/AR_large` son las cifras
equivalentes de recall promedio. `metrics/AR_max_det` y `metrics/max_det`
registran el límite de detecciones que usó la ejecución.

Lee `metrics/precision` y `metrics/recall` con cuidado en esta tarea. Se
mantienen por retrocompatibilidad y son alias, no un punto de operación:
`metrics/precision` contiene el mismo valor que `metrics/mAP50-95`, y
`metrics/recall` el mismo valor que `metrics/AR100`. Graficarlas como un par de
precisión y recall reporta el mismo número dos veces. Cuatro claves también se
repiten con el sufijo `(B)`, de box, para que una clave de detección se lea
igual en un modelo que también predice máscaras: `metrics/mAP50-95(B)`,
`metrics/mAP50(B)`, `metrics/precision(B)` y `metrics/recall(B)`.

## Exportación

<code-tabs name="export" />

Un artefacto exportado se vuelve a cargar a través de `LibreYOLO()` por la
extensión del archivo, así que un archivo `.onnx` o `.engine` se comporta como
un checkpoint y devuelve el mismo `Results`. La cobertura de formatos varía
según la familia; la matriz de cada página de modelo se genera a partir del
conjunto validado en lugar de escribirse a mano. Consulta
[exportación y despliegue](/docs/export) para los formatos, sus extras y sus
restricciones.
