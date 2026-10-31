---
title: YOLOv9
families:
  - yolo9
seo_title: 'YOLOv9: predecir, entrenar y exportar bajo MIT'
description: >-
  Ejecuta YOLOv9 en LibreYOLO, incluida la cabeza end-to-end sin NMS y la cabeza
  de stride 4 para objetos pequeños. Instala, predice, entrena, valida y
  exporta.
lead: >-
  Un detector convolucional de una sola etapa: una pasada puntúa una rejilla
  densa de boxes y NMS descarta los duplicados. LibreYOLO incluye tres
  variantes, una de ellas sin paso de NMS.
keywords:
  - YOLOv9
  - YOLO9
  - detección de objetos
  - detección sin NMS
  - detección end-to-end
  - detección de objetos pequeños
  - yolov9 python
  - entrenar yolov9
  - programmable gradient information
  - GELAN
last_verified: 1.6.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Sin NMS
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # La misma llamada, distinto checkpoint. La cabeza end-to-end devuelve
        sus

        # propias predicciones mejor puntuadas, así que no se ejecuta NMS y iou
        se ignora.

        model = LibreYOLO("LibreYOLO9E2Es.pt")

        result = model(SAMPLE_IMAGE, conf=0.25, max_det=300)


        print(len(result.boxes))
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(data="my-dataset.yaml", epochs=100, imgsz=640, batch=16)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          epochs=100 imgsz=640 batch=16
    - label: Objetos pequeños
      language: python
      code: >
        from libreyolo import LibreYOLO9P2


        # La variante de stride 4 no tiene checkpoint COCO propio, así que
        indica

        # uno de detección base: su backbone y su neck se cargan sin cambios y
        la

        # torre de la cabeza de stride 4 parte de una inicialización aleatoria.

        model = LibreYOLO9P2(None, size="s")

        model.train(data="my-dataset.yaml", epochs=100,
        pretrained="LibreYOLO9s.pt")
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLO9s.pt data=my-dataset.yaml
    - label: Contra COCO
      language: bash
      code: |
        # El yaml de COCO incluido lleva un script de descarga embebido, así que
        # necesita permiso explícito salvo que el dataset ya esté en local.
        libreyolo val model=LibreYOLO9c.pt data=coco.yaml imgsz=640 \
          allow_download_scripts=True
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.export(format="onnx", imgsz=640)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreYOLO9s.pt format=onnx imgsz=640
    - label: Con NMS en el grafo
      language: bash
      code: |
        libreyolo export model=LibreYOLO9s.pt format=onnx nms=True \
          conf=0.25 iou=0.45 max_det=300
    - label: Usar el archivo exportado
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # La factoría enruta según el sufijo del archivo, así que un artefacto

        # exportado se carga como cualquier checkpoint y devuelve el mismo
        objeto Results.

        model = LibreYOLO("LibreYOLO9s.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: 624f3e70d2937683
---

## Instalación

YOLOv9 no necesita ningún extra más allá del paquete base.

```bash
pip install libreyolo
```

## Predicción

Los pesos se descargan de Hugging Face en el primer uso y se guardan en la caché
local.

<code-tabs name="predict" />

El objeto `Results` devuelto es el mismo que devuelve cada familia, así que
cambiar a otro detector es un cambio de una línea. En los modelos base y de
stride 4, `conf` fija el umbral de confianza e `iou` el umbral de NMS. El modelo
end-to-end no ejecuta NMS e ignora `iou`, así que `conf` y `max_det` son lo que
da forma a su salida. Consulta [predicción](/docs/predict) para fuentes,
streaming y manejo de resultados.

## Variantes

Tres variantes comparten un backbone. Las tres solo detectan, y aceptan los
mismos argumentos.

El modelo base predice en tres escalas de características y elimina los boxes
duplicados con NMS.

El modelo end-to-end conserva esa cabeza y añade junto a ella una rama de
emparejamiento uno a uno. La inferencia lee únicamente la rama uno a uno y toma
sus predicciones mejor puntuadas, así que no se ejecuta NMS. Elígelo cuando el
runtime al que despliegas no tenga operador de NMS.

El modelo de stride 4 aflora un nivel más arriba del backbone, extiende el neck
hasta él y predice en cuatro escalas en lugar de tres. La escala extra es para
objetos que cubren pocos píxeles; el único checkpoint publicado para él está
entrenado con imágenes aéreas. Los checkpoints de detección base se transfieren
a él: el backbone y el neck se cargan sin cambios, las tres torres de cabeza
preentrenadas se desplazan un puesto hacia arriba y la torre de stride 4 parte
de una inicialización aleatoria.

<benchmark-table task="detect" />

<va-embed />

## Entrenamiento

<code-tabs name="train" />

`pretrained` decide desde dónde parte la ejecución. Pasa `True` para cargar el
checkpoint publicado del mismo modelo y tamaño, o un nombre o una ruta para
cualquier otra cosa. Los tensores cuya forma no coincide se omiten en lugar de
rechazarse, y la ejecución registra cuántos se cargaron, así que un checkpoint
entrenado con un número de clases distinto sigue siendo un punto de partida
utilizable.

El modelo de stride 4 no tiene checkpoint COCO publicado propio, así que `True`
se resuelve ahí en un archivo que no existe y la descarga falla. Indica en su
lugar un checkpoint de detección base.

Consulta [entrenamiento](/docs/train) para datasets, aumento de datos,
multi-GPU y loggers.

Los nuevos fine-tunings de detección estándar activan una rama PGI exclusiva del entrenamiento con `aux_weight=0.25`. `max_labels=300`; el momentum de SGD aumenta de 0.8 a 0.937 durante tres épocas de calentamiento. Los checkpoints antiguos de una sola cabeza se reanudan con ese grafo. La predicción y la exportación usan la cabeza principal. `letterbox_pad=None` hereda la marca del checkpoint: los pesos sin marca usan `topleft`, mientras que las nuevas conversiones oficiales registran `center`.

El mosaic de YOLO9 y YOLOX prefiere imágenes acompañantes con anotaciones y realiza un máximo de 20 intentos; MixUp de YOLO9 usa la misma política. Consulta [histogramas de eventos](/docs/train/event-histograms) para perfiles de entrada que no sean RGB.

## Validación

`val()` devuelve un diccionario de claves `metrics/` que cubren precisión,
recall, mAP 50 y mAP 50-95, medidas contra cualquier dataset en el formato con
el que entrenaste.

<code-tabs name="val" />

## Exportación

<export-matrix />

Una marca vale para las tres variantes: donde difieren, la matriz recoge la más
débil de las tres.

Un artefacto exportado se vuelve a cargar a través de `LibreYOLO()` por su
sufijo de archivo, así que un archivo `.onnx` o `.engine` se comporta como un
checkpoint y devuelve el mismo `Results`. Ejecutar el grafo en un runtime
desnudo, sin LibreYOLO instalado, también está soportado, pero entonces el
preprocesamiento y el postprocesamiento corren de tu cuenta.

Para el modelo de detección base, la mitad de postprocesamiento puede moverse
al grafo. `nms=True` en una exportación a ONNX mete la supresión dentro del
modelo, y la primera salida pasa a ser un tensor fijo `(1, max_det, 6)` cuyas
filas son `x1, y1, x2, y2, score, class`, rellenadas con ceros más allá del
número de detecciones. Ese grafo es de batch 1 y no lleva ejes dinámicos. Los
modelos end-to-end y de stride 4 no aceptan el flag.

Cada formato instala un extra distinto y acepta unos pocos argumentos propios.
Ambas cosas están en la página de ese formato.

<code-tabs name="export" />

[TFLite INT8](/docs/export/tflite) usa `int8=True` con datos de calibración.

## Checkpoints

Todos los archivos de pesos publicados de esta familia.

<checkpoint-table />

## Licencia

<provenance-box>

El checkpoint aéreo de stride 4 predice las clases de VisDrone. Usa la licencia declarada por el editor y registrada en el repositorio de sus pesos.

</provenance-box>

## Cita

<citation-block />
