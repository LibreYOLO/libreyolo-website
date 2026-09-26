---
title: YOLOX
families:
  - yolox
seo_title: 'YOLOX: predecir, entrenar y exportar bajo Apache-2.0'
description: >-
  Usa YOLOX en LibreYOLO para detección de objetos: instala, predice, entrena,
  valida y exporta bajo Apache-2.0.
lead: >-
  YOLOX es un detector de una sola etapa sin anchors, con una cabeza desacoplada
  de clasificación y regresión, entrenado con asignación de etiquetas SimOTA.
  LibreYOLO lo soporta para detección.
keywords:
  - YOLOX
  - detección de objetos
  - detección anchor-free
  - detección sin anchors
  - cabeza desacoplada
  - SimOTA
  - detección de objetos en tiempo real
  - yolox python
  - entrenar yolox
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLOXs.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreYOLOXs.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLOXs.pt")

        model.train(data="my-dataset.yaml", epochs=300, imgsz=640, batch=16,
        lr0=0.01)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLOXs.pt data=my-dataset.yaml \
          epochs=300 imgsz=640 batch=16 lr0=0.01
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLOXs.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLOXs.pt data=my-dataset.yaml
    - label: Contra COCO
      language: bash
      code: |
        # El yaml de COCO incluido lleva un script de descarga embebido, así que
        # necesita permiso explícito salvo que el dataset ya esté en local.
        libreyolo val model=LibreYOLOXn.pt data=coco.yaml imgsz=416 \
          allow_download_scripts=True
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLOXs.pt")
        model.export(format="onnx", imgsz=640)
        model.export(format="tensorrt", imgsz=640, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreYOLOXs.pt format=onnx imgsz=640

        libreyolo export model=LibreYOLOXs.pt format=tensorrt imgsz=640
        half=True
    - label: Usar el archivo exportado
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # La factoría enruta según el sufijo del archivo, así que un artefacto

        # exportado se carga como cualquier checkpoint y devuelve el mismo
        objeto Results.

        model = LibreYOLO("LibreYOLOXs.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: f5ab735a29f85a95
---

## Instalación

YOLOX no necesita ningún extra más allá del paquete base.

```bash
pip install libreyolo
```

## Predicción

Los pesos se descargan de Hugging Face en el primer uso y se guardan en la caché
local.

<code-tabs name="predict" />

El objeto `Results` devuelto es el mismo que devuelve cada familia, así que
cambiar a otro detector es un cambio de una línea. `conf` fija el umbral de
confianza e `iou` el umbral de NMS que se aplica sobre las tres escalas de
predicción desacopladas. Consulta [predicción](/docs/predict) para fuentes,
streaming y manejo de resultados.

## Variantes

Seis tamaños comparten el mismo backbone CSP y el mismo neck PAFPN. Los dos más
pequeños, `n` y `t`, funcionan con una resolución de entrada fija menor que los
otros cuatro; la tabla de benchmark de abajo recoge la cifra exacta de cada uno.

<benchmark-table task="detect" />

<va-embed />

## Entrenamiento

<code-tabs name="train" />

Si no se toca nada, el entrenador ejecuta 300 épocas con `lr0=0.01`, SGD con
momentum 0.9, un warmup de 5 épocas y el aumento de datos de mosaic y mixup
desactivado durante las últimas 15 épocas. `train()` también acepta un argumento
`pretrained`, pero su valor nunca se lee dentro del método: el entrenamiento
continúa siempre desde los pesos con los que se construyó el modelo, así que
`pretrained=False` no reinicializa la red.

`imgsz` toma por defecto un valor fijo de la configuración base de
entrenamiento, no la resolución nativa del checkpoint cargado. Eso afecta en
concreto a los checkpoints `n` y `t`: seguir entrenando cualquiera de los dos
sin fijar `imgsz` explícitamente lo sube al valor por defecto, mayor, en lugar
del tamaño menor con el que se publicó.

Consulta [entrenamiento](/docs/train) para datasets, aumento de datos, multi-GPU
y loggers.

## Validación

`val()` devuelve un diccionario de claves `metrics/` que cubren precisión,
recall, mAP 50 y mAP 50-95, medidas contra cualquier dataset en el formato con
el que entrenaste.

<code-tabs name="val" />

## Exportación

<export-matrix />

Un artefacto exportado se vuelve a cargar a través de `LibreYOLO()` por su
sufijo de archivo, así que un archivo `.onnx` o `.engine` se comporta como un
checkpoint y devuelve el mismo `Results`. Ejecutar el grafo en un runtime
desnudo, sin LibreYOLO instalado, también está soportado, pero entonces el
preprocesamiento y el postprocesamiento corren de tu cuenta. Una exportación a
CoreML puede meter NMS dentro del grafo con `nms=True`; YOLOX y YOLOv9 son las
dos únicas familias que ese flag acepta actualmente.

<code-tabs name="export" />

## Checkpoints

Todos los archivos de pesos publicados de esta familia.

<checkpoint-table />

## Licencia

<provenance-box></provenance-box>

## Cita

<citation-block />
