---
title: RF-DETR
families:
  - rfdetr
seo_title: 'RF-DETR: entrena, haz fine-tuning y exporta bajo MIT'
description: >-
  Usa RF-DETR en LibreYOLO para detección, segmentación de instancias, pose y
  cajas orientadas. Instala, predice, entrena, valida y exporta, todo con
  licencia MIT.
lead: >-
  Un transformer de detección que predice un conjunto fijo de objetos en lugar
  de una rejilla densa, por lo que no necesita NMS en inferencia. LibreYOLO lo
  soporta para cuatro tareas.
keywords:
  - RF-DETR
  - transformer de detección en tiempo real
  - DETR
  - detección de objetos
  - segmentación de instancias
  - estimación de pose
  - bounding boxes orientados
last_verified: 1.5.0
hero:
  src: /showcase/parkour-detection.mp4
  poster: /showcase/parkour-detection-poster.jpg
  caption: 'LibreRFDETRs, detección en video a 512 px.'
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreRFDETRs.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreRFDETRs.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Vídeo
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs.pt")

        # Cualquier fuente que acepta la biblioteca: archivo, carpeta, URL,
        # índice de webcam, stream RTSP o una lista .streams
        for result in model.predict("clip.mp4", stream=True, save=True):
            print(len(result.boxes))
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreRFDETRs.pt")

        model.train(data="my-dataset.yaml", epochs=50, imgsz=512, batch=8,
        lr0=1e-4)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreRFDETRs.pt data=my-dataset.yaml \
          epochs=50 imgsz=512 batch=8 lr0=1e-4
    - label: LoRA
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs.pt")
        model.train(data="my-dataset.yaml", epochs=50, lora=True)
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreRFDETRs.pt data=my-dataset.yaml \
          epochs=50 device=0,1 batch=-1
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs.pt")

        # val() devuelve un dict plano, no un objeto
        metrics = model.val(data="my-dataset.yaml", imgsz=512)

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreRFDETRs.pt data=my-dataset.yaml imgsz=512
    - label: Contra COCO
      language: bash
      code: |
        # El yaml de COCO incluido lleva un script de descarga embebido, así
        # que necesita permiso explícito salvo que el dataset ya esté en local.
        libreyolo val model=LibreRFDETRn.pt data=coco.yaml imgsz=384 \
          allow_download_scripts=True
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs.pt")
        model.export(format="onnx", imgsz=512)
        model.export(format="tensorrt", imgsz=512, half=True)

        # Argumentos aceptados para todos los formatos:
        #
        #   format    "onnx" | "torchscript" | "executorch" | "tensorrt"
        #             | "openvino" | "paddle" | "mnn" | "rknn" | "ncnn"
        #             | "tflite" | "coreml" | "coreai".
        #             "engine" es un alias de tensorrt, "litert" de tflite.
        #   imgsz     int, o (alto, ancho). Por defecto, la resolución nativa
        #             del checkpoint.
        #   batch     int, por defecto 1.
        #   half      bool, exporta en FP16. Por defecto False.
        #   int8      bool, exporta en INT8. Por defecto False. Necesita `data`.
        #   data      ruta a un YAML de dataset, usado para calibrar int8.
        #   fraction  float, fracción del conjunto de calibración a usar.
        #             Por defecto 1.0.
        #   dynamic   bool, ejes dinámicos. Por defecto True.
        #   simplify  bool, ejecuta la simplificación del grafo ONNX.
        #             Por defecto True.
        #   opset     int, opset de ONNX. Se elige por familia si no se indica.
        #   device    str, dispositivo en el que trazar. Por defecto, el del
        #             modelo.
        #   output_path  str, por defecto un nombre derivado del checkpoint.
        #   verbose   bool, por defecto False.
        #   allow_download_scripts  bool, por defecto False. Permite Python
        #             embebido en un YAML de dataset que haya que descargar.
        #
        # Algunos formatos aceptan argumentos extra propios, como una
        # plataforma objetivo de RKNN. Están documentados en la página de
        # cada formato.
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreRFDETRs.pt format=onnx imgsz=512

        libreyolo export model=LibreRFDETRs.pt format=tensorrt imgsz=512
        half=True
    - label: Usar el archivo exportado
      language: python
      code: |
        from libreyolo import LibreYOLO

        # La factoría enruta según la extensión del archivo, así que un
        # artefacto exportado se carga como cualquier checkpoint y devuelve
        # el mismo objeto Results.
        model = LibreYOLO("LibreRFDETRs.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
    - label: Sin LibreYOLO
      language: python
      code: >
        import numpy as np

        import onnxruntime as ort


        # Ejecutar el grafo directamente implica hacer tu propio preprocesado

        # y postprocesado. Inspecciona la firma antes de conectar nada.

        session = ort.InferenceSession("LibreRFDETRs.onnx")

        name = session.get_inputs()[0].name

        outputs = session.run(None, {name: np.zeros((1, 3, 512, 512),
        dtype=np.float32)})


        for meta, array in zip(session.get_outputs(), outputs):
            print(meta.name, array.shape)
source_hash: 8c464aa759131694
---

## Instalación

RF-DETR necesita su propio extra, que instala `transformers` para el backbone.

```bash
pip install "libreyolo[rfdetr]"
```

## Predicción

Los pesos se descargan de Hugging Face en el primer uso y se guardan en la
caché local.

<code-tabs name="predict" />

El objeto `Results` devuelto es el mismo que devuelven todas las familias, así
que cambiar a otro detector es un cambio de una línea. `conf` y `max_det`
filtran la selección de queries; no hay un paso de NMS que ajustar. Consulta
[predicción](/docs/predict) para fuentes, streaming y manejo de resultados.

## Variantes

Cuatro tamaños, y cuatro tareas que comparten una misma arquitectura: la
segmentación, la pose y las cajas orientadas reutilizan el decoder de detección
con una cabeza distinta, así que aceptan los mismos argumentos. Los tamaños
tienen un número de parámetros similar y se diferencian sobre todo en la
resolución de entrada.

<benchmark-table task="detect" />

<va-embed />

## Entrenamiento

El entrenamiento parte de un checkpoint publicado, para las cuatro tareas.
RF-DETR incluye `pretrained` entre los argumentos que su trainer nativo ignora,
así que pasar `pretrained=False` no te da aquí un modelo inicializado
aleatoriamente.

<code-tabs name="train" />

Dos argumentos importan aquí más que en un detector CNN. Mantén `lr0` en `1e-4`
o por debajo, porque los detectores transformer divergen con learning rates que
un modelo YOLO tolera. Deja `imgsz` en la resolución nativa del checkpoint salvo
que tengas un motivo para cambiarla. La entrada debe ser divisible exactamente
por el tamaño de patch del backbone multiplicado por el número de ventanas;
LibreYOLO lo comprueba antes de empezar la ejecución e indica los tamaños
válidos más cercanos.

Consulta [entrenamiento](/docs/train) para datasets, aumento de datos
(data augmentation), multi-GPU y loggers.

## Validación

`val()` devuelve un diccionario de claves `metrics/` que cubren precisión,
recall, mAP 50 y mAP 50-95, medidas contra cualquier dataset en el formato con
el que entrenaste.

<code-tabs name="val" />

## Exportación

<export-matrix />

Un artefacto exportado se vuelve a cargar con `LibreYOLO()` según su extensión
de archivo, así que un archivo `.onnx` o `.engine` se comporta como un
checkpoint y devuelve el mismo `Results`. Ejecutar el grafo en un runtime pelado,
sin LibreYOLO instalado, también está soportado, pero entonces el preprocesado y
el postprocesado corren de tu cuenta.

<code-tabs name="export" />

## Checkpoints

Todos los archivos de pesos publicados de esta familia.

<checkpoint-table />

## Licencia

<provenance-box></provenance-box>

## Cita

<citation-block />
