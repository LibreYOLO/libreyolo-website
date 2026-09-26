---
title: RTMDet
families:
  - rtmdet
seo_title: 'RTMDet en LibreYOLO: predecir, entrenar y exportar'
description: >-
  Ejecuta RTMDet en LibreYOLO para detección de objetos y segmentación de
  instancias con RTMDet-Ins. Instala, predice, entrena, valida y exporta bajo
  Apache-2.0.
lead: >-
  RTMDet es un detector de una etapa que predice a partir de un único prior
  basado en puntos por posición de la rejilla, sin anchors, a través de una
  cabeza cuyas convoluciones se comparten entre niveles de características.
  LibreYOLO lo admite para detección y para segmentación de instancias con
  RTMDet-Ins.
keywords:
  - RTMDet
  - detección de objetos python
  - segmentación de instancias
  - RTMDet-Ins
  - detección sin anchors
  - mmdetection
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreRTMDets.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreRTMDets.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Segmentación de instancias
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # El sufijo -seg en el nombre del archivo selecciona la cabeza de
        # máscaras de RTMDet-Ins, así que aquí no hace falta argumento task.
        model = LibreYOLO("LibreRTMDets-seg.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.masks.data.shape)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTMDets.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreRTMDets.pt data=my-dataset.yaml
    - label: Segmentación de instancias
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTMDets-seg.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95(M)"])   # máscaras
        print(metrics["metrics/mAP50-95(B)"])   # boxes
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTMDets.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=300, imgsz=640, batch=16, lr0=0.004,
        )
    - label: CLI
      language: bash
      code: >
        libreyolo train model=LibreRTMDets.pt data=my-dataset.yaml imgsz=640
        epochs=300 batch=16 lr0=0.004
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTMDets.pt")
        model.export(format="onnx", imgsz=640)
        model.export(format="tensorrt", imgsz=640, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreRTMDets.pt format=onnx imgsz=640

        libreyolo export model=LibreRTMDets.pt format=tensorrt imgsz=640
        half=True
    - label: Usar el archivo exportado
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # La factoría enruta según la extensión del archivo, así que un
        artefacto

        # exportado se carga como cualquier checkpoint y devuelve el mismo

        # objeto Results.

        model = LibreYOLO("LibreRTMDets.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: 2f5033bdc1c3c931
---

## Instalación

RTMDet no necesita ningún extra más allá del paquete base.

```bash
pip install libreyolo
```

## Predicción

Los pesos se descargan de Hugging Face en el primer uso y se guardan en la
caché local.

<code-tabs name="predict" />

El objeto `Results` devuelto es el mismo que devuelven todas las familias, así
que cambiar a otro detector es un cambio de una línea. Un nombre de archivo con
`-seg` se resuelve por sí solo a la tarea RTMDet-Ins, y entonces `result.masks`
lleva las máscaras de instancia junto a los boxes. `conf` fija el umbral de
confianza e `iou` el umbral de NMS. Consulta [predicción](/docs/predict) para
fuentes, streaming y manejo de resultados.

## Variantes

Cinco tamaños, de `t` a `x`, comparten una misma arquitectura con una resolución
de entrada común. Esta familia no trae aquí ninguna tabla de benchmarks: compara
los tamaños por el tamaño de archivo de cada checkpoint en la tabla de abajo.

## Entrenamiento

<code-tabs name="train" />

La detección se entrena con `train()`. Los componentes QualityFocalLoss, GIoU y
DynamicSoftLabelAssigner están portados del mmdetection original, y el forward
pass y la exportación a ONNX son bit a bit equivalentes a él, con un
postprocesado que coincide con la salida de mmdet dentro de 0,001 mAP en
subconjuntos de val2017.

Lo que no se ha comprobado, según el propio docstring de `train()`: la
convergencia del fine-tuning en datasets pequeños, la paridad con el paper
entrenando desde cero, el comportamiento multi-GPU, el rendimiento de Mosaic y
MixUp con caché, el cambio estricto al pipeline de dos etapas del original, y
los overrides de weight decay por parámetro que ponen a cero el decay en los
parámetros de norm y bias.

RTMDet-Ins no tiene ruta de entrenamiento. Llamar a `train()` sobre un
checkpoint `-seg`, o con `task="segment"`, lanza `NotImplementedError`; la
segmentación de instancias solo admite inferencia y validación.

`train()` también acepta un argumento `pretrained`, pero el valor nunca se lee
dentro del método: el entrenamiento siempre continúa desde los pesos con los que
se construyó el modelo, así que `pretrained=False` no reinicializa la red.

Si no se toca nada más, el entrenador ejecuta 300 épocas con AdamW a
`lr0=0.004` y `weight_decay=0.05`, un warmup de 1 época sobre un schedule
coseno, y Mosaic y MixUp desactivados durante las últimas 20 épocas.

Consulta [entrenamiento](/docs/train) para datasets, aumento de datos, multi-GPU
y loggers.

## Validación

`val()` devuelve un diccionario de claves `metrics/` que cubren precisión,
recall, mAP 50 y mAP 50-95, medidas contra cualquier dataset en el formato con
el que entrenaste.

<code-tabs name="val" />

Sobre un checkpoint `-seg`, la clave `metrics/mAP50-95` a secas contiene la
puntuación de las máscaras, y la misma ejecución informa además de los boxes
bajo `(B)` y de las máscaras bajo `(M)`, de modo que ambos están disponibles en
una sola pasada.

## Exportación

<export-matrix />

La detección exporta a la mayoría de formatos; la segmentación de instancias no
exporta ahora mismo a ninguno de ellos; la matriz de arriba refleja esa
división. Un artefacto de detección exportado se vuelve a cargar con
`LibreYOLO()` según la extensión del archivo, así que un archivo `.onnx` o
`.engine` se comporta como un checkpoint y devuelve el mismo `Results`. También
se admite ejecutar el grafo en un runtime pelado, sin LibreYOLO instalado, pero
entonces el preprocesado y el postprocesado corren de tu cuenta.

<code-tabs name="export" />

## Checkpoints

Todos los archivos de pesos publicados de esta familia.

<checkpoint-table />

## Licencia

<provenance-box></provenance-box>

## Cita

<citation-block />
