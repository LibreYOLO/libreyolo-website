---
title: PicoDet
families: [picodet]
seo_title: "PicoDet en LibreYOLO: predecir, entrenar y exportar"
description: "Ejecuta PicoDet en LibreYOLO para detección de objetos en móvil. Instala, predice, entrena, valida y exporta con licencia Apache-2.0."
lead: "PicoDet es un detector de una etapa pensado para CPUs de móvil y edge: un backbone ESNet, un neck CSP-PAN y una cabeza compartida con Generalized Focal Loss. LibreYOLO lo soporta para detección."
keywords: [PicoDet, PP-PicoDet, "detección de objetos", "detección de objetos en móvil", "detección de objetos en dispositivos edge", "detector ligero", ESNet, "Generalized Focal Loss"]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibrePICODETs.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibrePICODETs.pt source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePICODETs.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibrePICODETs.pt data=my-dataset.yaml
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePICODETs.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=300, batch=16, lr0=0.01,
        )
    - label: CLI
      language: bash
      code: |
        # Merece la pena fijar imgsz: el CLI usa 640 por defecto, mientras que
        # el checkpoint s es nativo a 320.
        libreyolo train model=LibrePICODETs.pt data=my-dataset.yaml imgsz=320 epochs=300 batch=16 lr0=0.01
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePICODETs.pt")
        model.export(format="onnx", imgsz=320)
        model.export(format="tensorrt", imgsz=320, half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibrePICODETs.pt format=onnx imgsz=320
        libreyolo export model=LibrePICODETs.pt format=tensorrt imgsz=320 half=True
    - label: Usar el archivo exportado
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La factoría enruta según la extensión del archivo, así que un
        # artefacto exportado se carga como cualquier checkpoint y devuelve el
        # mismo objeto Results.
        model = LibreYOLO("LibrePICODETs.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
---

## Instalación

PicoDet no necesita nada más allá del paquete base.

```bash
pip install libreyolo
```

## Predicción

Los pesos se descargan de Hugging Face en el primer uso y se guardan en la
caché local.

<code-tabs name="predict" />

El objeto `Results` devuelto es el mismo que devuelven todas las familias, así
que cambiar a otro detector es un cambio de una línea. `conf` fija el umbral de
confianza y `iou` el umbral de NMS. Consulta [predicción](/docs/predict) para
fuentes, streaming y manejo de resultados.

## Variantes

Tres tamaños, cada uno a su propia resolución de entrada fija: `s` el más
pequeño y `l` el más grande. La resolución crece con el tamaño, así que los
checkpoints más grandes también son más caros de ejecutar por imagen, además de
llevar más parámetros.

<benchmark-table task="detect" />

<va-embed />

## Entrenamiento

<code-tabs name="train" />

Los componentes de la loss y el assigner siguen la receta de upstream: VFL, DFL,
GIoU y SimOTA, con ponderación por calidad de clasificación y targets de VFL con
IoU dinámico. La inferencia es equivalente bit a bit a la de upstream sobre el
mismo checkpoint.

Lo que no se ha comprobado, según el propio docstring de `train()`: la
convergencia sobre un dataset completo, el comportamiento multi-GPU y cualquier
aumento de datos (data augmentation) más allá del volteo horizontal. El
checkpoint `s` a su resolución nativa de 320 tampoco ha superado de forma fiable
el mínimo de precisión de LibreYOLO en el fixture de 30 imágenes y dos clases
con el que la biblioteca prueba los fine-tunes pequeños. Ese tamaño encaja mejor
a escala de COCO completo.

`train()` también acepta un argumento `pretrained`, pero el valor nunca se lee
dentro del método: el entrenamiento siempre continúa desde los pesos con los que
se construyó el modelo, así que `pretrained=False` no reinicializa la red. Si
dejas `imgsz` sin fijar en Python, toma la resolución nativa del checkpoint
cargado: 320 para `s`, 416 para `m` y 640 para `l`. El CLI siempre envía un
`imgsz`, con 640 por defecto, así que fíjalo ahí para que coincida con el
checkpoint.

Si no se toca nada más, el trainer ejecuta 300 épocas con SGD a `lr0=0.01`,
momentum 0.9, weight decay 4e-5 y un warmup de 1 época sobre un schedule coseno.
El volteo horizontal es el único aumento de datos que se aplica.

Consulta [entrenamiento](/docs/train) para datasets, aumento de datos
(data augmentation), multi-GPU y loggers.

## Validación

`val()` devuelve un diccionario de claves `metrics/` que cubren precisión,
recall, mAP 50 y mAP 50-95, medidas contra cualquier dataset en el formato con
el que entrenaste.

<code-tabs name="val" />

## Exportación

<export-matrix />

Un artefacto exportado se vuelve a cargar con `LibreYOLO()` según la extensión
del archivo, así que un archivo `.onnx` o `.engine` se comporta como un
checkpoint y devuelve el mismo `Results`. Ejecutar el grafo en un runtime pelado,
sin LibreYOLO instalado, también está soportado, pero entonces el preprocesado y
el postprocesado corren de tu cuenta.

<code-tabs name="export" />

## Checkpoints

Todos los archivos de pesos publicados de esta familia.

<checkpoint-table />

## Licencia

<provenance-box>

El port de LibreYOLO sigue a Bo396543018/Picodet_Pytorch, una reimplementación
en PyTorch del PP-PicoDet original de PaddleDetection, con mmcv eliminado y
todas las activaciones replicadas con exactitud para que los checkpoints de
PaddlePaddle convertidos con el pipeline de Bo carguen sin ninguna deriva
numérica. Ambas fuentes llevan los mismos términos Apache-2.0 que los autores
del paper.

</provenance-box>

## Cita

<citation-block />
