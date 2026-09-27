---
title: LingBot-Vision
families:
  - lingbotvision
seo_title: 'LingBot-Vision: segmentación semántica en LibreYOLO'
description: >-
  Usa LingBot-Vision en LibreYOLO para segmentación semántica sobre un backbone
  ViT con licencia Apache-2.0. Instala, predice, entrena, valida y exporta,
  tamaños s/b/l.
lead: >-
  LingBot-Vision es una familia de backbones vision transformer autosupervisados
  entrenados con masked modeling centrado en bordes para percepción espacial
  densa, publicada por Robbyant. LibreYOLO combina el backbone con una cabeza
  densa y lo soporta para una tarea: segmentación semántica.
keywords:
  - LingBot-Vision
  - segmentación semántica python
  - vision transformer
  - aprendizaje autosupervisado
  - detección de bordes
  - Robbyant
  - predicción densa
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreLingBotVisions-sem.pt")
        result = model(SAMPLE_IMAGE, save=True)

        mask = result.semantic_mask
        print(mask.data.shape, mask.classes)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreLingBotVisions-sem.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  train:
    - label: Python (linear probe)
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Backbone congelado por defecto, siguiendo el protocolo de evaluación
        # original: solo se entrena la cabeza densa 1x1.
        model = LibreYOLO("LibreLingBotVisions-sem.pt")
        model.train(data="my-dataset.yaml", epochs=20, imgsz=512, batch=16)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreLingBotVisions-sem.pt data=my-dataset.yaml \
          epochs=20 imgsz=512 batch=16
    - label: Fine-tune completo
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreLingBotVisions-sem.pt")
        model.train(
            data="my-dataset.yaml", epochs=20, imgsz=512, batch=16,
            freeze_backbone=False,
        )
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreLingBotVisions-sem.pt data=my-dataset.yaml \
          epochs=20 device=0,1 batch=32
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreLingBotVisions-sem.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreLingBotVisions-sem.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreLingBotVisions-sem.pt")
        model.export(format="onnx", imgsz=512)
        model.export(format="coreai", imgsz=512)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreLingBotVisions-sem.pt format=onnx imgsz=512
    - label: Usar el archivo exportado
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # La factoría enruta según la extensión del archivo, así que un
        artefacto

        # exportado se carga como cualquier checkpoint y devuelve el mismo
        objeto

        # Results.

        model = LibreYOLO("LibreLingBotVisions-sem.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.semantic_mask.data.shape)
source_hash: c47b33fdc6fa1139
---

## Instalación

LingBot-Vision no necesita ningún extra opcional. Todo lo que importa está en
la instalación base.

```bash
pip install libreyolo
```

## Predicción

Los pesos se descargan de Hugging Face en el primer uso y se guardan en la
caché local.

<code-tabs name="predict" />

`result.semantic_mask` lleva el mapa denso de clases: `.data` es un tensor
`(H, W)` de ids de clase al tamaño original de la imagen, y `.classes` enumera
los ids de clase realmente presentes. `result.boxes` es `None`, ya que no hay
detecciones por instancia. `conf` e `iou` se aceptan por paridad de API pero no
cambian la salida, ya que el modelo devuelve una clase por píxel en lugar de
detecciones que filtrar. Consulta [predicción](/docs/predict) para fuentes,
streaming y manejo de resultados.

## Variantes

Tres tamaños publicados, s, b y l, destilados de un profesor ViT-g/16 de 1.100
millones de parámetros. El propio profesor, tamaño `g`, se carga y se le puede
hacer fine-tuning en LibreYOLO, pero LibreYOLO no aloja ningún checkpoint `g`
propio.

<checkpoint-table />

## Entrenamiento

`train()` hace fine-tuning de un checkpoint publicado. La receta por defecto es
el linear probe del informe original: el backbone ViT se congela y solo se
entrena la cabeza densa 1x1, igual que se produjeron los pesos alojados por
LibreYOLO de arriba. Pasa `freeze_backbone=False` para hacer fine-tuning de toda
la red en su lugar, y cuenta con bajar `lr0` en consecuencia.

<code-tabs name="train" />

Consulta [entrenamiento](/docs/train) para datasets, aumento de datos, multi-GPU
y loggers.

## Validación

`val()` devuelve un diccionario de claves `metrics/`: mIoU y precisión por
píxel, medidos contra cualquier dataset en el formato con el que entrenaste.

<code-tabs name="val" />

## Exportación

<export-matrix />

Un artefacto exportado se vuelve a cargar con `LibreYOLO()` según su extensión
de archivo, así que un archivo `.onnx` o `.engine` se comporta como un
checkpoint y devuelve el mismo `Results`. [Exportación](/docs/export) enumera
los argumentos que acepta cada formato.

<code-tabs name="export" />

## Checkpoints

Todos los archivos de pesos publicados de esta familia.

<checkpoint-table />

## Licencia

<provenance-box>

La release original documenta su ViT como construido sobre la arquitectura
DINOv2/DINOv3 publicada por Meta AI. Robbyant distribuye su implementación bajo
Apache-2.0, y este port a LibreYOLO se hizo únicamente a partir del repositorio
de Robbyant, nunca del código DINOv2 o DINOv3 de Meta.

</provenance-box>

## Cita

<citation-block />
