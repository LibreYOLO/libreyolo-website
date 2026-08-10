---
title: DINO-DETR
families:
  - dinodetr
seo_title: 'DINO-DETR: predice y exporta, con licencia Apache-2.0'
description: >-
  Usa DINO-DETR en LibreYOLO para detección de objetos. Instala, predice, valida
  y exporta tres tamaños con anchors de denoising, todos con licencia
  Apache-2.0.
lead: >-
  DINO-DETR, publicado por IDEA Research con el nombre DINO, combina el
  entrenamiento con denoising contrastivo y la selección mixta de queries sobre
  la atención dispersa de Deformable DETR. LibreYOLO incluye tres tamaños para
  detección, solo inferencia.
keywords:
  - DINO-DETR
  - DINO
  - transformer de detección
  - denoising contrastivo
  - anchor boxes con denoising
  - selección mixta de queries
  - detección de objetos
  - IDEA Research
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDINODETRr50.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreDINODETRr50.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDINODETRr50.pt")

        # val() devuelve un dict normal, no un objeto
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDINODETRr50.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDINODETRr50.pt")
        model.export(format="onnx", imgsz=800)
        model.export(format="tensorrt", imgsz=800, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreDINODETRr50.pt format=onnx imgsz=800

        libreyolo export model=LibreDINODETRr50.pt format=tensorrt imgsz=800
        half=True
    - label: Usar el archivo exportado
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # La factoría enruta según la extensión del archivo, así que un
        artefacto

        # exportado se carga como cualquier checkpoint y devuelve el mismo

        # objeto Results.

        model = LibreYOLO("LibreDINODETRr50.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: dda176ebee3a83de
---

## Instalación

DINO-DETR no necesita ningún extra opcional. Todo lo que importa está en la
instalación base, y usa el mismo núcleo de atención deformable multiescala en
PyTorch puro que la familia Deformable DETR de LibreYOLO.

```bash
pip install libreyolo
```

Instalar `libreyolo[hub-kernels]` es opcional. Una vez presente el paquete
`kernels`, LibreYOLO descarga en tiempo de ejecución un kernel compilado de
atención deformable multiescala desde el Hugging Face Hub y lo usa en lugar del
núcleo en PyTorch puro; `LIBREYOLO_HUB_KERNELS=0` lo vuelve a desactivar.

## Predicción

Los pesos se descargan de Hugging Face en el primer uso y se guardan en la
caché local.

<code-tabs name="predict" />

El objeto `Results` devuelto es el mismo que devuelven todas las familias, así
que cambiar a otro detector es un cambio de una línea. `conf` y `max_det`
filtran la selección de queries; `iou` se acepta por paridad de API pero no
tiene efecto, porque el decoder es un predictor de conjuntos sin paso de NMS.
Consulta [predicción](/docs/predict) para fuentes, streaming y manejo de
resultados.

DINO-DETR es solo de inferencia en LibreYOLO. Upstream lo entrena con denoising
contrastivo y matching húngaro; esa receta no está implementada aquí, así que
`train()` lanza `NotImplementedError`.

## Variantes

Tres checkpoints, todos a la misma resolución de entrada. `r50` y `r50s5`
comparten un backbone ResNet-50 y se diferencian en cuántas escalas de mapas de
características alimentan el decoder, cuatro frente a cinco. `swinl` cambia el
backbone por Swin-L y también muestrea cinco escalas.

## Validación

`val()` devuelve un diccionario de claves `metrics/` que cubren precisión,
recall, mAP 50 y mAP 50-95, medidas contra cualquier dataset en el formato con
el que entrenaste.

<code-tabs name="val" />

## Exportación

<export-matrix />

Un artefacto exportado se vuelve a cargar con `LibreYOLO()` según su extensión
de archivo, así que un archivo `.onnx` o `.engine` se comporta como un
checkpoint y devuelve el mismo `Results`.
[Exportación](/docs/export) lista los argumentos que acepta cada formato.

<code-tabs name="export" />

## Checkpoints

Todos los archivos de pesos publicados de esta familia.

<checkpoint-table />

## Licencia

<provenance-box>

Los tres checkpoints oficiales provienen de la carpeta de publicación en Google
Drive de los autores, no de una model card de Hugging Face. El repositorio
upstream declara Apache-2.0 a nivel de repositorio, pero no adjunta un archivo
de licencia ni metadatos de licencia a los checkpoints en sí, así que la base
para la redistribución es esa declaración a nivel de repositorio y no una
concesión específica de los checkpoints. Todos los mirrors de LibreYOLO
incluyen el texto literal de la licencia Apache-2.0 de upstream junto con un
aviso que lo explica.

</provenance-box>

## Cita

<citation-block />
