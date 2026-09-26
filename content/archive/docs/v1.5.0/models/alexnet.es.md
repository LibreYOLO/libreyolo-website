---
title: AlexNet
families:
  - alexnet
seo_title: 'AlexNet: ejecuta el clásico clasificador de ImageNet en LibreYOLO'
description: >-
  Predice, valida y exporta AlexNet con LibreYOLO. Pesos de torchvision con
  licencia BSD-3-Clause; el fine-tuning todavía no está soportado.
lead: >-
  AlexNet es la red convolucional que ganó la ILSVRC 2012 y ayudó a poner en
  marcha la era del deep learning en visión por computador. LibreYOLO incluye la
  revisión posterior de una sola torre de la arquitectura para clasificación de
  imágenes.
keywords:
  - AlexNet
  - ImageNet
  - red neuronal convolucional
  - clasificación de imágenes python
  - clasificador de imágenes preentrenado
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreAlexNetb-cls.pt")
        result = model(SAMPLE_IMAGE, save=True)

        probs = result.probs
        print(probs.top1, probs.top1conf)
        print(probs.top5, probs.top5conf)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreAlexNetb-cls.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreAlexNetb-cls.pt")

        # data es un directorio raíz con splits train/ y val/ en carpetas por
        # clase (formato ImageFolder), no un YAML de dataset.
        metrics = model.val(data="imagenet-1k/")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreAlexNetb-cls.pt data=imagenet-1k/
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreAlexNetb-cls.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreAlexNetb-cls.pt format=onnx
        libreyolo export model=LibreAlexNetb-cls.pt format=tensorrt half=True
    - label: Usar el archivo exportado
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # La factoría enruta según la extensión del archivo, así que un
        artefacto

        # exportado se carga como cualquier checkpoint y devuelve el mismo

        # objeto Results.

        model = LibreYOLO("LibreAlexNetb-cls.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.probs.top1)
source_hash: 68c09f080c74bb87
---

## Instalación

AlexNet no necesita ningún extra opcional. Todo lo que importa está en la
instalación base.

```bash
pip install libreyolo
```

## Predicción

Los pesos se descargan de Hugging Face en el primer uso y se guardan en la
caché local.

<code-tabs name="predict" />

Un clasificador devuelve `result.probs` en lugar de `result.boxes`: `top1`
y `top5` dan los índices de clase, y `top1conf` y `top5conf` dan sus
confianzas. Consulta [predicción](/docs/predict) para fuentes, streaming y
manejo de resultados.

## Variantes

Un solo tamaño. El grafo que se incluye es la revisión posterior de una sola
torre publicada por torchvision, con 64 filtros en la primera capa y sin local
response normalization, no la arquitectura original de dos GPU de 2012.
LibreYOLO incluye esta familia solo para inferencia: la predicción, la
validación top-1/top-5 al estilo ImageNet y la exportación están soportadas, y
el fine-tuning no está implementado.

## Validación

`val()` se ejecuta sobre un split con formato ImageFolder (un directorio con
subcarpetas `train/` y `val/`, una carpeta por clase) y devuelve la precisión
top-1 y top-5.

<code-tabs name="val" />

## Exportación

<export-matrix />

Un artefacto exportado se vuelve a cargar con `LibreYOLO()` según su extensión
de archivo, así que un archivo `.onnx` o `.engine` se comporta como un
checkpoint y devuelve el mismo `Results`. [Exportación](/docs/export) enumera
los argumentos que acepta cada formato y los extras que añaden algunos de ellos.

<code-tabs name="export" />

## Checkpoints

Todos los archivos de pesos publicados de esta familia.

<checkpoint-table />

## Licencia

<provenance-box></provenance-box>
