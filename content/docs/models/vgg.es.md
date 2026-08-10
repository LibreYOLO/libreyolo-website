---
title: VGG
families: [vgg]
seo_title: "VGG: ejecuta clasificadores de imágenes VGG-16/19 en LibreYOLO"
description: "Predice, valida y exporta clasificadores VGG con LibreYOLO. Pesos de torchvision con licencia BSD-3-Clause; el fine-tuning todavía no está soportado."
lead: "VGG es un clasificador de imágenes convolucional construido con pilas uniformes de convoluciones pequeñas de 3x3 en lugar de filtros más grandes. LibreYOLO incluye los tamaños de 16 y 19 capas, en versión simple y con batch normalization, para clasificación de imágenes."
keywords: [VGG, VGG-16, VGG-19, "red neuronal convolucional", "clasificación de imágenes python", "clasificador de imágenes preentrenado"]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreVGG16-cls.pt")
        result = model(SAMPLE_IMAGE, save=True)

        probs = result.probs
        print(probs.top1, probs.top1conf)
        print(probs.top5, probs.top5conf)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreVGG16-cls.pt source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreVGG16-cls.pt")

        # data es un directorio raíz con splits train/ y val/ en carpetas por
        # clase (formato ImageFolder), no un YAML de dataset.
        metrics = model.val(data="imagenet-1k/")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreVGG16-cls.pt data=imagenet-1k/
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreVGG16-cls.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreVGG16-cls.pt format=onnx
        libreyolo export model=LibreVGG16-cls.pt format=tensorrt half=True
    - label: Usar el archivo exportado
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La factoría enruta según la extensión del archivo, así que un artefacto
        # exportado se carga como cualquier checkpoint y devuelve el mismo
        # objeto Results.
        model = LibreYOLO("LibreVGG16-cls.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.probs.top1)
---

## Instalación

VGG no necesita ningún extra opcional. Todo lo que importa está en la
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
confianzas. La predicción se ejecuta con una entrada fija de 224px y lanza un
error si pasas un `imgsz` distinto. Consulta [predicción](/docs/predict) para
fuentes, streaming y manejo de resultados.

## Variantes

Cuatro tamaños: 16 y 19 capas convolucionales, cada uno con una variante simple
y otra con batch normalization. Los pesos que se incluyen son los del
entrenamiento posterior desde cero sobre ImageNet hecho por torchvision, no
conversiones de la publicación original en Caffe de 2014 del grupo de Oxford.
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
