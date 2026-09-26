---
title: Swin Transformer
families:
  - swin
seo_title: 'Swin Transformer: clasifica imágenes con LibreSwin de LibreYOLO'
description: >-
  Predice, valida y exporta clasificadores Swin Transformer con LibreYOLO. Pesos
  con licencia MIT; el fine-tuning todavía no está soportado.
lead: >-
  Swin Transformer V1: un vision transformer jerárquico que calcula la atención
  dentro de ventanas locales desplazadas en lugar de sobre la imagen entera.
  LibreYOLO incluye cuatro tamaños para clasificación de imágenes.
keywords:
  - Swin Transformer
  - vision transformer jerárquico
  - atención por ventanas desplazadas
  - clasificación de imágenes python
  - clasificador de imágenes preentrenado
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreSwint-cls.pt")
        result = model(SAMPLE_IMAGE, save=True)

        probs = result.probs
        print(probs.top1, probs.top1conf)
        print(probs.top5, probs.top5conf)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreSwint-cls.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSwint-cls.pt")

        # data es un directorio raíz con splits train/ y val/ en carpetas por
        # clase (estructura ImageFolder), no un YAML de dataset.
        metrics = model.val(data="imagenet-1k/")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreSwint-cls.pt data=imagenet-1k/
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSwint-cls.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreSwint-cls.pt format=onnx
        libreyolo export model=LibreSwint-cls.pt format=tensorrt half=True
    - label: Usar el archivo exportado
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # La factoría enruta según la extensión del archivo, así que un
        artefacto

        # exportado se carga como cualquier checkpoint y devuelve el mismo
        objeto

        # Results.

        model = LibreYOLO("LibreSwint-cls.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.probs.top1)
source_hash: faa6bbacae62d88e
---

## Instalación

Swin no necesita ningún extra opcional. Todo lo que importa está en la
instalación base.

```bash
pip install libreyolo
```

## Predicción

Los pesos se descargan de Hugging Face en el primer uso y se guardan en la
caché local.

<code-tabs name="predict" />

Un clasificador devuelve `result.probs` en lugar de `result.boxes`: `top1` y
`top5` dan los índices de clase, y `top1conf` y `top5conf` dan sus confianzas.
Todos los tamaños están fijados a una entrada de 224px, porque la etapa final
de atención está construida para esa resolución; la predicción, la validación y
la exportación lanzan un error si pasas un `imgsz` distinto. Consulta
[predicción](/docs/predict) para fuentes, streaming y manejo de resultados.

## Variantes

Cuatro tamaños, de tiny a large, construidos sobre la misma torre de ventanas
desplazadas y que se diferencian en el ancho del embedding y en la profundidad
de cada etapa. El large está preentrenado en ImageNet-22k y ajustado con
fine-tuning en ImageNet-1k; los otros tres se entrenan directamente en
ImageNet-1k. LibreYOLO incluye esta familia solo para inferencia: la
predicción, la validación top-1/top-5 al estilo ImageNet y la exportación están
soportadas, y la receta de entrenamiento en ImageNet del proyecto original no
está implementada.

## Validación

`val()` se ejecuta sobre un split al estilo ImageFolder (un directorio con
subcarpetas `train/` y `val/`, una carpeta por clase) y devuelve la precisión
top-1 y top-5.

<code-tabs name="val" />

## Exportación

<export-matrix />

Un artefacto exportado se vuelve a cargar con `LibreYOLO()` según su extensión
de archivo, así que un archivo `.onnx` o `.engine` se comporta como un
checkpoint y devuelve el mismo `Results`. [Exportación](/docs/export) enumera
los argumentos que acepta cada formato y los extras que añaden algunos de
ellos.

<code-tabs name="export" />

## Checkpoints

Todos los archivos de pesos publicados de esta familia.

<checkpoint-table />

## Licencia

<provenance-box></provenance-box>

## Cita

<citation-block />
