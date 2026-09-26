---
title: ViT
families:
  - vit
seo_title: 'ViT: ejecuta clasificadores Vision Transformer clásicos en LibreYOLO'
description: >-
  Predice, valida y exporta clasificadores ViT con LibreYOLO. Pesos AugReg con
  licencia Apache-2.0; el fine-tuning todavía no está soportado.
lead: >-
  El Vision Transformer clásico: un transformer puro aplicado a parches de
  imagen de tamaño fijo, con un class token aprendido y sin convoluciones.
  LibreYOLO incluye cuatro tamaños preentrenados con AugReg para clasificación
  de imágenes.
keywords:
  - ViT
  - Vision Transformer
  - AugReg
  - clasificación de imágenes python
  - clasificador transformer
  - vision transformer preentrenado
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreViTti-cls.pt")
        result = model(SAMPLE_IMAGE, save=True)

        probs = result.probs
        print(probs.top1, probs.top1conf)
        print(probs.top5, probs.top5conf)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreViTti-cls.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreViTti-cls.pt")

        # data es un directorio raíz con splits train/ y val/ en carpetas por
        # clase (estructura ImageFolder), no un YAML de dataset.
        metrics = model.val(data="imagenet-1k/")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreViTti-cls.pt data=imagenet-1k/
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreViTti-cls.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreViTti-cls.pt format=onnx
        libreyolo export model=LibreViTti-cls.pt format=tensorrt half=True
    - label: Usar el archivo exportado
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # La factoría enruta según la extensión del archivo, así que un
        artefacto

        # exportado se carga como cualquier checkpoint y devuelve el mismo
        objeto

        # Results.

        model = LibreYOLO("LibreViTti-cls.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.probs.top1)
source_hash: f63e98454913765a
---

## Instalación

ViT no necesita ningún extra opcional. Todo lo que importa está en la
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
El preprocesado redimensiona y recorta al centro hasta una entrada fija de
224px, siguiendo la receta de evaluación AugReg de timm: interpolación bicúbica
con una fracción de recorte de 0,9. Consulta [predicción](/docs/predict) para
fuentes, streaming y manejo de resultados.

## Variantes

Cuatro tamaños, de tiny a large, que comparten un mismo grafo fijo de 224px con
patch-16 y se diferencian en el ancho del embedding y en la profundidad del
transformer. LibreYOLO incluye esta familia solo para inferencia: la
predicción, la validación top-1/top-5 al estilo ImageNet y la exportación están
soportadas, y la receta de fine-tuning de AugReg no está implementada.

## Validación

`val()` se ejecuta sobre un split con estructura ImageFolder (un directorio con
subcarpetas `train/` y `val/`, una carpeta por clase) y devuelve la precisión
top-1 y top-5.

<code-tabs name="val" />

## Exportación

<export-matrix />

Un artefacto exportado se vuelve a cargar con `LibreYOLO()` según su extensión
de archivo, así que un archivo `.onnx` o `.engine` se comporta como un
checkpoint y devuelve el mismo `Results`. [Exportación](/docs/export) enumera
los argumentos que acepta cada formato y los extras que unos pocos añaden.

<code-tabs name="export" />

## Checkpoints

Todos los archivos de pesos publicados de esta familia.

<checkpoint-table />

## Licencia

<provenance-box></provenance-box>

## Cita

<citation-block />
