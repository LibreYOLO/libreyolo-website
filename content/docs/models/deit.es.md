---
title: DeiT
families: [deit]
seo_title: "Clasificador de imágenes DeiT: predicción, validación y exportación"
description: "Ejecuta clasificadores de imágenes DeiT en LibreYOLO: una familia de museo congelada y solo de inferencia, en tamaños tiny, small y base, con licencia Apache-2.0."
lead: "DeiT (Data-efficient image Transformer) es un clasificador Vision Transformer puro entrenado únicamente con ImageNet-1k, sin datos de preentrenamiento adicionales. LibreYOLO incluye los tamaños tiny, small y base con patch-16 como una pieza de museo congelada y solo de inferencia."
keywords: [DeiT, Vision Transformer, ViT, "clasificación de imágenes python", ImageNet, "clasificador de imágenes preentrenado", "familia de museo"]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDeiTb-cls.pt")
        result = model(SAMPLE_IMAGE)

        print(result.probs.top1, result.probs.top1conf)
        print(result.probs.top5)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreDeiTb-cls.pt source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDeiTb-cls.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDeiTb-cls.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDeiTb-cls.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreDeiTb-cls.pt format=onnx
        libreyolo export model=LibreDeiTb-cls.pt format=tensorrt half=True
    - label: Usar el archivo exportado
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La factoría enruta según la extensión del archivo, así que un artefacto
        # exportado se carga como cualquier checkpoint y devuelve el mismo objeto
        # Results.
        model = LibreYOLO("LibreDeiTb-cls.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.probs.top1)
---

## Instalación

DeiT no necesita ningún extra más allá del paquete base.

```bash
pip install libreyolo
```

## Predicción

Esta familia es solo de inferencia: `train()` lanza `NotImplementedError`, así
que esta página no tiene sección de entrenamiento. La predicción, la validación
y la exportación sí están soportadas. Los pesos se descargan de Hugging Face en
el primer uso y se guardan en la caché local. El sufijo `-cls` del nombre de
archivo es obligatorio y selecciona la tarea de clasificación.

<code-tabs name="predict" />

El objeto `Results` que se devuelve lleva un tensor `probs` en lugar de
`boxes`; `top1` y `top5` indexan las 1.000 clases de ImageNet-1k y `top1conf`
es la puntuación softmax de la predicción principal. Cada tamaño tiene una
resolución de entrada fija que viene de su positional embedding: el
preprocesado redimensiona y recorta al centro hasta esa resolución, y pasar un
`imgsz` distinto lanza un error en lugar de remuestrear en silencio. Consulta
[predicción](/docs/predict) para fuentes, streaming y manejo de resultados.

## Validación

`val()` devuelve un diccionario con la precisión top-1 y top-5, medida sobre un
dataset organizado en la estructura de carpetas convencional `train/<class>/` y
`val/<class>/`.

<code-tabs name="val" />

## Exportación

<export-matrix />

Un artefacto exportado se vuelve a cargar con `LibreYOLO()` según su extensión
de archivo, así que un archivo `.onnx` o `.engine` se comporta como un
checkpoint y devuelve el mismo `Results`. Ejecutar el grafo en un runtime
pelado, sin LibreYOLO instalado, también está soportado, pero entonces el
preprocesado y el postprocesado corren de tu cuenta.

<code-tabs name="export" />

## Checkpoints

Todos los archivos de pesos publicados de esta familia.

<checkpoint-table />

## Licencia

<provenance-box></provenance-box>

## Cita

<citation-block />
