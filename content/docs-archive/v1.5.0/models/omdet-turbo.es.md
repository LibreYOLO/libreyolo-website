---
title: OMDet-Turbo
families:
  - omdet_turbo
seo_title: 'OMDet-Turbo en LibreYOLO: detección zero-shot en tiempo real'
description: >-
  Usa OMDet-Turbo en LibreYOLO para detección de vocabulario abierto en tiempo
  real. Instala el extra openvocab y predice con un vocabulario de texto libre.
lead: >-
  OMDet-Turbo es un detector de objetos de vocabulario abierto en tiempo real,
  desarrollado por Om AI Lab, que desacopla los embeddings de clase del prompt
  de tarea de lenguaje. LibreYOLO lo envuelve como una familia solo de
  predicción dentro de su tier de detectores de vocabulario abierto.
keywords:
  - OMDet-Turbo
  - OmDet
  - detección de vocabulario abierto
  - detección de objetos en tiempo real
  - detección zero-shot
  - detectar objetos por texto
  - LibreOpenVocab
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("omdet-turbo")
        model.set_classes(["person", "dog", "skateboard"])

        result = model.predict(SAMPLE_IMAGE, conf=0.3)
        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Umbral NMS personalizado
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("omdet-turbo")
        model.set_classes(["traffic light", "bicycle"])

        # OMDet-Turbo es la única familia de este tier que respeta iou=: su
        # propio post-procesado toma el umbral de supresión como argumento,
        # y vale 0.5 por defecto cuando iou= se deja sin indicar.
        result = model.predict(SAMPLE_IMAGE, conf=0.3, iou=0.7)
        print(result.names, len(result))
source_hash: c2a375d234341b7e
---

## Instalación

OMDet-Turbo se carga a través del tier de detectores de vocabulario abierto de
LibreYOLO, que necesita el extra `openvocab`:

```bash
pip install "libreyolo[openvocab]"
```

Ese extra instala `transformers` y `timm`, las bibliotecas de Hugging Face a
las que llama este tier; el backbone Swin de OMDet-Turbo se carga a través del
wrapper `TimmBackbone` de `transformers`.

## Predicción

OMDet-Turbo no es un checkpoint que LibreYOLO cargue a través de `LibreYOLO()`.
Se carga a través de la factoría hermana `LibreOpenVocab`, que descarga un
snapshot de Hugging Face en el primer uso y lo cachea en `weights/`.

<code-tabs name="predict" />

`set_classes()` fija un vocabulario de texto persistente: vuelve a llamarla
para sustituir la lista por completo, u omítela para conservar las etiquetas
COCO-80 por defecto, y un resultado vacío es un desenlace válido en lugar de un
error. A diferencia de Grounding DINO, OMDet-Turbo desacopla sus embeddings de
clase del prompt de tarea de lenguaje, así que el post-procesado de
`transformers` devuelve etiquetas que se mapean directamente de vuelta a la
lista de clases consultada, sin paso de desambiguación de frases.

OMDet-Turbo no tiene umbral de token de texto: solo `conf` filtra las
detecciones, y pasar `text_threshold` lanza un error. Es la única familia de
este tier que ejecuta su propia non-maximum suppression dentro de
`post_process_grounded_object_detection`, así que aquí `iou` se respeta en
lugar de avisar. `imgsz` y `augment=True` se rechazan directamente: el
procesador de `transformers` es quien se encarga del redimensionado, y el
aumento de datos en tiempo de test queda fuera del alcance de este tier.
`predict()` sobre una única imagen devuelve un `Results`, no una lista; pasa un
directorio, una lista de imágenes o `stream=True` con una fuente de vídeo para
obtener varios. No hay ruta de CLI para esta familia, `libreyolo predict` solo
carga checkpoints `.pt` a través de `LibreYOLO()`, así que las familias de
`LibreOpenVocab` se ejecutan desde Python. Consulta
[predicción](/docs/predict) para tipos de fuente y streaming.

## Variantes

Un único checkpoint, `t`, el único tamaño del tier. Replica
`omlab/omdet-turbo-swin-tiny-hf` en una revisión upstream fijada a través de
`OmDetTurboForObjectDetection` de `transformers`; el archivo de pesos replicado
es idéntico byte a byte a ese snapshot upstream. Todavía no hay cifras
publicadas de precisión ni de latencia para esta familia.

El entrenamiento, la validación de datasets y la exportación quedan todos fuera
del alcance de este tier: `train()`, `val()` y `export()` lanzan
`NotImplementedError` de forma incondicional. Esto es un wrapper solo de
predicción alrededor de un checkpoint publicado.

## Checkpoints

Todos los archivos de pesos publicados de esta familia.

<checkpoint-table />

## Licencia

<provenance-box></provenance-box>

## Cita

<citation-block />
