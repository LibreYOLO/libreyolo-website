---
title: OWLv2
families:
  - owlv2
seo_title: 'OWLv2 en LibreYOLO: detección de objetos zero-shot'
description: >-
  Usa OWLv2 en LibreYOLO para detectar cualquier objeto descrito con texto.
  Instala el extra openvocab y predice con un vocabulario de texto libre.
lead: >-
  OWLv2 es un detector de objetos de vocabulario abierto, desarrollado por
  Google Research, que puntúa regiones de la imagen frente a embeddings de texto
  de un codificador estilo CLIP. LibreYOLO lo envuelve como una familia solo de
  predicción dentro de su tier de detectores de vocabulario abierto.
keywords:
  - OWLv2
  - OWL-ViT
  - detección de vocabulario abierto
  - detección zero-shot
  - detectar objetos por texto
  - detectar objetos sin entrenar
  - LibreOpenVocab
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("owlv2-b16")
        model.set_classes(["person", "dog", "skateboard"])

        result = model.predict(SAMPLE_IMAGE, conf=0.1)
        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Vocabulario por defecto
      language: python
      code: >
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE


        # Omitir set_classes() conserva el vocabulario COCO-80 por defecto del
        tier.

        model = LibreOpenVocab("owlv2-l14")

        result = model.predict(SAMPLE_IMAGE, conf=0.1)

        print(result.names)
source_hash: 2d0ce68af0daabb7
---

## Instalación

OWLv2 se carga a través del tier de detectores de vocabulario abierto de
LibreYOLO, que necesita el extra `openvocab`:

```bash
pip install "libreyolo[openvocab]"
```

Ese extra instala `transformers` y `timm`, las bibliotecas de Hugging Face a
las que llama este tier.

## Predicción

OWLv2 no es un checkpoint que LibreYOLO cargue a través de `LibreYOLO()`. Se
carga a través de la factoría hermana `LibreOpenVocab`, que descarga un
snapshot de Hugging Face en el primer uso y lo cachea en `weights/`.

<code-tabs name="predict" />

`set_classes()` fija un vocabulario de texto persistente: vuelve a llamarla
para sustituir la lista, u omítela para conservar las etiquetas COCO-80 por
defecto. Cada etiqueta se envuelve en una plantilla de prompt fija antes de
llegar a la torre de texto, igual que se entrenó `Owlv2ForObjectDetection` de
`transformers`.

OWLv2 no tiene umbral de token de texto: solo `conf` filtra las detecciones, y
pasar `text_threshold` lanza un error. `iou` se acepta por compatibilidad de la
API, pero avisa y no hace nada, ya que aquí nada ejecuta non-maximum
suppression. `imgsz` y `augment=True` se rechazan directamente: el procesador
de `transformers` es quien se encarga del redimensionado, y el aumento de datos
en tiempo de test queda fuera del alcance de este tier. `predict()` sobre una
única imagen devuelve un `Results`, no una lista; pasa un directorio, una lista
de imágenes o `stream=True` con una fuente de vídeo para obtener varios. No hay
ruta de CLI para esta familia, `libreyolo predict` solo carga checkpoints `.pt`
a través de `LibreYOLO()`, así que las familias de `LibreOpenVocab` se ejecutan
desde Python. Consulta [predicción](/docs/predict) para tipos de fuente y
streaming.

## Variantes

Dos checkpoints, `b16` (base, tamaño de patch 16) y `l14` (large, tamaño de
patch 14). `b16` es el tamaño por defecto de este tier cuando no se indica
ninguno. Ambos replican la publicación oficial de Google Research a través de
`Owlv2ForObjectDetection` de `transformers`, descargado una sola vez en un
snapshot de Hugging Face alojado por LibreYOLO que conserva los archivos
originales. Todavía no hay cifras publicadas de precisión ni de latencia para
esta familia.

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
