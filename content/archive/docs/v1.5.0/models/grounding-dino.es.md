---
title: Grounding DINO
families:
  - grounding_dino
seo_title: 'Grounding DINO en LibreYOLO: detección de conjunto abierto'
description: >-
  Usa Grounding DINO en LibreYOLO para detectar cualquier objeto descrito con
  texto. Instala el extra openvocab y predice con un vocabulario de texto libre.
lead: >-
  Grounding DINO es un detector de objetos de conjunto abierto, desarrollado por
  IDEA Research, que puntúa una imagen frente a un prompt de texto libre en
  lugar de una lista fija de clases. LibreYOLO lo envuelve como una familia solo
  de predicción dentro de su tier de detectores de vocabulario abierto.
keywords:
  - Grounding DINO
  - detección de vocabulario abierto
  - detectar objetos por texto
  - detección zero-shot
  - detectar objetos sin entrenar
  - open-set detection
  - LibreOpenVocab
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("grounding-dino-t")
        model.set_classes(["person", "dog", "skateboard"])

        result = model.predict(SAMPLE_IMAGE, conf=0.25)
        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Umbral de texto
      language: python
      code: >
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE


        model = LibreOpenVocab("grounding-dino-b")

        model.set_classes(["remote control", "school bus"])


        # conf filtra por la puntuación de la caja y text_threshold por la

        # puntuación de token de la frase decodificada. Ambos valen 0.25 por
        defecto.

        result = model.predict(SAMPLE_IMAGE, conf=0.25, text_threshold=0.3)

        print(result.names)
source_hash: 06bd13b8e6a66038
---

## Instalación

Grounding DINO se carga a través del tier de detectores de vocabulario abierto
de LibreYOLO, que necesita el extra `openvocab`:

```bash
pip install "libreyolo[openvocab]"
```

Ese extra instala `transformers` y `timm`, las bibliotecas de Hugging Face a
las que llama este tier.

## Predicción

Grounding DINO no es un checkpoint que LibreYOLO cargue a través de
`LibreYOLO()`. Se carga a través de la factoría hermana `LibreOpenVocab`, que
descarga un snapshot de Hugging Face en el primer uso y lo cachea en
`weights/`.

<code-tabs name="predict" />

`set_classes()` fija un vocabulario de texto persistente: vuelve a llamarla
para sustituir la lista, u omítela para conservar las etiquetas COCO-80 por
defecto. Grounding DINO decodifica frases libres a partir de su propia salida
de texto y las mapea de vuelta a ese vocabulario por sí mismo, gana la
coincidencia normalizada exacta, se acepta una coincidencia de token completo,
y una frase ambigua o sin coincidencia se descarta en lugar de adivinarla, así
que `school bus` nunca acaba mapeado a `bus` ni a `school` por separado. Un
vocabulario lo bastante largo como para superar el límite de tokens del
codificador de texto se divide en varios prompts, se ejecuta como pases hacia
delante separados y se vuelve a fusionar en un único conjunto de detecciones
limitado por `max_det`.

`iou` se acepta por compatibilidad de la API, pero avisa y no hace nada, ya que
aquí nada ejecuta non-maximum suppression. `imgsz` y `augment=True` se rechazan
directamente: el procesador de `transformers` es quien se encarga del
redimensionado, y el aumento de datos en tiempo de test queda fuera del alcance
de este tier. `predict()` sobre una única imagen devuelve un `Results`, no una
lista; pasa un directorio, una lista de imágenes o `stream=True` con una fuente
de vídeo para obtener varios. No hay ruta de CLI para esta familia, `libreyolo
predict` solo carga checkpoints `.pt` a través de `LibreYOLO()`, así que las
familias de `LibreOpenVocab` se ejecutan desde Python. Consulta
[predicción](/docs/predict) para tipos de fuente y streaming.

## Variantes

Dos checkpoints, `t` y `b`. `t` es el tamaño por defecto de este tier cuando no
se indica ninguno. Ambos replican la publicación oficial de IDEA Research a
través de `GroundingDinoForObjectDetection` de `transformers`, descargado una
sola vez en un snapshot de Hugging Face alojado por LibreYOLO que conserva los
archivos originales. Todavía no hay cifras publicadas de precisión ni de
latencia para esta familia.

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
