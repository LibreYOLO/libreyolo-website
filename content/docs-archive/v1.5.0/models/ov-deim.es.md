---
title: OV-DEIM
families:
  - ov_deim
seo_title: 'OV-DEIM en LibreYOLO: detección de vocabulario abierto'
description: >-
  Usa OV-DEIM en LibreYOLO para detección de vocabulario abierto en tiempo real,
  al estilo DETR. Instala el extra openvocab y predice con un vocabulario de
  texto libre.
lead: >-
  OV-DEIM es un detector de objetos de vocabulario abierto al estilo DETR que
  empareja las queries del decoder con embeddings de texto de una torre de texto
  MobileCLIP incluida. LibreYOLO lo porta de forma nativa como una familia solo
  de predicción dentro de su tier de detectores de vocabulario abierto.
keywords:
  - OV-DEIM
  - DEIMv2
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

        model = LibreOpenVocab("ov-deim-s")
        model.set_classes(["person", "dog", "skateboard"])

        result = model.predict(SAMPLE_IMAGE, conf=0.25)
        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Sustituir el vocabulario
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("ov-deim-l")
        model.set_classes(["traffic light", "bicycle"])
        first = model.predict(SAMPLE_IMAGE, conf=0.3)

        # Una segunda llamada a set_classes() sustituye el vocabulario por
        # completo y lo vuelve a embeber a través de la torre de texto; un
        # resultado vacío es un desenlace válido, no un error.
        model.set_classes(["giraffe"])
        second = model.predict(SAMPLE_IMAGE, conf=0.5)
        print(second.names, len(second))
source_hash: 0c295f555a9eb303
---

## Instalación

OV-DEIM se carga a través del tier de detectores de vocabulario abierto de
LibreYOLO, que necesita el extra `openvocab`:

```bash
pip install "libreyolo[openvocab]"
```

A diferencia del resto de este tier, OV-DEIM es un port nativo de LibreYOLO en
lugar de un wrapper de `transformers`, no existe ninguna clase de modelo de
`transformers` para él, pero el mismo extra cubre los paquetes
`huggingface_hub`, `safetensors`, `regex` y `ftfy` que necesita en el momento
de la predicción.

## Predicción

OV-DEIM no es un checkpoint que LibreYOLO cargue a través de `LibreYOLO()`. Se
carga a través de la factoría hermana `LibreOpenVocab`, que descarga un
snapshot de Hugging Face en el primer uso y lo cachea en `weights/`.

<code-tabs name="predict" />

`set_classes()` fija un vocabulario de texto persistente: vuelve a llamarla
para sustituir la lista por completo, u omítela para conservar las etiquetas
COCO-80 por defecto, y un resultado vacío es un desenlace válido, no un error.
Cada query del decoder se puntúa por similitud del coseno frente a los
embeddings de texto de una torre de texto MobileCLIP-B(LT) incluida,
calculados en línea para el vocabulario que esté fijado y cacheados hasta que
cambie, así que funcionan prompts arbitrarios sin ningún archivo de embeddings
precalculados.

OV-DEIM no tiene umbral de token de texto: solo `conf` filtra las detecciones,
y pasar `text_threshold` lanza un error. La coincidencia es una selección
top-K uno a uno, así que aquí nada ejecuta non-maximum suppression, e `iou` se
acepta por compatibilidad de la API, pero avisa y no hace nada. `imgsz` y
`augment=True` se rechazan directamente: el modelo tiene su propia entrada con
letterbox fijo, y el aumento de datos en tiempo de test queda fuera del
alcance de este tier. `predict()` sobre una única imagen devuelve un
`Results`, no una lista; pasa un directorio, una lista de imágenes o
`stream=True` con una fuente de vídeo para obtener varios. No hay ruta de CLI
para esta familia, `libreyolo predict` solo carga checkpoints `.pt` a través
de `LibreYOLO()`, así que las familias de `LibreOpenVocab` se ejecutan desde
Python. Consulta [predicción](/docs/predict) para tipos de fuente y streaming.

Cada llamada a `predict()` ejecuta también la torre de texto MobileCLIP-B(LT)
incluida para embeber el vocabulario actual; consulta Licencia para ver qué
añade eso a los términos.

## Variantes

Tres checkpoints, `s`, `m` y `l`. `s` es el tamaño por defecto de este tier
cuando no se indica ninguno. A diferencia del resto de este tier, OV-DEIM es
un port nativo en lugar de un wrapper de `transformers`: LibreYOLO incorpora
los módulos del detector bajo la misma licencia Apache-2.0 que el código
upstream y reutiliza el adaptador de backbone DINOv3 ya construido para la
familia DEIMv2. El backbone del checkpoint `l` es un fine-tune de DINOv3-S,
con licencia aparte bajo la DINOv3 License de Meta. Todavía no hay cifras
publicadas de precisión ni de latencia para esta familia.

El entrenamiento, la validación de datasets y la exportación quedan todos
fuera del alcance de este tier: `train()`, `val()` y `export()` lanzan
`NotImplementedError` de forma incondicional. Esto es un wrapper solo de
predicción alrededor de un checkpoint publicado.

## Checkpoints

Todos los archivos de pesos publicados de esta familia.

<checkpoint-table />

## Licencia

<provenance-box>

OV-DEIM superpone tres licencias upstream a cada llamada de predicción: los
pesos del detector bajo la CC BY-NC 4.0 propia de OV-DEIM, la torre de texto
en línea bajo la Machine Learning Research Model license de Apple (solo para
uso de investigación) y, en el caso del checkpoint `l`, un fine-tune del
backbone DINOv3-S bajo la DINOv3 License de Meta. Los textos de las tres
licencias se incluyen dentro del repositorio de pesos de LibreYOLO.

</provenance-box>

## Cita

<citation-block />
