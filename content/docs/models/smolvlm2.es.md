---
title: SmolVLM2
families:
  - smolvlm2
seo_title: 'SmolVLM2 en LibreYOLO: detección de vocabulario abierto'
description: >-
  SmolVLM2 en LibreYOLO: instálalo, fija un vocabulario abierto y predice o
  chatea con el modelo de visión y lenguaje Apache-2.0 de Hugging Face.
lead: >-
  SmolVLM2 es el modelo de visión y lenguaje pequeño de Hugging Face. LibreYOLO
  lo envuelve como detector de objetos de vocabulario abierto y expone su chat
  libre directamente: pásale una lista de clases para detectar, o hazle una
  pregunta.
keywords:
  - SmolVLM2
  - modelo de visión y lenguaje
  - detección de vocabulario abierto
  - modelo multimodal pequeño
  - Hugging Face
  - VLM
  - detectar objetos con texto
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("smolvlm2-500m")
        model.set_classes(["cat", "dog"])
        result = model.predict(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Chat
      language: python
      code: >
        from libreyolo import LibreVLM, SAMPLE_IMAGE


        model = LibreVLM("smolvlm2-500m")


        # La vía de escape bajo la comodidad de la detección: cualquier
        pregunta,

        # no solo una consulta de bounding boxes.

        answer = model.chat(SAMPLE_IMAGE, "What is the cat doing?")

        print(answer)
source_hash: b30823b62d6347b5
---

## Instalación

SmolVLM2 pertenece al nivel VLM-como-detector de LibreYOLO, una superficie de
producto aparte de las familias basadas en checkpoints y con su propia
factoría. Necesita el extra `vlm`, que además arrastra `num2words`, una
dependencia del propio procesador de SmolVLM2.

```bash
pip install "libreyolo[vlm]"
```

## Predicción

Los pesos se descargan de Hugging Face en el primer uso y se cachean en local.

<code-tabs name="predict" />

Esta familia se carga a través de la factoría `LibreVLM()`, no de
`LibreYOLO()`: las familias VLM no declaran ningún cargador de checkpoints, así
que el enrutado por sufijo de archivo que describen otras páginas de modelos no
aplica aquí. `set_classes()` fija el vocabulario que se le pide encontrar a
SmolVLM2; es persistente, así que sigue en vigor en todas las llamadas
posteriores a `predict()`/`track()` hasta que vuelvas a fijarlo. SmolVLM2 no
necesita sobrescribir el parser en LibreYOLO: sigue la misma salida de
plantilla de chat más JSON que el valor por defecto compartido del nivel, así
que ni su prompt de detección ni su formato de cajas son específicos de la
familia. Todas las detecciones llevan la misma confianza de marcador de
posición, así que el filtrado por `conf` es todo o nada en vez de una
ordenación; `iou` sí tiene efecto: descarta una caja posterior de la misma
clase en cuanto solapa por encima del umbral con otra ya conservada, porque un
generador que se repite puede emitir cajas casi duplicadas para un mismo
objeto. SmolVLM2 también responde preguntas libres mediante `chat()`, la misma
vía de escape documentada en la factoría `LibreVLM`. La CLI de LibreYOLO no
cubre este nivel: no existe una forma `libreyolo predict model=...` para él.
Consulta [predicción](/docs/predict) para fuentes, streaming y manejo de
resultados.

## Variantes

Un solo tamaño en el registro: SmolVLM2-500M-Video-Instruct, que se carga como
`LibreVLM("smolvlm2-500m")`. SmolVLM2 detecta peor que los modelos de grounding
específicos de este nivel; el propio envoltorio de LibreYOLO lo describe como
una demostración de que una familia nueva no necesita parseo a medida para
funcionar aquí, no como su opción de vocabulario abierto más potente.

LibreYOLO no entrena, valida ni exporta SmolVLM2: `train()`, `val()` y
`export()` lanzan todos `NotImplementedError` para todas las familias de este
nivel (consulta el nivel de soporte más arriba). Haz el fine-tuning de SmolVLM2
upstream y carga los pesos resultantes si necesitas llevar integrado un
vocabulario propio; revisa a ojo la salida de `predict()` en vez de una pasada
de validación al estilo COCO, ya que todas las detecciones llevan la misma
confianza de marcador de posición.

## Licencia

<provenance-box></provenance-box>

## Cita

<citation-block />
