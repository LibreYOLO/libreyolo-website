---
title: Florence-2
families:
  - florence2
seo_title: 'Florence-2 en LibreYOLO: detección de vocabulario abierto'
description: >-
  Florence-2 en LibreYOLO: instalación, definición de un vocabulario abierto y
  predicción de cajas con el modelo de visión con licencia MIT de Microsoft.
lead: >-
  Florence-2 es el modelo fundacional de visión de Microsoft, que se controla
  con un token de tarea en lugar de pasar por una cabeza de detección fija.
  LibreYOLO lo envuelve como detector de objetos de vocabulario abierto: la
  lista de clases se indica en el momento de predecir.
keywords:
  - Florence-2
  - modelo de visión y lenguaje
  - detección de vocabulario abierto
  - detectar objetos sin entrenar
  - florence 2 python
  - grounding
  - Microsoft
  - VLM
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("florence-2-base")
        model.set_classes(["car", "person", "traffic light"])
        result = model.predict(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Vídeo
      language: python
      code: |
        from libreyolo import LibreVLM

        model = LibreVLM("florence-2-base")
        model.set_classes(["car", "person", "traffic light"])

        # Cualquier fuente que acepta la biblioteca: archivo, carpeta, URL,
        # índice de webcam, stream RTSP o una lista .streams
        for result in model.predict("clip.mp4", stream=True, save=True):
            print(len(result.boxes))
source_hash: ad26d9056465d662
---

## Instalación

Florence-2 pertenece al nivel VLM-como-detector de LibreYOLO, una superficie de
producto separada de las familias basadas en checkpoints y con su propia
factoría. Necesita el extra `vlm`.

```bash
pip install "libreyolo[vlm]"
```

## Predicción

Los pesos se descargan de Hugging Face en el primer uso y se guardan en la
caché local. LibreYOLO descarga la resubida del checkpoint hecha por
florence-community en lugar del repositorio original `microsoft/Florence-2-*`;
consulta Licencia para saber por qué.

<code-tabs name="predict" />

Esta familia se carga con la factoría `LibreVLM()`, no con `LibreYOLO()`: las
familias VLM no declaran ningún cargador de checkpoints, así que el enrutado por
sufijo de archivo que se describe en otras páginas de modelos no se aplica aquí.
`set_classes()` define el vocabulario que se le pide a Florence-2 que encuentre
en la imagen; es persistente, así que sigue vigente en todas las llamadas
posteriores a `predict()`/`track()` hasta que vuelvas a definirlo. El objeto
`Results` devuelto lleva `boxes` con la misma forma que cualquier otra familia,
pero todas las detecciones llevan la misma confianza de relleno, así que filtrar
por `conf` es todo o nada en lugar de una ordenación, y `iou` no tiene efecto:
el wrapper de Florence-2 construye la lista de detecciones directamente a partir
de la salida parseada del token de tarea, sin ningún paso de deduplicación.
Aquí `chat()` lanza `NotImplementedError`, porque Florence-2 se controla con el
token de tarea `<OPEN_VOCABULARY_DETECTION>` y no con una plantilla de chat. La
CLI de LibreYOLO no cubre este nivel: no existe una forma
`libreyolo predict model=...` para él. Consulta
[predicción](/docs/predict) para fuentes, streaming y manejo de resultados.

## Variantes

Dos tamaños: Florence-2-base y Florence-2-large, ambos a 768 px, que se cargan
como `LibreVLM("florence-2-base")` o `LibreVLM("florence-2-large")`. LibreYOLO
no ha publicado ningún benchmark que compare la precisión entre ellos.

LibreYOLO no entrena, valida ni exporta Florence-2: `train()`, `val()` y
`export()` lanzan `NotImplementedError` en todas las familias de este nivel
(consulta el nivel de soporte de arriba). Haz fine-tuning de Florence-2 upstream
y carga los pesos resultantes si necesitas un vocabulario personalizado
integrado; revisa a ojo la salida de `predict()` en lugar de recurrir a una
pasada de validación al estilo COCO, ya que todas las detecciones llevan la
misma confianza de relleno.

## Licencia

<provenance-box></provenance-box>

## Cita

<citation-block />
