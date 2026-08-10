---
title: Kosmos-2
families: [kosmos2]
seo_title: "Kosmos-2 en LibreYOLO: detección de objetos con grounding"
description: "Kosmos-2 en LibreYOLO: instalación, definición de un vocabulario abierto y predicción de cajas con grounding usando el modelo con licencia MIT de Microsoft."
lead: "Kosmos-2 es el modelo de grounding de Microsoft: genera una descripción de la imagen y después localiza con una caja cada sintagma nominal de esa descripción. LibreYOLO lo envuelve como detector de objetos de vocabulario abierto: la lista de clases se indica en el momento de predecir."
keywords: [Kosmos-2, "modelo de visión y lenguaje", grounding, "detección de vocabulario abierto", "detectar objetos sin entrenar", "kosmos 2 python", Microsoft, VLM]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("kosmos-2")
        model.set_classes(["boat", "person"])
        result = model.predict(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Vídeo
      language: python
      code: |
        from libreyolo import LibreVLM

        model = LibreVLM("kosmos-2")
        model.set_classes(["boat", "person"])

        # Cualquier fuente que acepta la biblioteca: archivo, carpeta, URL,
        # índice de webcam, stream RTSP o una lista .streams
        for result in model.predict("clip.mp4", stream=True, save=True):
            print(len(result.boxes))
---

## Instalación

Kosmos-2 pertenece al nivel VLM-como-detector de LibreYOLO, una superficie de
producto separada de las familias basadas en checkpoints y con su propia
factoría. Necesita el extra `vlm`.

```bash
pip install "libreyolo[vlm]"
```

## Predicción

Los pesos se descargan de Hugging Face en el primer uso y se guardan en la
caché local. LibreYOLO carga directamente el repositorio
`microsoft/kosmos-2-patch14-224` de la propia Microsoft; a diferencia de
Florence-2, aquí no hace falta ninguna resubida de la comunidad.

<code-tabs name="predict" />

Esta familia se carga con la factoría `LibreVLM()`, no con `LibreYOLO()`: las
familias VLM no declaran ningún cargador de checkpoints, así que el enrutado por
sufijo de archivo que se describe en otras páginas de modelos no se aplica aquí.
`set_classes()` define el vocabulario que se le pide a Kosmos-2 que encuentre;
es persistente, así que sigue vigente en todas las llamadas posteriores a
`predict()`/`track()` hasta que vuelvas a definirlo. Kosmos-2 ancla sintagmas
nominales en lugar de emparejar una etiqueta de forma exacta, así que el wrapper
de LibreYOLO acepta una coincidencia parcial: una clase llamada `"boat"` también
coincide con un sintagma generado como "the boats". Todas las detecciones llevan
la misma confianza de relleno, así que filtrar por `conf` es todo o nada en
lugar de una ordenación, y aquí `iou` no tiene efecto, ya que el wrapper
construye la lista de detecciones directamente a partir de las entidades
ancladas, sin ningún paso de deduplicación. `chat()` lanza
`NotImplementedError`, porque Kosmos-2 se controla con un prompt `<grounding>` y
no con una plantilla de chat. La CLI de LibreYOLO no cubre este nivel: no existe
una forma `libreyolo predict model=...` para él. Consulta
[predicción](/docs/predict) para fuentes, streaming y manejo de resultados.

## Variantes

Un solo tamaño: `kosmos-2-patch14-224`, a 224 px, que se carga como
`LibreVLM("kosmos-2")`. Es un modelo de la época de 2023, y el propio wrapper de
LibreYOLO señala que su grounding es más grueso que el de los detectores más
recientes de este nivel.

LibreYOLO no entrena, valida ni exporta Kosmos-2: `train()`, `val()` y
`export()` lanzan `NotImplementedError` en todas las familias de este nivel
(consulta el nivel de soporte de arriba). Haz fine-tuning de Kosmos-2 upstream y
carga los pesos resultantes si necesitas un vocabulario personalizado integrado;
revisa a ojo la salida de `predict()` en lugar de recurrir a una pasada de
validación al estilo COCO, ya que todas las detecciones llevan la misma
confianza de relleno.

## Licencia

<provenance-box></provenance-box>

## Cita

<citation-block />
