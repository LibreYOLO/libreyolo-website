---
title: Qwen3-VL
families: [qwen3vl]
seo_title: "Qwen3-VL en LibreYOLO: detección de vocabulario abierto"
description: "Qwen3-VL en LibreYOLO: instalación, definición de un vocabulario abierto y predicción o chat con el modelo de visión y lenguaje Apache-2.0 de Alibaba."
lead: "Qwen3-VL es el modelo de visión y lenguaje de Alibaba con grounding 2D nativo. LibreYOLO lo envuelve como detector de objetos de vocabulario abierto y expone su chat libre directamente: pásale una lista de clases para detectar, o hazle una pregunta."
keywords: [Qwen3-VL, "modelo de visión y lenguaje", "detección de vocabulario abierto", "qwen3 vl python", "detectar objetos sin entrenar", grounding, Alibaba, VLM]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("qwen3-vl-4b")
        model.set_classes(["forklift", "pallet", "safety vest"])
        result = model.predict(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Chat
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("qwen3-vl-4b")

        # La vía de escape bajo la comodidad de la detección: cualquier pregunta,
        # no solo una consulta de bounding boxes.
        answer = model.chat(SAMPLE_IMAGE, "How many people are wearing a safety vest?")
        print(answer)
---

## Instalación

Qwen3-VL pertenece al nivel VLM-como-detector de LibreYOLO, una superficie de
producto separada de las familias basadas en checkpoints y con su propia
factoría. Necesita el extra `vlm`.

```bash
pip install "libreyolo[vlm]"
```

## Predicción

Los pesos se descargan de Hugging Face en el primer uso y se guardan en la caché
local. `LibreVLM()`, llamado sin argumentos, usa Qwen3-VL-4B por defecto.

<code-tabs name="predict" />

Esta familia se carga con la factoría `LibreVLM()`, no con `LibreYOLO()`: las
familias VLM no declaran ningún cargador de checkpoints, así que el enrutado por
sufijo de archivo que se describe en otras páginas de modelos no se aplica aquí.
`set_classes()` define el vocabulario que se le pide a Qwen3-VL que encuentre; es
persistente, así que sigue vigente en todas las llamadas posteriores a
`predict()`/`track()` hasta que vuelvas a definirlo. Todas las detecciones llevan
la misma confianza de relleno, así que filtrar por `conf` es todo o nada en lugar
de una ordenación; `iou` sí tiene efecto en esta familia, y descarta un bounding
box posterior de la misma clase en cuanto se solapa con uno ya conservado por
encima del umbral, ya que un generador repetitivo puede emitir cajas casi
duplicadas para un mismo objeto. A diferencia de Florence-2 y Kosmos-2, Qwen3-VL
también responde preguntas libres a través de `chat()`, la misma vía de escape
que se documenta en la factoría `LibreVLM`. La CLI de LibreYOLO no cubre este
nivel: no existe una forma `libreyolo predict model=...` para él. Consulta
[predicción](/docs/predict) para fuentes, streaming y manejo de resultados.

## Variantes

Tres tamaños: Qwen3-VL-2B-Instruct, Qwen3-VL-4B-Instruct y Qwen3-VL-8B-Instruct,
que se cargan como `LibreVLM("qwen3-vl-2b")`, `LibreVLM("qwen3-vl-4b")` y
`LibreVLM("qwen3-vl-8b")`. Los tres declaran una entrada nominal de 1024 px, pero
es el propio smart-resize del procesador de Qwen el que decide el lienzo real que
se pasa a la red, así que esa cifra no es una resolución de trabajo fija como sí
lo es en las demás familias de este sitio. LibreYOLO no ha publicado ningún
benchmark que compare la precisión entre los tres tamaños.

LibreYOLO no entrena, valida ni exporta Qwen3-VL: `train()`, `val()` y
`export()` lanzan `NotImplementedError` en todas las familias de este nivel
(consulta el nivel de soporte de arriba). Haz fine-tuning de Qwen3-VL upstream y
carga los pesos resultantes si necesitas un vocabulario personalizado integrado;
revisa a ojo la salida de `predict()` en lugar de recurrir a una pasada de
validación al estilo COCO, ya que todas las detecciones llevan la misma confianza
de relleno.

## Licencia

<provenance-box></provenance-box>

## Cita

<citation-block />
