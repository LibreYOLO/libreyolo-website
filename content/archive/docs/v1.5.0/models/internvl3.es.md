---
title: InternVL3
families:
  - internvl3
seo_title: 'InternVL3: detección de vocabulario abierto en LibreYOLO'
description: >-
  Usa InternVL3 en LibreYOLO para detección de objetos de vocabulario abierto.
  Predice con cualquier etiqueta de texto; el entrenamiento, la validación y la
  exportación no están soportados.
lead: >-
  InternVL3 es un modelo de lenguaje grande multimodal nativo publicado por
  OpenGVLab que aprende visión y lenguaje de forma conjunta en una única etapa
  de preentrenamiento. LibreYOLO lo envuelve como detector de objetos de
  vocabulario abierto: cualquier lista de etiquetas de texto se convierte en el
  conjunto de clases, sin cabeza fija y sin necesidad de hacer fine-tuning.
keywords:
  - InternVL3
  - InternVL
  - modelo de visión y lenguaje
  - detección de vocabulario abierto
  - VLM
  - OpenGVLab
  - detectar objetos con texto
  - LibreVLM
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreInternVL3, SAMPLE_IMAGE


        model = LibreInternVL3(size="2b")


        # Vocabulario abierto: vale cualquier palabra, no una cabeza de clases

        # fija. Persiste en cada predict()/track() posterior hasta volver a
        fijarlo.

        model.set_classes(["person", "bicycle", "dog"])

        result = model(SAMPLE_IMAGE, save=True)


        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Chat directo
      language: python
      code: |
        from libreyolo import LibreInternVL3, SAMPLE_IMAGE

        model = LibreInternVL3(size="2b")

        # La vía de escape bajo la comodidad de la detección: preguntas libres,
        # recuentos o cualquier prompt que el envoltorio de cajas no cubra.
        text = model.chat(SAMPLE_IMAGE, "Describe the scene in one sentence.")
        print(text)
source_hash: 6305f020d3079d71
---

## Instalación

InternVL3 necesita el extra `vlm`, que arrastra `transformers` para el backbone
de plantillas de chat.

```bash
pip install "libreyolo[vlm]"
```

## Predicción

`LibreInternVL3` es una clase de Python, no un checkpoint `.pt`: no se carga a
través de la factoría `LibreYOLO()`, y la CLI `libreyolo` no lo resuelve. La
factoría `LibreVLM(...)` (`from libreyolo import LibreVLM`) también llega a
esta familia por alias, p. ej. `LibreVLM("internvl3-2b")`; la clase que se usa
abajo es la que construye. Los pesos vienen de los propios repositorios `-hf`
de Hugging Face de OpenGVLab, no de un mirror de LibreYOLO; la primera llamada
los descarga y los cachea en local, y antes registra un aviso de licencia único
por los pesos de Qwen, que son de acceso restringido.

<code-tabs name="predict" />

`result.boxes` lleva las detecciones parseadas igual que en cualquier otra
familia. La confianza es un marcador de posición: InternVL3 no emite ninguna
puntuación por caja, así que todas las detecciones reciben la misma confianza
constante, y `conf=` solo descarta las filas por debajo de esa constante, no
las ordena. `iou` descarta las cajas casi duplicadas de la misma clase por
encima del solape indicado, un efecto secundario de que el decodificado voraz
repita un objeto; no es una pasada de NMS por clases. Si te saltas
`set_classes()`, el vocabulario cae por defecto en los nombres de COCO-80.
Consulta [predicción](/docs/predict) para fuentes, streaming y manejo de
resultados.

## Variantes

Tres tamaños: 1b, 2b y 8b, todos ellos checkpoints `-hf` nativos de OpenGVLab
(un backbone de LLM Qwen, no la arquitectura de dos torres que describe el
paper original de InternVL). El harness de benchmarks de LibreYOLO no ha medido
esta familia, así que no hay cifras de precisión publicadas con las que
compararlos; elige un tamaño según tu propio presupuesto de cómputo.

LibreYOLO expone esta familia solo para predicción. `train()`, `val()` y
`export()` lanzan todos `NotImplementedError`: haz el fine-tuning upstream y
carga el resultado, la validación sobre dataset se omite porque una confianza
de marcador de posición haría engañoso el mAP de COCO, y la exportación queda
fuera del alcance para un modelo generativo sin state dict que trazar.

## Licencia

<provenance-box>

El código propio de InternVL3 es MIT, permisivo y utilizable en productos
comerciales y de código cerrado. Los checkpoints `-hf` que carga esta familia
llevan un backbone de LLM Qwen y se licencian aparte, bajo la Qwen License de
Alibaba Cloud: libre de usar, modificar y redistribuir con la obligación de
atribuir mediante un "Built with Qwen" o "Improved using Qwen", y con un techo
de 100 millones de usuarios activos mensuales en el uso comercial por encima
del cual hace falta la autorización de la propia Alibaba. LibreYOLO no aloja ni
redistribuye estos pesos: `LibreInternVL3` descarga el tamaño correspondiente
directamente de `OpenGVLab/InternVL3-<size>-hf` en Hugging Face la primera vez
que se ejecuta, y registra un aviso único por la Qwen License antes de esa
descarga.

</provenance-box>

## Cita

<citation-block />
