---
title: LFM2-VL
families:
  - lfm2vl
seo_title: 'LFM2-VL: detección de vocabulario abierto en LibreYOLO'
description: >-
  Usa LFM2-VL en LibreYOLO para detección de objetos de vocabulario abierto en
  el dispositivo. Predice con cualquier etiqueta de texto; el entrenamiento, la
  validación y la exportación no están soportados.
lead: >-
  LFM2-VL es un modelo de visión y lenguaje compacto, pensado para ejecutarse en
  el propio dispositivo y publicado por Liquid AI. LibreYOLO lo envuelve como
  detector de objetos de vocabulario abierto: cualquier lista de etiquetas de
  texto se convierte en el conjunto de clases, sin cabeza fija y sin necesidad
  de hacer fine-tuning.
keywords:
  - LFM2-VL
  - LFM2
  - Liquid AI
  - modelo de visión y lenguaje
  - detección de vocabulario abierto
  - VLM
  - VLM en el dispositivo
  - LibreVLM
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreLFM2VL, SAMPLE_IMAGE


        model = LibreLFM2VL(size="450m")


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
        from libreyolo import LibreLFM2VL, SAMPLE_IMAGE

        model = LibreLFM2VL(size="450m")

        # La vía de escape bajo la comodidad de la detección: preguntas libres,
        # recuentos o cualquier prompt que el envoltorio de cajas no cubra.
        text = model.chat(SAMPLE_IMAGE, "Describe the scene in one sentence.")
        print(text)
source_hash: 40237f0ecc0d2cd5
---

## Instalación

LFM2-VL necesita el extra `vlm`, que arrastra `transformers` para el backbone
de plantillas de chat.

```bash
pip install "libreyolo[vlm]"
```

## Predicción

`LibreLFM2VL` es una clase de Python, no un checkpoint `.pt`: no se carga a
través de la factoría `LibreYOLO()`, y la CLI `libreyolo` no lo resuelve. La
factoría `LibreVLM(...)` (`from libreyolo import LibreVLM`) también llega a
esta familia por alias, p. ej. `LibreVLM("lfm2-vl-450m")`; la clase que se usa
abajo es la que construye. Los pesos vienen del propio repositorio de Hugging
Face de Liquid AI, no de un mirror de LibreYOLO; la primera llamada los
descarga y los cachea en local, y registra un aviso de licencia único antes de
hacerlo.

<code-tabs name="predict" />

`result.boxes` lleva las detecciones parseadas igual que en cualquier otra
familia. La confianza es un marcador de posición: LFM2-VL no emite ninguna
puntuación por caja, así que todas las detecciones reciben la misma confianza
constante, y `conf=` solo descarta las filas por debajo de esa constante, no
las ordena. `iou` descarta las cajas casi duplicadas de la misma clase por
encima del solape indicado, un efecto secundario de que el decodificado voraz
repita un objeto; no es una pasada de NMS por clases. Si te saltas
`set_classes()`, el vocabulario cae por defecto en los nombres de COCO-80.
Consulta [predicción](/docs/predict) para fuentes, streaming y manejo de
resultados.

## Variantes

Dos tamaños: 450m y 1.6b, ambos de la release LFM2.5-VL de Liquid AI,
construidos para el despliegue en el propio dispositivo. El harness de
benchmarks de LibreYOLO no ha medido esta familia, así que no hay cifras de
precisión publicadas con las que compararlos; elige un tamaño según tu propio
presupuesto de cómputo.

LibreYOLO expone esta familia solo para predicción. `train()`, `val()` y
`export()` lanzan todos `NotImplementedError`: haz el fine-tuning upstream y
carga el resultado, la validación sobre dataset se omite porque una confianza
de marcador de posición haría engañoso el mAP de COCO, y la exportación queda
fuera del alcance para un modelo generativo sin state dict que trazar.

## Licencia

<provenance-box>

La LFM Open License v1.0 permite el uso comercial, la reproducción y la
modificación, pero solo por debajo de un umbral de 10 millones de dólares de
ingresos anuales; una entidad jurídica que alcance o supere ese umbral no
queda licenciada en absoluto bajo este acuerdo para uso comercial, y debe
ponerse en contacto directamente con Liquid AI. Las organizaciones sin ánimo
de lucro cualificadas quedan exentas del umbral para uso no comercial o de
investigación. LibreYOLO no distribuye código fuente de LiquidAI, ya que el
modelo se carga a través de la biblioteca `transformers` (Apache-2.0), y no
aloja ni redistribuye los pesos: `LibreLFM2VL` descarga el tamaño
correspondiente directamente del propio repositorio de Hugging Face de Liquid
AI la primera vez que se ejecuta, y registra un aviso único antes de esa
descarga.

</provenance-box>

## Cita

<citation-block />
