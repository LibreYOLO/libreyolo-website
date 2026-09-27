---
title: SenseNova-Vision
families:
  - sensenovavision
seo_title: 'SenseNova-Vision en LibreYOLO: 7 tareas, un solo checkpoint'
description: >-
  Usa SenseNova-Vision en LibreYOLO para detección, segmentación, panóptica,
  pose, puntos, profundidad y OCR desde un único checkpoint generativo guiado
  por prompt.
lead: >-
  SenseNova-Vision es un modelo multimodal unificado que plantea las tareas de
  visión como generación guiada por prompt sobre un decodificador compartido:
  las cajas, los puntos, los keypoints y las palabras de OCR salen como texto
  etiquetado, y los mapas de profundidad, de máscara y panópticos salen como
  imágenes que renderiza un decodificador. LibreYOLO lo carga a través de
  LibreVLM y admite siete tareas desde un único checkpoint de 7B.
keywords:
  - SenseNova-Vision
  - SenseTime
  - modelo multimodal unificado
  - Bagel
  - detección por prompt
  - percepción densa
  - segmentación referring
  - segmentación panóptica python
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM

        model = LibreVLM("sensenova-vision", task="detect")
        model.set_classes(["bird", "boat"])
        result = model.predict("image.jpg")
        print(result.boxes.xyxy)

        # set_task() cambia de tarea sobre el mismo modelo ya cargado.
        model.set_task("depth")
        result = model.predict("image.jpg")
        depth = result.depth_map.data
    - label: Segmentación referring y panóptica
      language: python
      code: >
        from libreyolo import LibreVLM


        model = LibreVLM("sensenova-vision", task="segment")

        # La segmentación es referring: necesita una frase objetivo, no una
        lista de clases.

        model.set_classes(["the person furthest to the right"])

        result = model.predict("street.jpg")

        mask = result.masks.data[0]


        model.set_task("panoptic")

        # Sin vocabulario propio, la panóptica recurre a las categorías
        panópticas

        # de COCO con las que se ajustó el checkpoint.

        result = model.predict("street.jpg")

        segment_map = result.panoptic.data

        for segment in result.panoptic.segments_info:
            print(segment)
    - label: 'Puntos, pose y OCR'
      language: python
      code: |
        from libreyolo import LibreVLM

        model = LibreVLM("sensenova-vision", task="point")
        model.set_classes(["screw"])
        result = model.predict("board.jpg")
        print(result.points.xy)

        # Sin vocabulario definido, pose recurre a "person".
        model.set_task("pose")
        result = model.predict("gym.jpg")
        print(result.boxes.xyxy, result.keypoints.data.shape)

        model.set_task("ocr")
        result = model.predict("sign.jpg")
        print(result.ocr.texts)
source_hash: 8749277e1910baa4
---

## Instalación

SenseNova-Vision necesita su propio extra, que arrastra `accelerate` para el dispatch de modelos grandes que requiere este checkpoint y, en plataformas que no son macOS, `bitsandbytes` para la carga en 4 bits.

```bash
pip install "libreyolo[sensenova]"
```

El checkpoint está replicado en Hugging Face bajo la organización propia de LibreYOLO y se descarga automáticamente en el primer uso; es CC BY-NC 4.0, solo para uso no comercial, y el cargador imprime ese aviso antes de cada descarga automática. Consulta Licencia más abajo.

## Predicción

<code-tabs name="predict" />

Cada predicción es un decodificado por difusión sobre el backbone Bagel-MoT compartido, así que es un modelo de capacidades y no de tiempo real: espera una latencia por imagen bastante más alta que la de un detector o un segmentador hechos a medida. `dtype="auto"` (el valor por defecto) carga bf16 en una GPU con memoria suficiente y recurre a la cuantización NF4 de 4 bits en el resto de casos, lo que necesita `bitsandbytes`; pasa `dtype="bf16"` para forzar la precisión completa en una GPU lo bastante grande. `noise_seed=42` en la construcción fija la semilla del muestreador de difusión para obtener salidas densas reproducibles; pasa `noise_seed=None` para desactivar la semilla.

Las siete tareas comparten un único checkpoint cargado: `set_task()` alterna entre ellas sin volver a cargarlo. `set_classes()` define el vocabulario activo; la detección, los puntos, la pose y la panóptica aceptan una lista de clases, mientras que la segmentación es referring y necesita exactamente la frase que hay que aislar. Cada tarea devuelve el objeto `Results` estándar con un contenido distinto relleno: `boxes` para detect, `points` para point, `boxes` y `keypoints` para pose, `ocr` para OCR, `depth_map` para depth, `masks` para segment y `panoptic` (con `segments_info`) para panoptic. Consulta [predicción](/docs/predict) para fuentes, streaming y manejo de resultados.

## Checkpoints

<checkpoint-table />

## Licencia

<provenance-box></provenance-box>

## Cita

<citation-block />
