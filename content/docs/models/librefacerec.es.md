---
title: LibreFaceRec
families: [facerec]
seo_title: "LibreFaceRec: reconocimiento y verificación facial"
description: "Usa LibreFaceRec en LibreYOLO para detección de caras, embeddings y verificación. Instala y predice; los pesos de embedding son Apache-2.0."
lead: "LibreFaceRec es la tarea de embeddings faciales de LibreYOLO: un detector de caras localiza y alinea las caras, y una cabeza de reconocimiento produce un embedding de identidad normalizado con L2 para verificación o búsqueda."
keywords: [LibreFaceRec, "reconocimiento facial python", "embeddings faciales", "verificación facial", "comparar dos caras", ArcFace]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Los nombres librefacerec-* dirigen a esta familia sin importar el
        # sufijo del archivo y se descargan de la organización de LibreYOLO en
        # Hugging Face en el primer uso, junto con el detector de caras por defecto.
        model = LibreYOLO("librefacerec-l.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.embeddings.data.shape)   # (N, D), normalizado con L2
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=librefacerec-l.onnx source=face.jpg
    - label: Verificar
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("librefacerec-l.onnx")

        # Compara la cara más prominente de cada imagen mediante la similitud
        # coseno de sus embeddings normalizados con L2.
        result = model.verify("person_a.jpg", "person_b.jpg", threshold=0.4)
        print(result["similarity"], result["same_person"])
    - label: Búsqueda en galería
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("librefacerec-l.onnx")

        query = model("query.jpg").embeddings          # las caras de esta imagen
        gallery = model.embed(["a.jpg", "b.jpg", "c.jpg"])   # (N_total, D)

        # Similitudes coseno (query_faces, N_total).
        scores = query.similarity(gallery)
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("librefacerec-l.onnx")
        model.export(format="onnx")
---

## Instalación

La cabeza de reconocimiento de LibreFaceRec funciona a través de
`onnxruntime`, que no forma parte de la instalación base.

```bash
pip install "libreyolo[onnx]"
```

## Predicción

<code-tabs name="predict" />

La detección y el reconocimiento son dos grafos ONNX distintos detrás de una
sola llamada: un detector de caras localiza y alinea cada cara a un recorte
canónico, y la cabeza de reconocimiento devuelve un embedding normalizado con
L2 por cara. Si no se toca nada, `predict()` descarga y empareja
automáticamente el detector por defecto incluido. `face_detector` acepta un
callable, un modelo de detección de LibreYOLO o una instancia de
`FaceDetector`; `face_boxes` se salta la detección por completo con bounding
boxes que ya tengas. `result.embeddings` contiene una fila por cara detectada,
alineada con `result.boxes`; su método `.similarity()` calcula la similitud
coseno frente a otro embedding o a toda una galería en una sola llamada. Para
comparar dos imágenes directamente en lugar de dos embeddings ya calculados,
`model.verify(image_a, image_b)` ejecuta la detección y el embedding en ambas
y compara su cara con más confianza. Se puede sustituir por cualquier otro
modelo de reconocimiento ONNX que siga la convención de ArcFace (entra un
recorte alineado, salen embeddings `(N, D)`) pasando la ruta de su archivo en
lugar de un nombre `librefacerec-*`. Consulta
[predicción](/docs/predict) para fuentes, streaming y manejo de resultados.

## Exportación

<export-matrix />

LibreFaceRec ya envuelve un grafo ONNX preexportado; volver a exportarlo a
otro formato no está implementado.

## Licencia

<provenance-box>

El detector de caras por defecto incluido es un segundo artefacto bajo una
segunda licencia: YuNet, de OpenCV Zoo, MIT, copyright Shiqi Yu. No se ha
portado código de arquitectura de ninguno de los dos proyectos; ambos grafos se
consumen de forma opaca a través de `onnxruntime`, así que el propio wrapper de
LibreYOLO no contiene código de terceros y es MIT de principio a fin.

</provenance-box>

## Cita

<citation-block />
