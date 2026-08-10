---
title: PicoSAM3
families: [picosam3]
seo_title: "PicoSAM3: segmentación en el edge guiada por cajas en LibreYOLO"
description: "Usa PicoSAM3 en LibreYOLO para segmentación de regiones guiada por cajas en sensores edge. Instala, predice y exporta el checkpoint pico bajo Apache-2.0."
lead: "PicoSAM3 es una CNN compacta destilada de SAM 2.1 y SAM 3, pensada para la segmentación de regiones de interés guiada por cajas en sensores como el Sony IMX500. LibreYOLO lo admite mediante una factoría LibreSAM dedicada, separada de la factoría de detectores LibreYOLO(), y solo con prompts de caja."
keywords: [PicoSAM3, Segment Anything, "segmentación en el edge", "región de interés", "prompt de caja", "inferencia en el sensor", IMX500, "destilación de conocimiento"]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Prompt de caja
      language: python
      code: |
        from libreyolo import LibreSAM, SAMPLE_IMAGE

        # PicoSAM3 tiene un único tamaño, "pico", así que no hace falta otro alias.
        model = LibreSAM("picosam3")

        # bboxes= es el único prompt admitido: [x1, y1, x2, y2] o una lista de
        # cajas, con una máscara por caja. Cada caja se expande un 10 %, se hace
        # cuadrada, se recorta a la imagen y se escala a 96x96 antes de la CNN.
        result = model.predict(SAMPLE_IMAGE, bboxes=[300, 200, 900, 700])
        print(result.masks.xy)      # un polígono por máscara
        print(result.boxes.xyxy)    # caja ajustada derivada de la máscara
    - label: Codifica una vez, lanza muchos prompts
      language: python
      code: |
        from libreyolo import LibrePicoSAM3, SAMPLE_IMAGE

        model = LibrePicoSAM3()

        # set_image() cachea la imagen de origen; PicoSAM3 ejecuta un forward
        # completo de la CNN por caja, así que esto ahorra la carga y decodificación
        # de la imagen, no un paso de encoder como en las demás familias SAM.
        model.set_image(SAMPLE_IMAGE)
        a = model.predict(bboxes=[300, 200, 900, 700])
        b = model.predict(bboxes=[100, 100, 400, 400])
        model.reset_image()
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibrePicoSAM3

        model = LibrePicoSAM3()
        model.export(format="onnx", output_path="LibrePicoSAM3pico.onnx")

        # opset (13 por defecto) y dynamic (True por defecto, solo el eje de batch)
        # son los únicos argumentos de exportación que acepta esta familia.
    - label: Usar el archivo exportado
      language: python
      code: |
        import numpy as np
        import onnxruntime as ort

        # PicoSAM3 exporta su CNN de ROI de 96x96 en crudo: roi_image -> mask_logits.
        # Aquí no hay pre/posprocesado del lado de LibreYOLO que reutilizar, porque
        # export() no se enruta de vuelta por LibreYOLO() como sí ocurre con el
        # checkpoint de un detector.
        session = ort.InferenceSession("LibrePicoSAM3pico.onnx")
        name = session.get_inputs()[0].name
        outputs = session.run(None, {name: np.zeros((1, 3, 96, 96), dtype=np.float32)})

        for meta, array in zip(session.get_outputs(), outputs):
            print(meta.name, array.shape)
---

## Instalación

PicoSAM3 necesita el extra `sam`: la descarga de pesos propia de LibreYOLO
sigue pasando por las herramientas de Hugging Face de `transformers`, aunque la
inferencia se ejecute sobre una CNN nativa que no usa `transformers`.

```bash
pip install "libreyolo[sam]"
```

## Predicción

`LibreSAM(...)` (o el `LibrePicoSAM3(...)` específico de la familia) es un punto
de entrada distinto de `LibreYOLO(...)`: devuelve un segmentador guiado por
prompts en lugar de un detector, porque aquí un forward pass no significa nada
sin un prompt. No hay comando de CLI `libreyolo predict` para esta familia; usa
la API de Python.

<code-tabs name="predict" />

PicoSAM3 solo acepta `bboxes=`; pasar `points=`, `labels=`, `masks=`, `text=`,
`multimask=True` u omitir la caja para segmentarlo todo lanza en todos los casos
un `ValueError` claro, porque ninguno de esos modos existe en el modelo
original. `conf` filtra por la calidad de máscara predicha (IoU), no por una
confianza de detección, y tiene que estar entre `0.0` y `1.0`. Cada máscara
lleva el id de clase `0`, con nombre `"object"`. `train()`, `val()` y `track()`
lanzan `NotImplementedError`; usa LibreSAM2 o LibreSAM3 para prompts de punto,
de texto, de máscara o de segmentarlo todo. Consulta
[predicción](/docs/predict) para los tipos de fuente.

## Variantes

Un único tamaño, pico, con una entrada de ROI fija de 96 px: PicoSAM3 ejecuta un
forward completo de la CNN por cada caja en lugar de codificar la imagen entera
una sola vez.

## Exportación

<export-matrix />

PicoSAM3 es la única familia del nivel SAM que exporta: lleva su CNN de ROI de
96x96 en crudo a ONNX, `roi_image -> mask_logits`, sin NMS ni posprocesado de
máscaras integrados. Las demás familias SAM lanzan `NotImplementedError` en
`export()`, porque su separación entre encoder y decoder todavía no tiene un
contrato de exportación definido para runtime. Un grafo de PicoSAM3 exportado no
se vuelve a cargar mediante `LibreYOLO()`; ejecútalo directamente con un runtime
como `onnxruntime`, aplicando el mismo preprocesado de ROI cuadrada con un 10 %
de margen que se muestra arriba.

<code-tabs name="export" />

## Checkpoints

Todos los archivos de pesos publicados de esta familia.

<checkpoint-table />

## Licencia

<provenance-box>

PicoSAM3 se destila a partir de SAM 2.1 y SAM 3 como modelos maestros.
LibreYOLO no incorpora ni redistribuye el código ni los pesos de ninguno de los
dos maestros en esta familia; solo se distribuyen la CNN estudiante compacta y
su checkpoint convertido.

</provenance-box>

## Cita

<citation-block />
