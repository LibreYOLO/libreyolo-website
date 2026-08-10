---
title: L2CS-Net
families: [l2cs]
seo_title: "L2CS-Net: estimación de la mirada en LibreYOLO"
description: "Usa L2CS-Net en LibreYOLO para estimar el pitch y el yaw de la mirada en dos etapas. Instala, predice y exporta; el checkpoint de Gaze360 es solo para investigación."
lead: "L2CS-Net es un estimador de mirada de dos etapas: un detector de caras las localiza, y un tronco ResNet con dos cabezas de clasificación por bins de ángulo predice el pitch y el yaw de cada cara. LibreYOLO lo integra solo para inferencia."
keywords: [L2CS-Net, "estimación de la mirada", "hacia dónde mira una persona", "eye tracking python", "pitch yaw mirada", Gaze360, "detección de caras python"]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Sin face_detector: recurre al detector de caras incluido en
        # OpenCV (Haar en OpenCV 4, YuNet en OpenCV 5), así que esto
        # funciona sin más descarga que el propio checkpoint de L2CS.
        model = LibreYOLO("LibreL2CSr50.pt")
        result = model(SAMPLE_IMAGE)

        print(result.gaze.pitch, result.gaze.yaw)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreL2CSr50.pt source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg save=True
    - label: Origen de las caras
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreL2CSr50.pt")

        # Pasa a L2CS los boxes de un detector que ya hayas ejecutado.
        result = model(SAMPLE_IMAGE, face_boxes=[[34, 12, 90, 80]])

        # O indica un detector de caras incluido concreto.
        result = model(SAMPLE_IMAGE, face_detector="yunet")
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreL2CSr50.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreL2CSr50.pt format=onnx
    - label: Usar el archivo exportado
      language: python
      code: |
        import numpy as np
        import onnxruntime as ort

        # El grafo exportado es solo el tronco ResNet y las dos cabezas
        # de bins de ángulo: recibe un recorte de cara de 448x448 ya
        # preprocesado y devuelve (yaw_logits, pitch_logits) en crudo,
        # no ángulos decodificados. El softmax, la esperanza sobre los
        # bins y la conversión a grados se quedan en Python; consulta
        # libreyolo.models.l2cs.utils.bin_logits_to_angles.
        session = ort.InferenceSession("LibreL2CSr50.onnx")
        name = session.get_inputs()[0].name
        yaw_logits, pitch_logits = session.run(
            None, {name: np.zeros((1, 3, 448, 448), dtype=np.float32)}
        )
---

## Instalación

L2CS-Net no necesita ningún extra para construir un modelo, predecir con él o
exportarlo si ya tienes su checkpoint.

```bash
pip install libreyolo
```

El único checkpoint que LibreYOLO puede descargar automáticamente, un
ResNet-50 entrenado con Gaze360, se descarga con `gdown` en lugar de desde un
mirror HTTP normal, porque vive en el Google Drive del autor y no en la
organización LibreYOLO. Ese camino necesita el extra `gaze`:

```bash
pip install "libreyolo[gaze]"
```

Sin él, LibreYOLO imprime instrucciones de descarga manual en lugar de fallar
en silencio.

## Predicción

<code-tabs name="predict" />

L2CS-Net es un estimador de dos etapas: primero se ejecuta un detector de
caras, y la cabeza de mirada lee el pitch y el yaw de cada recorte de cara que
este devuelve. Si no indicas nada, la predicción recurre al detector incluido
en OpenCV, así que una llamada pelada funciona sin descargas adicionales una
vez tienes en la mano el checkpoint de L2CS. `face_boxes` acepta boxes de un
detector que ya hayas ejecutado; `face_detector` acepta `"auto"`, `"haar"`,
`"yunet"`, un modelo de detección de LibreYOLO o un callable corriente.
`result.gaze` lleva el pitch y el yaw en radianes, alineados fila a fila con
`result.boxes`, los boxes de las caras detectadas. Consulta
[predicción](/docs/predict) para fuentes, streaming y manejo de resultados.

## Variantes

Cinco profundidades de backbone comparten una misma resolución de entrada y
aceptan los mismos argumentos. Gaze360, el dataset detrás del único checkpoint
publicado, entrenó un ResNet-50; las otras cuatro profundidades están
soportadas a nivel de arquitectura, pero no tienen pesos publicados que cargar.

## Exportación

<export-matrix />

<code-tabs name="export" />

## Licencia

<provenance-box>

LibreYOLO no aloja ni replica ningún checkpoint de L2CS: en la organización de
LibreYOLO en Hugging Face no hay nada de esta familia, a diferencia de la
mayoría de las demás familias de este sitio. El único checkpoint que la
biblioteca puede descargar automáticamente viene directamente de la
distribución en el Google Drive del propio autor, tras el aviso de licencia de
Gaze360 que se imprime antes de que empiece la transferencia, y no es la copia
«republicada en huggingface.co/LibreYOLO» que da a entender el resumen de
arriba.

</provenance-box>
