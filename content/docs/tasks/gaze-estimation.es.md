---
title: Estimación de la mirada
seo_title: "Estimación de la mirada en LibreYOLO"
description: "Estima el pitch y el yaw de la mirada de cada cara en LibreYOLO. Predice desde Python o la CLI, lee los ángulos en radianes y exporta la cabeza de mirada a ONNX."
lead: "La estimación de la mirada devuelve una dirección de mirada para cada cara de una imagen. LibreYOLO la modela como una tarea en dos etapas: primero se ejecuta un detector de caras, y una cabeza de mirada lee el pitch y el yaw de cada recorte de cara que devuelve."
keywords: [estimación de la mirada python, seguimiento ocular python, eye tracking, dirección de la mirada, pitch yaw mirada, L2CS-Net, pose de la cabeza]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Si no se indica face_detector, la predicción recurre al detector
        # incluido en OpenCV, así que no se descarga nada más allá del checkpoint.
        model = LibreYOLO("LibreL2CSr50.pt")
        result = model(SAMPLE_IMAGE)

        gaze = result.gaze
        print(gaze.pitch, gaze.yaw)              # radianes, una fila por cara
        print(gaze.pitch_deg, gaze.yaw_deg)      # los mismos ángulos en grados
        print(gaze.direction_3d)                 # vectores unitarios (N, 3)
    - label: CLI
      language: bash
      code: |
        # A diferencia de la vía de Python, la CLI no tiene fallback automático:
        # los modelos de mirada exigen un detector de caras explícito, y debe
        # ser un detector de LibreYOLO cuyos boxes sean caras.
        libreyolo predict model=LibreL2CSr50.pt source=photo.jpg face_detector=face-detector.pt save=True
    - label: Elegir la fuente de caras
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreL2CSr50.pt")

        # Pasa a la cabeza de mirada los boxes de un detector que ya hayas ejecutado.
        result = model(SAMPLE_IMAGE, face_boxes=[[34, 12, 90, 80]])

        # O nombra uno de los detectores incluidos.
        result = model(SAMPLE_IMAGE, face_detector="yunet")
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreL2CSr50.pt")
        model.export(format="onnx")
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreL2CSr50.pt format=onnx
---

## Definición

La estimación de la mirada devuelve dos ángulos por cara. `result.gaze` es un
payload `Gaze` de forma `(N, 2)`, con el pitch en la columna 0 y el yaw en la
columna 1, en radianes, alineado fila a fila con `result.boxes`, los boxes de
cara detectados. La convención es la que usa L2CS-Net: un yaw positivo gira la
mirada hacia la izquierda del sujeto, y un pitch positivo la gira hacia abajo.

El mismo payload expone `pitch_deg` y `yaw_deg` para los grados, y
`direction_3d`, un vector unitario `(N, 3)` en el sistema de referencia de la
cámara con columnas `(x, y, z)`.

Como la tarea es en dos etapas, una predicción depende de dos modelos. Las caras
que el detector no encuentra no tienen fila de mirada, y los boxes que coloca mal
producen ángulos a partir de una cara mal recortada. La clave canónica de la
tarea es `gaze`; `gaze-estimation` se normaliza a ella.

## Modelos

[L2CS-Net](/docs/models/l2cs) es la única familia que sirve esta tarea. Combina
un tronco ResNet con dos cabezas paralelas de clasificación por bins de ángulo,
una para el pitch y otra para el yaw, sobre recortes de cara de 448x448.
Arquitectónicamente admite cinco profundidades de backbone, y una, la ResNet-50,
tiene un checkpoint publicado.

Los pesos llevan una restricción de licencia. Están entrenados con Gaze360, cuya
licencia solo permite el uso investigador y no comercial y prohíbe la
redistribución, así que LibreYOLO no replica nada para esta familia. El único
checkpoint que la biblioteca puede descargar automáticamente viene directamente
de la distribución en Google Drive de los propios autores, a través de `gdown`,
después de imprimir los términos de la licencia. Lee
[L2CS-Net](/docs/models/l2cs) antes de desplegarlo.

Esa vía de descarga necesita el extra `gaze`:

```bash
pip install "libreyolo[gaze]"
```

Sin él, la biblioteca imprime instrucciones de descarga manual en lugar de
intentar la transferencia. Predecir con un checkpoint que ya tengas, y
exportarlo, no necesita ningún extra.

## Predicción

<code-tabs name="predict" />

La fuente de caras se elige de una de estas tres formas. `face_boxes` pasa boxes
que ya has calculado y se salta la detección. `face_detector` acepta `"auto"`,
`"haar"`, `"yunet"`, un modelo de detección de LibreYOLO o un callable normal, y
se puede fijar en el constructor o por llamada. Si se deja sin fijar en Python,
la predicción recurre al detector incluido en OpenCV, así que una llamada pelada
funciona sin cablear nada. En OpenCV 4 ese detector es el cascade de Haar que
viene dentro del wheel, que no necesita ninguna descarga; en OpenCV 5, donde se
eliminó la API de Haar, es YuNet, que descarga una vez un archivo de modelo
pequeño del zoo de OpenCV.

La CLI no comparte ese fallback. `libreyolo predict` rechaza un modelo de mirada
sin `face_detector=`, y el valor que acepta es un nombre de detector de LibreYOLO
o una ruta a un checkpoint. Consulta [predicción](/docs/predict) para las
fuentes, el streaming y el manejo de resultados.

## Entrenamiento

Ninguna familia de esta tarea se entrena dentro de LibreYOLO.
`LibreL2CS.train()` lanza una excepción: entrena en el proyecto original de
L2CS-Net y carga aquí el state dict resultante.

## Validación

La validación contra datasets de ground truth de mirada queda fuera de alcance, y
`val()` lanza una excepción en lugar de devolver métricas que no ha calculado. No
hay diccionario `metrics/` para esta tarea. Evalúa en el proyecto original, sobre
el dataset para el que se entrenó el checkpoint.

## Exportación

<code-tabs name="export" />

El contrato de exportación de la mirada cubre ONNX, TorchScript, ExecuTorch,
TensorRT y OpenVINO. Lo que sale de la biblioteca es únicamente el tronco ResNet
y las dos cabezas de bins de ángulo: el grafo toma un recorte de cara de 448x448
ya preprocesado y devuelve los logits crudos de yaw y pitch. La detección de
caras, el recorte, el softmax, la esperanza sobre los bins y la conversión a
ángulos se quedan todos en Python, en `libreyolo.models.l2cs.utils`. Consulta
[exportación](/docs/export) para los formatos y sus argumentos.
