---
title: Normales de superficie
seo_title: Estimación de normales de superficie en LibreYOLO
description: >-
  Predice un campo denso de normales de superficie a partir de una sola imagen
  en LibreYOLO. Consulta el convenio del sistema de cámara, valida el error
  angular y exporta un modelo.
lead: >-
  La estimación de normales de superficie predice hacia dónde mira cada
  superficie visible. LibreYOLO la expone como la tarea normal, que devuelve un
  campo denso de vectores unitarios sobre el lienzo de la imagen original.
keywords:
  - estimación de normales de superficie python
  - mapa de normales desde una imagen
  - geometría monocular python
  - métrica de error angular
  - predicción densa de normales
last_verified: 1.5.0
snippets:
  predict:
    - label: Predecir un campo de normales
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreMoGe2s-normal.pt")

        result = model(SAMPLE_IMAGE, save=True)


        normals = result.normal_map

        print(normals.data.shape)      # vectores unitarios float32 (H, W, 3)

        normals.assert_normalized()    # falla si algún píxel no tiene longitud
        unitaria
    - label: Leer un píxel
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreMoGe2s-normal.pt")

        result = model(SAMPLE_IMAGE)


        # Sistema de cámara de OpenCV: +x a la derecha, +y hacia abajo, +z hacia

        # la escena. Una superficie que mira a la cámara da algo cercano a (0,
        0, -1).

        field = result.normals.data

        h, w = field.shape[:2]

        print(field[h // 2, w // 2])
    - label: Guardar la visualización
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreMoGe2s-normal.pt")

        result = model(SAMPLE_IMAGE)


        # plot() dibuja el campo; está definido para resultados de normales y de
        bordes.

        result.plot().save("normals.png")
  val:
    - label: Validar y leer las claves de las métricas
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMoGe2s-normal.pt")
        metrics = model.val(data="my-dataset.yaml", imgsz=518)

        print(metrics["metrics/mean_angular_error"])     # grados
        print(metrics["metrics/median_angular_error"])   # grados
        print(metrics["metrics/within_11_25"])           # porcentaje de píxeles
        print(metrics["metrics/within_22_5"], metrics["metrics/within_30"])
  export:
    - label: Exportar
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMoGe2s-normal.pt")
        model.export(format="onnx", imgsz=518)
    - label: Ejecutar el archivo exportado
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # La factory decide según la extensión del archivo, así que un artefacto

        # exportado se carga como cualquier checkpoint y devuelve el mismo
        Results.

        model = LibreYOLO("LibreMoGe2s-normal.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.normal_map.data.shape)
source_hash: d26d26d894b436ff
---

## Definición

La tarea `normal` predice un vector unitario de tres componentes por píxel a
partir de una sola imagen RGB: la dirección hacia la que mira la superficie en
ese píxel. A diferencia de la profundidad, la salida no tiene una escala libre,
así que dos predicciones son directamente comparables sin alinearlas.

Una predicción rellena `result.normal_map`, un payload `NormalMap` que contiene
un array `(H, W, 3)` float32 sobre el lienzo de la imagen original, accesible
también como `result.normals`. Los vectores usan el sistema de cámara de OpenCV
de LibreYOLO, con `+x` a la derecha, `+y` hacia abajo y `+z` hacia la escena, y
miran hacia la cámara, así que una superficie fronto-paralela da `(0, 0, -1)`.
`.assert_normalized()` comprueba que todos los píxeles son finitos y tienen
longitud unitaria dentro de una tolerancia. `result.boxes` queda vacío, así que
`conf`, `iou` y `max_det` no tienen efecto, y `Results.plot()` cubre esta tarea.

## Modelos

Dos familias cubren `normal`.

[MoGe-2](/docs/models/moge-2) es la dedicada: un modelo de geometría monocular de
un solo forward, en tres tamaños de encoder. LibreYOLO no copia estos checkpoints
a su propia organización; cargar uno descarga el tamaño correspondiente desde los
repositorios oficiales en una revisión fijada y lo verifica contra un SHA-256
registrado.

[LibreMODUS](/docs/models/libremodus) produce normales como uno de los objetivos
de un modelo any-to-any, y puede tomar como entrada un mapa de profundidad en
lugar de una imagen RGB. Necesita el extra `modus` y tu propia cuenta autenticada
de Hugging Face, y no ofrece ni `val()` ni `export()`, así que no participa en las
secciones de validación y exportación de más abajo.

## Predicción

Los pesos de MoGe-2 se descargan en el primer uso y se cachean en local.

<code-tabs name="predict" />

`imgsz` debe ser divisible por el tamaño de parche del encoder ViT, algo que
LibreYOLO comprueba antes de que empiece la ejecución. Predecir sobre una lista
de imágenes ejecuta un forward por imagen; esta tarea no tiene una ruta rápida
por batch apilado. Consulta [predicción](/docs/predict) para fuentes, streaming y
manejo de resultados.

## Formato del dataset

La validación de normales empareja cada imagen con un PNG de 16 bits y tres
canales, con el mismo nombre base y la misma resolución, más una máscara de
validez opcional.

```text
dataset/
  data.yaml
  images/
    val/room.jpg
  normals/
    val/room.png
  masks/
    val/room.png
```

```yaml
path: dataset
train: images/train
val: images/val
normals_dir: normals
masks_dir: masks
nc: 1
names: {0: normal}
```

El PNG objetivo es exactamente `uint16` de tres canales, con los canales
almacenados como RGB. La decodificación es `n = png / 65535 * 2 - 1` seguida de
renormalizar cada vector, y los vectores decodificados usan el mismo sistema de
cámara de OpenCV que las predicciones. Un píxel de la máscara cuenta como válido
cuando es distinto de cero; sin archivo de máscara, todo vector decodificado
finito y distinto de cero es válido. Los píxeles objetivo inválidos y los de
padding se guardan internamente como `(0, 0, 0)` y nunca contribuyen a una
métrica. Consulta [formatos de dataset](/docs/reference/dataset-formats) para el
contrato completo.

## Entrenamiento

Ninguna de las dos familias de normales tiene implementación de entrenamiento:
`train()` lanza `NotImplementedError` en ambas. La página de MoGe-2 señala sus
checkpoints oficiales fijados para predecir, validar y exportar.

## Validación

`val()` mide el ángulo entre cada vector predicho y su vector de ground truth,
sobre los píxeles que el dataset marca como válidos.

<code-tabs name="val" />

`metrics/mean_angular_error` y `metrics/median_angular_error` son ese ángulo en
grados, y cuanto más bajo mejor. `metrics/within_11_25`, `metrics/within_22_5` y
`metrics/within_30` son el porcentaje de píxeles válidos cuyo error angular queda
por debajo de 11,25, 22,5 y 30 grados, así que cuanto más alto mejor. Fíjate en
la unidad: esos tres son porcentajes, no fracciones. `fitness` es
`metrics/within_11_25` dividido entre 100, lo que sitúa la selección del mejor
checkpoint en la misma escala `[0, 1]` que en todas las demás tareas.

## Exportación

Un modelo de normales exportado se vuelve a cargar con `LibreYOLO()` según la
extensión de su archivo, así que un `.onnx` se comporta como un checkpoint y
devuelve el mismo `Results`.

<code-tabs name="export" />

La exportación de normales usa un contrato de runtime de resolución fija y batch
1: se rechazan `dynamic` y cualquier `batch` distinto de 1, y `imgsz` debe ser
divisible por el tamaño de parche del encoder. La cobertura por formato está en
la [página de MoGe-2](/docs/models/moge-2) y en la
[matriz completa de exportación](/docs/reference/export-matrix).
[Exportación](/docs/export) lista los argumentos que acepta cada formato.
