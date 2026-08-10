---
title: Eliminación de fondo
seo_title: Eliminación de fondo en LibreYOLO
description: >-
  Recorta un sujeto y sepáralo de su fondo en LibreYOLO. Predice un matte alfa
  suave, escribe un PNG transparente y valida con MAE y S-measure.
lead: >-
  La eliminación de fondo separa un sujeto de todo lo que hay detrás de él.
  LibreYOLO la expone como la tarea matte, que devuelve un valor alfa suave por
  píxel en lugar de una máscara binaria de primer plano.
keywords:
  - quitar fondo imagen python
  - modelo alpha matting
  - segmentación dicotómica de imágenes
  - recorte png transparente
  - matte alfa suave
last_verified: 1.5.0
snippets:
  predict:
    - label: Predecir un matte
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreBiRefNetl-matte.pt")
        result = model(SAMPLE_IMAGE)

        matte = result.matte
        print(matte.array.shape, matte.array.dtype)   # (H, W) float32 en [0, 1]
    - label: Escribir un PNG transparente
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreBiRefNetl-matte.pt")
        result = model(SAMPLE_IMAGE)

        # save() compone la imagen original con el matte como canal alfa.
        result.save("subject.png")

        rgba = result.cutout()   # el mismo array (H, W, 4) uint8 en memoria
        print(rgba.shape)
    - label: Componer sobre un fondo nuevo
      language: python
      code: >
        import numpy as np

        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreBiRefNetl-matte.pt")

        result = model(SAMPLE_IMAGE)


        rgba = result.cutout()

        alpha = rgba[..., 3:4].astype(np.float32) / 255.0

        backdrop = np.full_like(rgba[..., :3], 255)          # blanco

        composited = (rgba[..., :3] * alpha + backdrop * (1 -
        alpha)).astype(np.uint8)

        print(composited.shape)
  val:
    - label: Validar y leer las claves de las métricas
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreBiRefNetl-matte.pt")

        # Un directorio que contenga images/ y un directorio de mattes sirve en
        # lugar de un YAML de dataset.
        metrics = model.val(data="my-matte-dataset/")

        print(metrics["metrics/MAE"])        # cuanto más bajo, mejor
        print(metrics["metrics/Smeasure"])   # fitness, cuanto más alto, mejor
  export:
    - label: Exportación
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreBiRefNetl-matte.pt")
        model.export(format="torchscript")
    - label: Ejecutar el archivo exportado
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # La factory decide según la extensión del archivo, así que un artefacto

        # exportado se carga como cualquier checkpoint y devuelve el mismo
        objeto Results.

        model = LibreYOLO("LibreBiRefNetl-matte.torchscript")

        result = model(SAMPLE_IMAGE)


        print(result.matte.array.shape)
source_hash: f7d88c74d9729268
---

## Definición

La tarea `matte` predice un valor alfa por píxel a partir de una sola imagen RGB:
`1` es primer plano completo y `0` es fondo completo. El valor es continuo en
lugar de binario, y ahí está el sentido de la tarea. Una máscara binaria queda a
un umbral de distancia, en 0.5, mientras que el matte suave además conserva la
cobertura parcial del pelo, el pelaje y los bordes con motion blur que una
máscara binaria descarta.

Una predicción rellena `result.matte`, un payload `Matte` que contiene un array
float32 `(H, W)` en `[0, 1]` sobre el lienzo de la imagen original, accesible
como NumPy a través de `.array`. `result.cutout()` compone la imagen original con
ese alfa en un array RGBA `(H, W, 4)` uint8, y `result.save(path)` escribe lo
mismo en un PNG de fondo transparente. `result.boxes` queda vacío, así que
`conf`, `iou` y `max_det` no tienen efecto.

## Modelos

Dos familias sirven `matte`, y comparten el mismo forward path.

[BiRefNet](/docs/models/birefnet) es la red de referencia bilateral en torno a la
que está construida la tarea, publicada aquí como un checkpoint del nivel Swin-L.

[FeyNobg](/docs/models/feynobg) es la variante profundizada de Feyn Inc.: la
arquitectura de BiRefNet con la tercera etapa Swin ampliada de 18 a 24 bloques y
reentrenada después. LibreYOLO reutiliza para ella el forward path, el
preprocesado y la salida de un solo logit de BiRefNet, así que la predicción, la
validación y el manejo de checkpoints se comportan de forma idéntica; los pesos y
la identidad de la familia son propios de FeyNobg.

Las dos llevan licencias de pesos distintas. Ambas están indicadas en las páginas
de los modelos, y la licencia del repositorio de Hugging Face del checkpoint
concreto es la autoritativa.

## Predicción

Los pesos se descargan de Hugging Face en el primer uso y se cachean localmente.

<code-tabs name="predict" />

Ambas familias funcionan a un lienzo nativo fijo de 1024x1024 y redimensionan el
matte de vuelta a la imagen original. No se admite otra resolución, porque las
tablas de posición relativa del backbone Swin están ligadas a ese tamaño, y un
desajuste las interpola mal en lugar de lanzar un error. `Results.save()` está
definido solo para resultados de matte y necesita la imagen original, que recarga
desde `Results.path` salvo que le pases una. Consulta
[predicción](/docs/predict) para fuentes, streaming y manejo de resultados.

## Formato del dataset

La validación de matte empareja cada imagen RGB con un matte alfa de ground truth
de un solo canal que comparte el mismo nombre base, donde 0 es fondo y 255 es
primer plano.

```text
my-matte-dataset/
  images/
    subject.jpg
  mattes/
    subject.png
```

Basta con pasar esa raíz como `data=`: el directorio de mattes se detecta
automáticamente entre `mattes/`, `matte/`, `gt/`, `masks/`, `mask/` y `alpha/`.
La alternativa es un YAML de dataset, con `path` más `val_images` y `val_mattes`
nombrando directorios relativos a él:

```yaml
path: my-matte-dataset
val_images: images
val_mattes: mattes
nc: 1
names: {0: matte}
```

`nc` y `names` son marcadores de posición del esquema; un modelo de matte
devuelve `Results.matte`, no detecciones. Los valores del matte se leen como alfa
en `[0, 1]` dividiendo por 255, y un matte cuya forma difiera del lienzo de
predicción se redimensiona bilinealmente para que coincida. Consulta
[formatos de dataset](/docs/reference/dataset-formats) para el contrato completo.

## Entrenamiento

Ninguna de las dos familias de matte tiene implementación de entrenamiento:
`train()` lanza `NotImplementedError` en ambas, y el soporte de matte cubre solo
predicción, validación y exportación. Cada página de modelo nombra el proyecto
upstream que publica el código de entrenamiento y el script de conversión que
trae un checkpoint de vuelta.

## Validación

`val()` ejecuta el propio `predict` del modelo, así que la validación usa el
preprocesado exacto de la familia, y ambas métricas se calculan sobre el lienzo
de la imagen original.

<code-tabs name="val" />

`metrics/MAE` es el error absoluto medio frente al alfa de ground truth, en
`[0, 1]`, y cuanto más bajo, mejor. `metrics/Smeasure` es la S-measure de Fan et
al. (ICCV 2017), una similitud estructural que valora acertar con la forma del
sujeto y sus huecos, algo que una media por píxel se pierde; cuanto más alto,
mejor. La S-measure es además el `fitness`, el número que lee la selección del
mejor checkpoint. Ninguna de las dos métricas depende de la resolución.

## Exportación

Un modelo de matte exportado se vuelve a cargar con `LibreYOLO()` según la
extensión de su archivo, así que el artefacto se comporta como un checkpoint y
devuelve el mismo `Results`.

<code-tabs name="export" />

TorchScript es la vía validada para esta tarea. La conversión a ONNX funciona
pero no ha superado el mismo listón de paridad, y el resto de formatos no están
disponibles. La cobertura por formato está en las páginas de
[BiRefNet](/docs/models/birefnet) y [FeyNobg](/docs/models/feynobg) y en la
[matriz de exportación completa](/docs/reference/export-matrix).
