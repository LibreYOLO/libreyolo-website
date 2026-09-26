---
title: Restauración de imágenes
seo_title: Restauración y escalado de imágenes en LibreYOLO
description: >-
  Elimina ruido, corrige el desenfoque y escala imágenes en LibreYOLO. Predice
  una imagen RGB restaurada, entrena NAFNet con datos emparejados y lee las
  claves PSNR y SSIM.
lead: >-
  La restauración de imágenes toma una imagen degradada y devuelve una limpia.
  LibreYOLO la expone como la tarea restore, que cubre la eliminación de ruido,
  la corrección de desenfoque y la superresolución detrás de un único contrato
  de salida: entra una imagen RGB, sale una imagen RGB.
keywords:
  - restauración de imágenes python
  - quitar ruido de imagen python
  - superresolución de imágenes python
  - escalar imagen sin perder calidad
  - modelo para quitar desenfoque
  - validación PSNR SSIM
last_verified: 1.5.0
snippets:
  predict:
    - label: Escalar una imagen
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # El generador 4x compacto; tile limita el pico de memoria con un origen
        grande.

        model = LibreYOLO("LibreRealESRGANx4t-restore.pt")

        result = model(SAMPLE_IMAGE, tile=512, tile_pad=10)


        result.restored.save("upscaled.png")

        print(result.restored.array.shape)   # 4x la entrada en cada eje
    - label: Quitar el ruido de una imagen
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Entrenado con ruido real de imagen de SIDD; la salida mantiene el
        tamaño de entrada.

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")

        result = model(SAMPLE_IMAGE)


        result.restored.save("denoised.png")

        print(result.restore_scale)   # 1: este checkpoint no escala
  train:
    - label: Hacer fine-tuning de NAFNet con imágenes emparejadas
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")

        model.train(data="my-dataset.yaml", epochs=100, imgsz=256, batch=16,
        lr0=1e-3)
    - label: Registrar la procedencia en el checkpoint
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")

        # degradation y dataset se escriben en el checkpoint guardado como
        # procedencia; no intervienen en el entrenamiento.
        model.train(
            data="my-dataset.yaml",
            epochs=100,
            degradation="denoise",
            dataset="MyDataset",
        )
  val:
    - label: Validar y leer las claves de métricas
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")

        # val() devuelve un dict simple, no un objeto.
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/PSNR"])   # fitness
        print(metrics["metrics/SSIM"])
  export:
    - label: Exportar
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")

        # imgsz queda fijado en el grafo, así que pasa el tamaño que tu
        # despliegue le va a dar realmente al modelo.
        model.export(format="onnx", imgsz=256)
    - label: Ejecutar el archivo exportado
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # La factory decide según la extensión del archivo, así que un artefacto

        # exportado se carga como cualquier checkpoint y devuelve el mismo
        objeto Results.

        model = LibreYOLO("LibreNAFNetl-restore-sidd.onnx")

        result = model(SAMPLE_IMAGE)


        result.restored.save("denoised.png")
source_hash: 9dc81cadb3ebf18b
---

## Definición

La tarea `restore` transforma una imagen en otra imagen. La eliminación de
ruido, la corrección de desenfoque y la superresolución son aquí la misma tarea,
porque comparten un contrato: el modelo consume una imagen RGB y devuelve una
imagen RGB, y la degradación que aprendió a deshacer es una propiedad del
checkpoint, no de la API.

Una predicción rellena `result.restored`, un payload `RestoredImage` que
contiene un array RGB uint8 de `(H, W, 3)`. `.array` lo devuelve como NumPy y
`.save(path)` lo escribe en disco. `result.restore_scale` registra el factor de
escalado que lleva el lienzo de salida, que es `1` para un checkpoint que
conserva la resolución. `result.boxes` queda vacío, así que `conf`, `iou` y
`max_det` se aceptan por paridad de firma pero no tienen efecto, y `save=True`
escribe directamente la imagen restaurada en lugar de una foto anotada.

## Modelos

Tres familias sirven `restore`, repartidas según la degradación que deshacen.

[NAFNet](/docs/models/nafnet) es el modelo de eliminación de ruido, y la única
familia de restauración que LibreYOLO puede entrenar. Su arquitectura sustituye
las activaciones no lineales de un bloque UNet por una multiplicación elemento a
elemento, y el checkpoint publicado está entrenado con ruido real de imagen de
SIDD. La salida mantiene la resolución de entrada.

[Real-ESRGAN](/docs/models/real-esrgan) es el escalador práctico: tres
checkpoints entrenados frente a degradaciones sintéticas y no solo frente al
reescalado bicúbico, a 4x, 2x y un generador 4x más pequeño y rápido pensado
para menor latencia.

[SwinIR](/docs/models/swinir) escala 4x con un backbone Swin Transformer, en
tres tamaños que cubren el generador ligero oficial y dos generadores para
imágenes reales.

## Predicción

Los pesos se descargan de Hugging Face en el primer uso y se cachean en local.

<code-tabs name="predict" />

La restauración se ejecuta a la resolución propia de la imagen de origen y no
sobre un lienzo de red fijo, con padding solo hasta el factor de submuestreo de
la red, así que tanto el tiempo como la memoria escalan con el número de píxeles
de tu entrada. `tile` divide el forward pass en tiles solapados y funde las
costuras al recomponerlos, y `tile_pad` es el halo que se añade alrededor de
cada tile antes de recortarlo de nuevo; ambos son argumentos de palabra clave de
Python. Consulta [predicción](/docs/predict) para fuentes, streaming y manejo de
resultados.

## Formato del dataset

La restauración empareja cada imagen de entrada degradada con una imagen
objetivo limpia exactamente de la misma resolución, casadas por el nombre base
del archivo.

```text
dataset/
  data.yaml
  inputs/
    train/photo.jpg
    val/photo.jpg
  targets/
    train/photo.jpg
    val/photo.jpg
```

```yaml
path: dataset
train: inputs/train
val: inputs/val
input_dir: inputs
target_dir: targets
degradation: denoise
dataset: MyDataset
nc: 1
names: {0: image}
```

`nc` y `names` son marcadores de posición del esquema; un modelo de restauración
devuelve `Results.restored`, no detecciones. `degradation` y `dataset` son
etiquetas opcionales de procedencia. `target_stem_suffix` cubre los datasets que
nombran la imagen limpia de forma distinta a su pareja degradada. La validación
conserva la resolución nativa y solo aplica el padding necesario para apilar un
batch, así que las métricas se calculan sobre el lienzo original. Consulta
[formatos de dataset](/docs/reference/dataset-formats) para el contrato
completo.

## Entrenamiento

NAFNet es la única familia de restauración con una implementación de
entrenamiento. `Real-ESRGAN.train()` y `SwinIR.train()` lanzan ambos
`NotImplementedError`: esos checkpoints proceden de entrenamiento GAN sobre
pipelines de degradación sintética, y el entrenador de restauración emparejada
se ejecutaría sin reproducir esa receta.

<code-tabs name="train" />

El entrenador toma recortes acoplados de la pareja de entrada y objetivo, de
modo que ambos lados quedan alineados. Consulta
[entrenamiento](/docs/train) para datasets, multi-GPU y loggers, y la
[página de NAFNet](/docs/models/nafnet) para los valores por defecto de esta
familia y el pooling de inferencia que desactiva durante el entrenamiento.

## Validación

`val()` compara la salida restaurada con el objetivo limpio, en RGB, sobre el
lienzo original, sin recorte de bordes ni reescalado.

<code-tabs name="val" />

`metrics/PSNR` es la relación señal-ruido de pico en decibelios, y es también
`fitness`, el número que lee la selección del mejor checkpoint. `metrics/SSIM`
es la similitud estructural en `[0, 1]`, calculada con una ventana gaussiana de
11x11 con sigma 1.5 y promediada sobre los tres canales de color. En ambos
casos, cuanto más alto, mejor.

## Exportación

Un modelo de restauración exportado se vuelve a cargar con `LibreYOLO()` según
la extensión de su archivo, así que un archivo `.onnx` o `.engine` se comporta
como un checkpoint y devuelve el mismo `Results`, con `restored` llevando la
imagen de salida.

<code-tabs name="export" />

La exportación de restauración fija la resolución espacial en el grafo, así que
pasa el `imgsz` que tu despliegue le va a dar realmente al modelo. Para NAFNet
ese tamaño debe ser divisible por el factor de submuestreo de la red, y solo la
dimensión de batch sigue siendo dinámica con `dynamic=True`. Para Real-ESRGAN y
SwinIR, omitir `imgsz` recurre a un tamaño de parche interno pequeño en lugar de
a tu resolución de trabajo. La cobertura por formato está en cada página de
modelo y en la
[matriz completa de exportación](/docs/reference/export-matrix).
[Exportación](/docs/export) enumera los argumentos que acepta cada formato.
