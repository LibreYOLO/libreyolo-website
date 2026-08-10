---
title: Clasificación de imágenes
seo_title: "Clasificación de imágenes en LibreYOLO"
description: "Etiqueta una imagen entera en LibreYOLO: las familias que cubren la tarea, la estructura de dataset ImageFolder y las llamadas de predicción, entrenamiento, validación y exportación."
lead: "La clasificación de imágenes asigna una distribución de etiquetas a la imagen entera y no localiza nada dentro de ella. La clave de la tarea es classify."
keywords: [clasificación de imágenes python, entrenar clasificador de imágenes, dataset ImageFolder, accuracy top-1, clasificación zero-shot, librería clasificación de imágenes licencia MIT]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # El sufijo -cls del nombre de archivo selecciona la tarea, así que no
        # hace falta ningún argumento task.
        model = LibreYOLO("LibreResNet50-cls.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.names[result.probs.top1], float(result.probs.top1conf))
        print(result.probs.top5)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreResNet50-cls.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: La distribución completa
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        result = LibreYOLO("LibreResNet50-cls.pt")(SAMPLE_IMAGE)
        probs = result.probs

        # .data es el vector completo (C,); top5/top5conf son vistas ordenadas.
        print(probs.data.shape)
        for index, score in zip(probs.top5, probs.top5conf):
            print(result.names[index], float(score))
    - label: Zero-shot, sin entrenamiento
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # CLIP puntúa la imagen frente a prompts de texto, así que el conjunto de
        # etiquetas se fija en la llamada en vez de venir dentro del checkpoint.
        model = LibreYOLO("LibreCLIPb32-cls.pt")
        model.set_classes(["a person jumping", "an empty street", "a parked car"])
        result = model(SAMPLE_IMAGE)

        print(model.names[result.probs.top1], float(result.probs.top1conf))
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # imagenette160 es un nombre de dataset conocido y se descarga en el primer uso.
        # Para tus propios datos, pasa un directorio con un split train/.
        model = LibreYOLO("LibreResNet50-cls.pt")
        model.train(data="imagenette160", epochs=5)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreResNet50-cls.pt data=imagenette160 epochs=5
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreResNet50-cls.pt data=imagenette160 \
          epochs=50 device=0,1 batch=-1
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreResNet50-cls.pt")

        # val() devuelve un dict plano, no un objeto.
        metrics = model.val(data="imagenette160")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreResNet50-cls.pt data=imagenette160
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreResNet50-cls.pt")
        model.export(format="onnx")
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreResNet50-cls.pt format=onnx
    - label: Usar el archivo exportado
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La factory decide según la extensión del archivo, así que un artefacto
        # exportado se carga como un checkpoint y devuelve el mismo objeto Results.
        model = LibreYOLO("LibreResNet50-cls.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.probs.top1, result.probs.top1conf)
---

## Definición

La clasificación de imágenes produce una puntuación por clase para la imagen
entera y ninguna coordenada. Responde qué hay en la imagen, nunca dónde, que es
lo que la separa de la [detección de objetos](/docs/tasks/object-detection).

`classify` es la clave canónica de la tarea, y el sufijo `-cls` en el nombre de
archivo de un checkpoint la selecciona. Ese sufijo es obligatorio, no opcional,
en las familias de clasificación, así que `LibreResNet50.pt` no se lee como
clasificador y solo `LibreResNet50-cls.pt` sí.

`predict()` rellena `result.probs` y deja `boxes` vacío. `.data` es el vector
completo de puntuaciones, `.top1` el índice de la puntuación más alta y
`.top1conf` su valor, `.top5` los cinco índices más altos en orden descendente y
`.top5conf` sus puntuaciones. Los índices apuntan dentro de `result.names`.
Hacer slicing de un objeto `Results` nunca trunca `probs`, porque el vector
pertenece a la imagen y no a una fila.

## Modelos

Cinco familias entrenan y predicen: [ResNet](/docs/models/resnet),
[ConvNeXt](/docs/models/convnext), [MobileNetV4](/docs/models/mobilenetv4),
[EfficientNetV2](/docs/models/efficientnetv2) y
[DINOv2](/docs/models/dinov2). Las cuatro primeras funcionan con el paquete base
y traen pesos publicados. DINOv2 necesita `pip install "libreyolo[rfdetr]"` y no
tiene ningún checkpoint alojado por LibreYOLO: carga el backbone original con
una cabeza lineal inicializada al azar, así que es un punto de partida para
hacer fine-tuning más que un predictor listo para usar.

Cinco más predicen, validan y exportan, pero su `train()` lanza
`NotImplementedError`: [ViT](/docs/models/vit), [Swin](/docs/models/swin),
[VGG](/docs/models/vgg), [AlexNet](/docs/models/alexnet) y
[DeiT](/docs/models/deit).

[CLIP](/docs/models/clip) y [SigLIP2](/docs/models/siglip2) clasifican sin un
conjunto fijo de etiquetas. Puntúan la imagen frente a prompts de texto, así que
`set_classes()` define las clases en el momento de la llamada y no hay ningún
paso de entrenamiento para un nuevo conjunto de etiquetas. Ambas sirven también
la tarea `embed`.

## Predicción

Los pesos se descargan de Hugging Face en el primer uso y se cachean localmente.

<code-tabs name="predict" />

`conf`, `iou` y `max_det` no tienen ningún efecto aquí: no hay candidatos que
filtrar por umbral ni que suprimir, solo una distribución. Consulta la
[predicción](/docs/predict) para las fuentes, el streaming y el manejo de
resultados.

## Formato del dataset

La clasificación usa un árbol de directorios, no archivos de etiquetas ni un
YAML. `data` es la raíz del dataset.

```text
dataset/
  train/
    tench/000001.jpg
    parachute/000002.jpg
  val/
    tench/000101.jpg
    parachute/000102.jpg
```

`train/` es obligatorio para entrenar y define el mapeo de clase a índice por
nombre de carpeta ordenado, así que la primera carpeta por orden alfabético pasa
a ser la clase 0. `val/` es obligatorio para validar. Puede haber un split
`test/`, y los comandos por defecto de entrenamiento y validación no lo usan.
Cualquier split distinto de `train` tiene que contener los mismos nombres de
carpeta de clase que el conjunto de clases esperado, que es lo que hace que un
desajuste falle de forma ruidosa en lugar de contar como una predicción
incorrecta. Las extensiones de imagen aceptadas son `.jpg`, `.jpeg`, `.png`,
`.bmp`, `.webp`, `.tif` y `.tiff`.

`data` acepta tres cosas: una ruta a un directorio que contenga un split
`train/`, una URL a un `.zip`, o uno de los nombres de dataset conocidos,
`imagenette160` y `smoke10`, que se descargan y se cachean en el primer uso.

El loader canónico es `libreyolo.data.classify_dataset`.

## Entrenamiento

<code-tabs name="train" />

No hay ningún `nc` que declarar: el número de clases sale de los nombres de
carpeta dentro de `train/`, y la capa lineal final se reconstruye para
ajustarse a él mientras el backbone se transfiere sin cambios. Consulta el
[entrenamiento](/docs/train) para los datasets, el aumento de datos, el
multi-GPU y los loggers.

## Validación

`val()` devuelve un diccionario plano de claves `metrics/`, calculadas sobre el
split `val/` de la raíz del dataset.

<code-tabs name="val" />

`metrics/accuracy_top1` es la proporción de imágenes cuya clase mejor puntuada
es la verdadera, y es la cifra principal, la que usa el entrenamiento para
elegir la mejor época. `metrics/accuracy_top5` es la proporción cuya clase
verdadera aparece en alguna de las cinco clases mejor puntuadas, que dice menos
cuantas menos clases tiene el dataset. El diccionario también lleva `fitness`,
una copia del valor de top-1.

## Exportación

<code-tabs name="export" />

Un artefacto exportado se vuelve a cargar a través de `LibreYOLO()` por la
extensión del archivo, así que un archivo `.onnx` o `.engine` se comporta como
un checkpoint y devuelve el mismo `Results`. La cobertura de formatos varía
según la familia; la matriz de cada página de modelo se genera a partir del
conjunto validado en lugar de escribirse a mano. Consulta
[exportación y despliegue](/docs/export) para los formatos, sus extras y sus
restricciones.
