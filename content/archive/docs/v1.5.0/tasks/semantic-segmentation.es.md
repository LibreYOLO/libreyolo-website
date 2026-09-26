---
title: Segmentación semántica
seo_title: Segmentación semántica en LibreYOLO
description: >-
  Etiqueta cada píxel con una clase en LibreYOLO: las familias que sirven la
  tarea, el formato de máscara densa y las llamadas de predicción,
  entrenamiento, validación y exportación.
lead: >-
  La segmentación semántica asigna una clase a cada píxel de una imagen y no
  distingue entre instancias de la misma clase. La clave de la tarea es
  semantic.
keywords:
  - segmentación semántica python
  - clasificación de píxeles
  - predicción densa
  - entrenar modelo de segmentación
  - mIoU
  - biblioteca de segmentación MIT
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # El sufijo -sem del nombre de archivo selecciona la tarea, así que no
        # hace falta el argumento task.
        model = LibreYOLO("LibreSegformerb0-sem.pt")
        result = model(SAMPLE_IMAGE, save=True)

        mask = result.semantic_mask
        print(mask.data.shape)   # (H, W) ids de clase sobre el lienzo original
        print(mask.classes)      # ids de clase presentes, ordenados, sin el 255
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreSegformerb0-sem.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Una clase cada vez
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        result = LibreYOLO("LibreSegformerb0-sem.pt")(SAMPLE_IMAGE)
        mask = result.semantic_mask

        for class_id in mask.classes:
            pixels = mask.class_mask(class_id)   # booleano (H, W)
            print(result.names[class_id], int(pixels.sum()))
    - label: 'Otra familia, la misma llamada'
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibrePIDNets-sem.pt")
        result = model(SAMPLE_IMAGE)

        print(result.semantic_mask.data.shape)
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSegformerb0-sem.pt")
        model.train(data="my-dataset.yaml", epochs=160, imgsz=512, batch=8)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreSegformerb0-sem.pt data=my-dataset.yaml \
          epochs=160 imgsz=512 batch=8
    - label: Con ADE20K
      language: bash
      code: |
        # ade20k.yaml lleva incrustado un script de descarga para el archivo de
        # ~1 GB, así que necesita permiso explícito salvo que los datos ya
        # estén en local.
        libreyolo train model=LibreSegformerb0-sem.pt data=ade20k.yaml \
          epochs=160 imgsz=512 batch=8 allow_download_scripts=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSegformerb0-sem.pt")

        # val() devuelve un dict corriente, no un objeto.
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreSegformerb0-sem.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSegformerb0-sem.pt")
        model.export(format="onnx", imgsz=512)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreSegformerb0-sem.pt format=onnx imgsz=512
    - label: Usar el archivo exportado
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # La factoría enruta según el sufijo del archivo, así que un artefacto

        # exportado se carga como un checkpoint y devuelve el mismo objeto
        Results.

        model = LibreYOLO("LibreSegformerb0-sem.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.semantic_mask.data.shape)
source_hash: 44b92d8ba6062f04
---

## Definición

La segmentación semántica etiqueta píxeles, no objetos. Cada píxel recibe un id
de clase, y dos coches que se tocan en la imagen se convierten en una única
región de la clase coche, sin frontera entre ellos. Contar instancias es
[segmentación de instancias](/docs/tasks/instance-segmentation); etiquetar cada
píxel y separar instancias a la vez es
[segmentación panóptica](/docs/tasks/panoptic-segmentation).

`semantic` es la clave canónica de la tarea, y el sufijo `-sem` en el nombre de
archivo de un checkpoint la selecciona, así que no hace falta `task=` al cargar
pesos publicados.

`predict()` rellena `result.semantic_mask`. `.data` es un mapa de clases entero
`(H, W)` sobre el lienzo de la imagen original, `.classes` lista los ids
presentes en orden, y `.class_mask(id)` devuelve la selección booleana `(H, W)`
de una clase. El valor `255` es la etiqueta de ignorar: nunca es una clase, se
excluye de la loss y de las métricas, y `.classes` la deja fuera.

## Modelos

Tres familias entrenan y predicen:
[SegFormer](/docs/models/segformer),
[LingBot-Vision](/docs/models/lingbot-vision) y
[DINOv2](/docs/models/dinov2). SegFormer y LingBot-Vision funcionan con el
paquete base y traen pesos publicados. DINOv2 necesita
`pip install "libreyolo[rfdetr]"` y no tiene ningún checkpoint alojado por
LibreYOLO: carga el backbone original y su cabeza densa arranca con
inicialización aleatoria, así que es un punto de partida para entrenar más que
un predictor listo para usar.

Otras cuatro predicen, validan y exportan, pero su `train()` lanza
`NotImplementedError`: [FCN](/docs/models/fcn),
[DeepLabv3](/docs/models/deeplabv3), [PIDNet](/docs/models/pidnet) y
[EoMT](/docs/models/eomt).

Los conjuntos de clases dependen del checkpoint, no de la familia. Los pesos
publicados vienen de datasets cuyos espacios de etiquetas tienen poco en común,
las 150 clases de ADE20K frente a las 19 de Cityscapes entre ellos, así que el
`names` de un checkpoint es lo que te dice qué puede etiquetar, y dos
checkpoints solo son comparables cuando se entrenaron con el mismo.

## Predicción

Los pesos se descargan de Hugging Face en el primer uso y se cachean en local.

<code-tabs name="predict" />

El mapa es un argmax por píxel, así que no hay paso de NMS y `iou` nunca tiene
efecto. `conf` y `max_det` se aceptan por paridad de API y no hacen nada en
SegFormer, PIDNet y los demás predictores densos; EoMT es la excepción, donde
`conf` filtra la selección de queries. Consulta [predicción](/docs/predict) para
fuentes, streaming y manejo de resultados.

## Formato del dataset

Cada imagen se empareja con una máscara densa de un solo canal en lugar de con
un archivo de etiquetas `.txt`, y se localiza cambiando `images` por el
directorio de máscaras en la ruta de la imagen.

```text
dataset/
  data.yaml
  images/
    train/000001.jpg
    val/000101.jpg
  masks/
    train/000001.png
    val/000101.png
```

Las máscaras son imágenes de un solo canal sin pérdida, normalmente PNG, y los
PNG en modo paleta se leen como índices de paleta. Cada valor de píxel es un id
de clase en `0..nc-1`, el valor `255` significa ignorar, y la resolución de la
máscara tiene que coincidir con la de la imagen emparejada.

El YAML admite dos claves además del contrato compartido:

```yaml
path: dataset
train: images/train
val: images/val
masks_dir: masks
nc: 19
names:
  0: road
  1: sidewalk
```

`masks_dir` es el nombre de directorio que sustituye a `images`, y por defecto
es `masks`. `label_mapping` es un remapeo opcional `{source_id: train_id}` que
se aplica a los valores de píxel de la máscara al cargarla, que es como un
dataset numerado de 1 a 150 pasa a ir de 0 a 149; cualquier valor de origen que
quede sin mapear se convierte en ignorar, y todo train id tiene que caer en
`0..nc-1`.

Omitir `masks_dir` cambia el loader a un modo alternativo: las máscaras se
rasterizan al cargar a partir de etiquetas poligonales resueltas por la
convención habitual de `images` a `labels`, y se añade una clase `background`
después de las clases de objeto, con lo que `nc` crece en uno.

El loader canónico es `libreyolo.data.SemanticDataset`.

## Entrenamiento

<code-tabs name="train" />

Aquí `imgsz` está restringido de un modo que no lo está en un detector. Cada
familia declara un divisor del que su entrada tiene que ser múltiplo, fijado por
su rejilla de parches o su stride de salida, y tanto el entrenamiento como la
validación lanzan un `ValueError` antes de arrancar cuando `imgsz` no divide
exacto. El divisor es 32 para SegFormer, 16 para LingBot-Vision y EoMT, 14 para
DINOv2, y 8 para FCN y PIDNet. Consulta [entrenamiento](/docs/train) para
datasets, aumento de datos, multi-GPU y loggers.

## Validación

`val()` devuelve un diccionario corriente de claves `metrics/`, calculadas sobre
el split que nombra `val` en el YAML del dataset.

<code-tabs name="val" />

`metrics/mIoU` es la media de la intersección sobre la unión: para cada clase,
el solape entre los píxeles predichos y los verdaderos dividido por su unión,
promediado sobre las clases. Es el número principal y el que se usa para elegir
la mejor época durante el entrenamiento. `metrics/pixel_accuracy` es la
proporción de píxeles a los que se dio la clase correcta, que una clase de fondo
grande puede inflar, así que mIoU es la cifra con la que comparar. Los píxeles
marcados con `255` no cuentan para ninguna de las dos. El diccionario lleva
además `fitness`, una copia del valor de mIoU.

## Exportación

<code-tabs name="export" />

Un artefacto exportado se vuelve a cargar con `LibreYOLO()` según su sufijo de
archivo, así que un archivo `.onnx` o `.engine` se comporta como un checkpoint y
devuelve el mismo `Results`. La cobertura de formatos varía según la familia; la
matriz de cada página de modelo se genera a partir del conjunto validado en vez
de escribirse a mano. Consulta [exportación y despliegue](/docs/export) para los
formatos, sus extras y sus restricciones.
