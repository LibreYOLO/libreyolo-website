---
title: Estimación de pose
seo_title: "Estimación de pose en LibreYOLO"
description: "Predice keypoints por instancia en LibreYOLO: las familias que cubren la tarea, el formato de etiquetas y las llamadas de predicción, entrenamiento, validación y exportación."
lead: "La estimación de pose localiza cada instancia y devuelve para ella un conjunto ordenado de keypoints con nombre, de modo que la salida lleva la estructura interna del objeto y no solo su extensión. La clave de la tarea es pose."
keywords: [estimación de pose python, detección de keypoints, modelo de pose humana, keypoints COCO, OKS mAP, entrenar modelo de pose]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # El sufijo -pose del nombre de archivo selecciona la cabeza de
        # keypoints, así que no hace falta ningún argumento de tarea.
        model = LibreYOLO("LibreECs-pose.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.keypoints.xy.shape)   # (N, K, 2) coordenadas en píxeles
        print(result.boxes.xyxy.shape)     # (N, 4), las mismas N instancias
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreECs-pose.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Solo keypoints visibles
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        result = LibreYOLO("LibreECs-pose.pt")(SAMPLE_IMAGE)
        kpts = result.keypoints

        # .has_visible se deriva de la tercera columna de keypoints, y es
        # toda verdadera cuando el checkpoint solo predice (x, y).
        for person, visible in zip(kpts.xy, kpts.has_visible):
            print(person[visible])
    - label: Top-down en su lugar
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # HRNet es top-down: primero recorta cada persona. Sin una fuente de
        # personas, se empareja con un detector LibreYOLO9t y registra la elección.
        model = LibreYOLO("LibreHRNetw32-pose.pt")
        result = model(SAMPLE_IMAGE)

        print(result.keypoints.xy.shape)
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # coco8-pose.yaml lleva un script de descarga embebido, así que necesita
        # permiso explícito salvo que los datos ya estén en local.
        model = LibreYOLO("LibreECs-pose.pt")
        model.train(
            data="coco8-pose.yaml",
            epochs=50,
            imgsz=640,
            batch=4,
            allow_download_scripts=True,
        )
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreECs-pose.pt data=coco8-pose.yaml \
          epochs=50 imgsz=640 batch=4 allow_download_scripts=True
    - label: Tu propio dataset
      language: python
      code: |
        from libreyolo import LibreYOLO

        # data.yaml debe declarar kpt_shape, y las filas de etiquetas deben
        # llevar exactamente 5 + K * D campos.
        model = LibreYOLO("LibreECs-pose.pt")
        model.train(data="my-pose-dataset.yaml", epochs=50, imgsz=640, batch=8)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs-pose.pt")

        # val() devuelve un dict plano, no un objeto.
        metrics = model.val(data="coco8-pose.yaml", allow_download_scripts=True)

        print(metrics["metrics/keypoints_mAP50-95"])
        print(metrics["metrics/keypoints_mAP50"], metrics["metrics/keypoints_mAP75"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreECs-pose.pt data=coco8-pose.yaml \
          allow_download_scripts=True
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs-pose.pt")
        model.export(format="onnx", imgsz=640)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreECs-pose.pt format=onnx imgsz=640
    - label: Usar el archivo exportado
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La factoría enruta según la extensión del archivo, así que un artefacto
        # exportado se carga como un checkpoint y devuelve el mismo objeto Results.
        model = LibreYOLO("LibreECs-pose.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.keypoints.xy)
---

## Definición

La estimación de pose devuelve estructura, no solo extensión. Cada instancia
sigue recibiendo un box, una clase y una puntuación, y además recibe `K`
keypoints en un orden fijo, de modo que el índice 5 significa la misma parte del
cuerpo en cada instancia y en cada imagen. El conjunto de etiquetas define ese
orden; nada en la salida identifica un keypoint por su nombre.

`pose` es la clave canónica de la tarea, y el sufijo `-pose` del nombre de
archivo de un checkpoint la selecciona, así que no hace falta `task=` al cargar
pesos publicados.

`predict()` rellena `result.keypoints` junto a `result.boxes`. `.data`
es `(N, K, 2)` o `(N, K, 3)`, alineado fila a fila con los boxes, de modo que la
instancia `i` de uno es la instancia `i` del otro. `.xy` extrae las coordenadas
en píxeles y `.xyn` las normaliza por el tamaño de la imagen original. `.conf`
es la tercera columna cuando el checkpoint la predice y `None` cuando no, y
`.has_visible` es la máscara booleana derivada de ella, toda verdadera cuando no
hay tercera columna.

Dos arquitecturas llegan a esta salida. Un modelo de una etapa predice boxes y
keypoints en una sola pasada. Un modelo top-down ejecuta primero un detector,
recorta cada instancia y regresa los keypoints dentro del recorte, así que su
precisión depende del detector que tiene delante.

## Modelos

Tres familias entrenan y predicen:
[RF-DETR](/docs/models/rf-detr), [EdgeCrafter](/docs/models/edgecrafter) y
[YOLO-NAS](/docs/models/yolo-nas), todas de una etapa. RF-DETR necesita su
propio extra, `pip install "libreyolo[rfdetr]"`. RF-DETR y EdgeCrafter traen
checkpoints de pose publicados y ambas hacen fine-tuning sobre datasets de una
sola clase, solo de personas; la cabeza de keypoints de EdgeCrafter queda fijada
en la construcción y rechaza un dataset que declare un número distinto, mientras
que RF-DETR reinicializa la suya para adaptarse. YOLO-NAS descarga sus pesos del
CDN propio de Deci.AI bajo una licencia no comercial, y LibreYOLO no publica
ninguno de ellos; su cabeza de pose también se reconstruye para un número nuevo
de keypoints, y es la única de las tres cuyo número de clases no está fijado a
uno, así que es la familia para un esqueleto multiclase o no humano, como la
pose de animales.

[HRNet](/docs/models/hrnet) es la opción top-down. Predice, valida y exporta, y
su `train()` lanza `NotImplementedError`. Si no se le da una fuente de personas,
se empareja automáticamente con un detector LibreYOLO9t; `cropped=True` trata la
imagen entera como una sola instancia, `person_boxes=` acepta boxes que ya
tengas y `person_detector=` nombra un detector distinto.

[SenseNova-Vision](/docs/models/sensenova-vision) también emite keypoints. Es un
modelo generativo guiado por prompts, con su propia factoría, `LibreVLM`, y su
propio extra; si no se fija ningún vocabulario, `set_task("pose")` recurre a la
categoría de personas. Sus pesos son no comerciales, y la latencia por imagen es
mucho mayor que la de una cabeza de pose diseñada para ello, porque cada
predicción es un decodificado por difusión.

## Predicción

Los pesos se descargan de Hugging Face en el primer uso y se cachean localmente.

<code-tabs name="predict" />

El número y el orden de los keypoints son propiedades del checkpoint, no de la
biblioteca, así que un modelo entrenado sobre un esqueleto distinto devuelve un
`K` distinto y un significado distinto por índice. Lo que contiene la tercera
columna de keypoints también es una propiedad del checkpoint: EdgeCrafter
escribe ahí una constante en lugar de una puntuación por punto, y no tiene
cabeza de boxes en absoluto, así que cada uno de sus boxes de pose es la
extensión que encierra los propios keypoints de esa instancia. Consulta
[predicción](/docs/predict) para las fuentes, el streaming y el manejo de
resultados.

## Formato del dataset

La estructura es la de detección: un archivo de etiquetas `.txt` por imagen, que
se localiza cambiando `images` por `labels` en la ruta de la imagen y cambiando
la extensión.

```text
dataset/
  data.yaml
  images/
    train/000001.jpg
    val/000101.jpg
  labels/
    train/000001.txt
    val/000101.txt
```

Una fila es una fila de detección con los keypoints añadidos al final:

```text
<class_id> <cx> <cy> <w> <h> <k1x> <k1y> [<k1v>] ... <kKx> <kKy> [<kKv>]
```

El número de campos es exactamente `5 + K * D`, donde `D` es el segundo valor de
`kpt_shape`. Las coordenadas de los boxes y de los keypoints son floats
normalizados respecto al ancho y al alto de la imagen original. La visibilidad
`v`, presente solo cuando `D` es 3, es `0`, `1` o `2`.

El YAML añade dos claves al contrato común:

```yaml
path: dataset
train: images/train
val: images/val
kpt_shape: [17, 3]
flip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
names:
  0: person
```

`kpt_shape` es obligatorio y vale `[K, 2]` o `[K, 3]`. `flip_idx` es opcional y
es una permutación de `0..K-1` que da, para cada keypoint, el índice que toma
tras un volteo horizontal, que es como una muñeca izquierda sigue siendo una
muñeca izquierda. Si lo omites, el aumento de datos por volteo horizontal se
desactiva para los keypoints en lugar de aplicarse con el orden de índices
equivocado.

## Entrenamiento

<code-tabs name="train" />

El entrenamiento continúa desde un checkpoint `-pose` publicado, que ya lleva la
cabeza de keypoints; la tarea se lee del checkpoint que cargas, no de un flag
que se pase en el momento de entrenar, así que un checkpoint de detección no se
convierte en una ejecución de pose por pedirlo. En EdgeCrafter, el `kpt_shape`
de tu YAML tiene que coincidir exactamente con la cabeza, ya que esta queda
fijada en la construcción, mientras que RF-DETR y YOLO-NAS sí redimensionan la
cabeza para un número distinto. Consulta
[entrenamiento](/docs/train) para los datasets, el aumento de datos, multi-GPU y
los loggers.

## Validación

`val()` devuelve un diccionario plano de claves `metrics/`. La puntuación es la
evaluación de keypoints de COCO sobre Object Keypoint Similarity, que pondera el
error de distancia de cada keypoint por la escala de la instancia y por una
tolerancia propia de cada keypoint, de modo que cumple el papel que IoU cumple
para los boxes. Necesita `pycocotools`, que viene en la instalación base.

<code-tabs name="val" />

`metrics/keypoints_mAP50-95` es la cifra principal, la mean average precision
promediada sobre los umbrales de OKS de 0.50 a 0.95, y es la que usa el
entrenamiento para elegir la mejor época. `metrics/keypoints_mAP50` y
`metrics/keypoints_mAP75` son las versiones de un solo umbral, y
`metrics/keypoints_mAP_M` y `metrics/keypoints_mAP_L` reparten el promedio por
área de la instancia, mediana y grande; la evaluación de keypoints de COCO no
define ningún grupo de instancias pequeñas. Las cifras de average recall
correspondientes son `metrics/keypoints_AR50-95`, `metrics/keypoints_AR50`,
`metrics/keypoints_AR75`, `metrics/keypoints_AR_M` y
`metrics/keypoints_AR_L`. Todas las claves de esta tarea llevan el prefijo
`keypoints_`, así que las claves de `mAP` de boxes que devuelve un detector no
aparecen.

## Exportación

<code-tabs name="export" />

Un artefacto exportado se vuelve a cargar con `LibreYOLO()` según su extensión
de archivo, así que un archivo `.onnx` o `.engine` se comporta como un
checkpoint y devuelve el mismo `Results`. La cobertura de formatos varía según
la familia; la matriz de cada página de modelo se genera a partir del conjunto
validado en lugar de escribirse a mano. Consulta
[exportación y despliegue](/docs/export) para los formatos, sus extras y sus
restricciones.
