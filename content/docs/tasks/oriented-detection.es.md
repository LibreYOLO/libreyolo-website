---
title: Detección orientada
seo_title: "Detección orientada en LibreYOLO"
description: "Detecta objetos rotados en LibreYOLO: las familias que ofrecen cajas orientadas, la fila de etiquetas de cuatro esquinas y las llamadas de predicción, entrenamiento, validación y exportación."
lead: "La detección orientada de objetos localiza cada instancia con un rectángulo rotado en vez de uno alineado a los ejes, así que un objeto inclinado queda bien ceñido en lugar de encerrado en una caja llena de fondo. La clave de la tarea es obb."
keywords: [detección de cajas orientadas, detección de objetos rotados, OBB python, dataset DOTA, detección de objetos en imágenes aéreas, IoU rotado]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        # Necesita el extra rfdetr: pip install "libreyolo[rfdetr]"
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # El sufijo -obb del nombre de archivo selecciona la tarea, así que no
        # hace falta ningún argumento de tarea.
        model = LibreYOLO("LibreRFDETRs-obb.pt")
        result = model(SAMPLE_IMAGE, save=True)

        obb = result.obb
        print(obb.xywhr)   # (N, 5): centro x, centro y, ancho, alto, radianes
        print(obb.conf, obb.cls)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreRFDETRs-obb.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Esquinas en vez de ángulos
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        result = LibreYOLO("LibreRFDETRs-obb.pt")(SAMPLE_IMAGE)
        obb = result.obb

        print(obb.xyxyxyxy.shape)    # (N, 4, 2) puntos de esquina en píxeles
        print(obb.xyxyxyxyn.shape)   # los mismos, normalizados
        print(obb.xyxy.shape)        # (N, 4) caja alineada a los ejes que la contiene
    - label: Un checkpoint más pequeño
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreRFDETRn-obb.pt")
        result = model(SAMPLE_IMAGE)

        print(result.obb.xywhr.shape)
    - label: RT-DETRv2
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Pesos de DOTA v1.0, 15 clases aéreas a 1024 px. El grafo orientado
        # se reconoce a partir de los propios tensores del checkpoint, así que
        # no hace falta argumento de tarea.
        model = LibreYOLO("LibreRTDETRv2n-obb.pt")
        result = model("aerial.png", save=True)

        obb = result.obb
        print(obb.xywhr)
        print(result.names)   # plane, ship, harbor, helicopter y 11 más
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Continúa desde pesos orientados publicados. data debe apuntar a un
        # dataset cuyas filas de etiquetas lleven cuatro esquinas.
        model = LibreYOLO("LibreRFDETRs-obb.pt")
        model.train(data="my-obb-dataset.yaml", epochs=50, imgsz=512, batch=8, lr0=1e-4)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreRFDETRs-obb.pt data=my-obb-dataset.yaml \
          epochs=50 imgsz=512 batch=8 lr0=1e-4
    - label: Desde pesos de detección
      language: bash
      code: |
        # Los pesos de detección no predicen ningún ángulo, así que esto es una
        # transferencia explícita. Pedir task=obb es lo que la autoriza.
        libreyolo train model=LibreRFDETRs.pt data=my-obb-dataset.yaml \
          task=obb epochs=50 imgsz=512
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs-obb.pt")

        # val() devuelve un dict normal, no un objeto.
        metrics = model.val(data="my-obb-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"], metrics["metrics/mAP75"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreRFDETRs-obb.pt data=my-obb-dataset.yaml
    - label: RT-DETRv2
      language: bash
      code: |
        libreyolo val model=LibreRTDETRv2n-obb.pt data=my-obb-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs-obb.pt")
        model.export(format="onnx", imgsz=512)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreRFDETRs-obb.pt format=onnx imgsz=512
    - label: RT-DETRv2
      language: bash
      code: |
        # ONNX y TorchScript son los destinos validados aquí, a FP32,
        # batch 1 y con un lienzo fijo de 1024 por 1024.
        libreyolo export model=LibreRTDETRv2n-obb.pt format=onnx imgsz=1024
        libreyolo export model=LibreRTDETRv2n-obb.pt format=torchscript imgsz=1024
    - label: Usar el archivo exportado
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La factoría enruta según el sufijo del archivo, así que un artefacto
        # exportado se carga como un checkpoint y devuelve el mismo objeto Results.
        model = LibreYOLO("LibreRFDETRs-obb.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.obb.xywhr)
---

## Definición

La detección orientada añade un número a una detección: el ángulo. Cada
instancia recibe un rectángulo rotado, una clase y una puntuación. La ganancia
es el ajuste. Un barco a 45 grados, el tejado de una nave industrial, una fila
de camiones aparcados: una caja alineada a los ejes alrededor de cualquiera de
ellos es fondo en su mayor parte, y dos cajas vecinas se solapan incluso cuando
los objetos no lo hacen. Por eso la tarea es estándar en imágenes aéreas y en
análisis de layout de documentos, y por eso el dataset de referencia es DOTA.

`obb` es la clave canónica de la tarea, y el sufijo `-obb` del nombre de archivo
de un checkpoint la selecciona, así que `task=` no hace falta al cargar pesos
publicados.

`predict()` rellena `result.obb`. `.xywhr` es la forma canónica `(N, 5)`:
centro x, centro y, ancho, alto y un ángulo en radianes que da la rotación del
lado del ancho alrededor del centro. `.conf` y `.cls` llevan la puntuación y el
índice de clase dentro de `result.names`, y `.id` un id de track cuando hay
seguimiento. `.xyxyxyxy` convierte cada fila en sus cuatro puntos de esquina
como `(N, 4, 2)` píxeles, `.xyxyxyxyn` normaliza esas esquinas y `.xyxy` da la
caja alineada a los ejes que la contiene, que es la que hay que usar cuando el
código posterior solo entiende rectángulos. `result.boxes` también se rellena,
con la forma alineada a los ejes.

## Modelos

Dos familias cubren esta tarea, y a cuál recurrir depende de si necesitas
entrenar.

[RF-DETR](/docs/models/rf-detr) es la que entrena. Predice, entrena, valida y
exporta cajas orientadas, y publica checkpoints orientados en cuatro tamaños,
n, s, m y l. Necesita su propio extra, `pip install "libreyolo[rfdetr]"`, y su
página de modelo lleva la licencia de los pesos y su procedencia.

Lee la sección de más abajo sobre qué predicen realmente esos checkpoints antes
de planificar nada con ellos.

[RT-DETRv2](/docs/models/rt-detr) es la que tiene pesos aéreos. Publica desde
`LibreRTDETRv2n-obb.pt` hasta `LibreRTDETRv2x-obb.pt`, los checkpoints oficiales
de DOTA v1.0 a una sola escala convertidos al formato de LibreYOLO, que cubren
las 15 clases de DOTA a 1024 px. No necesita ningún extra más allá del paquete
base, el grafo orientado se reconoce a partir de los propios tensores del
checkpoint, y la predicción, la validación y la exportación a ONNX y TorchScript
están todas soportadas. El entrenamiento no: la tarea orientada es solo de
inferencia en esa familia, `train()` lanza un error, y no hay transferencia
desde sus pesos de detección, que usan un backbone distinto. El seguimiento y el
aumento de datos en tiempo de test tampoco están disponibles para cajas
orientadas.

Así que: categorías de DOTA listas para usar, RT-DETRv2. Tus propias etiquetas
orientadas, RF-DETR.

## Predicción

Los pesos se descargan de Hugging Face en el primer uso y se cachean en local.

<code-tabs name="predict" />

Ten claro qué son los checkpoints publicados de RF-DETR antes de ejecutarlos. A
pesar de que DOTA es el benchmark de referencia de esta tarea, esos pesos no se
entrenaron con él. Los cuatro se inicializaron desde los pesos de detección de
RF-DETR y se les hizo fine-tuning sobre un único dataset de Roboflow Universe de
grabaciones de dron, con seis clases de vehículos: bike, bus, car,
other_vehicle, taxi y truck. Sus model cards los describen como pesos de
desarrollo, producidos mientras se validaba el soporte de entrenamiento
orientado, y dicen que no deben interpretarse como pesos de producción ni
oficiales de benchmark.

En la práctica eso significa que son un punto de partida utilizable para cajas
orientadas sobre vehículos vistos desde arriba, y para verificar que tu pipeline
funciona de principio a fin. Cualquier otro dominio implica entrenar con tus
propias etiquetas orientadas, y para las categorías aéreas por las que DOTA es
conocido, los checkpoints de RT-DETRv2 son los que están realmente entrenados
con esos datos. `conf` y `max_det` moldean la salida igual que en detección.
Consulta la [predicción](/docs/predict) para las fuentes, el streaming y el
manejo de resultados.

## Formato del dataset

La estructura es la de detección: un archivo de etiquetas `.txt` por imagen, que
se encuentra cambiando `images` por `labels` en la ruta de la imagen y
cambiando la extensión.

```text
dataset/
  data.yaml
  images/
    train/P0001.png
    val/P0101.png
  labels/
    train/P0001.txt
    val/P0101.txt
```

Una fila son exactamente nueve campos, un índice de clase seguido de cuatro
puntos de esquina en orden:

```text
<class_id> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4>
```

Los cuatro puntos son floats normalizados en `[0, 1]` y tienen que formar un
rectángulo orientado no degenerado. En el archivo de etiquetas no se guarda
ningún ángulo: el loader deriva el `xywhr` canónico a partir de las esquinas. El
parser es estricto por defecto y rechaza coordenadas fuera de rango, mientras
que la ingesta del dataset y de la validación puede recortar primero a `[0, 1]`
las etiquetas por lo demás válidas que caen en el borde de un crop, y aun así
rechazar después las cajas degeneradas.

El parseo de filas depende de la tarea. Nueve campos significan una caja
orientada solo en modo `obb`; en modo `segment` esa misma fila se lee como un
polígono de cuatro puntos.

El YAML es el YAML de detección:

```yaml
path: dataset
train: images/train
val: images/val
names:
  0: plane
  1: ship
```

El COCO JSON nativo también se carga, con un mapeo `annotations` de nombre de
split a archivo JSON. Las anotaciones se leen por orden de prioridad: un campo
`obb` con ocho esquinas en espacio de píxeles, un campo `obb` con
`[cx, cy, w, h, angle]` y el ángulo en radianes, un polígono o RLE de
`segmentation` reajustado a su rectángulo de área mínima, o un `bbox` de COCO
normal, que se trata como un rectángulo alineado a los ejes y se canoniza a
`xywhr`.

El parser canónico de filas es `libreyolo.data.parse_yolo_obb_label_line`.

## Entrenamiento

<code-tabs name="train" />

Entrenar en esta tarea significa RF-DETR. El entrenamiento continúa por defecto
desde un checkpoint `-obb` publicado. Partir de pesos de detección es una
transferencia deliberada: esos pesos no predicen ningún ángulo, y pasar
`task=obb` es lo que autoriza el cambio. Mantén `lr0` en `1e-4` o por debajo,
igual que en las demás tareas de la familia. Los checkpoints orientados de
RT-DETRv2 no admiten fine-tuning; úsalos tal cual, o entrena un modelo RF-DETR
con tus propias etiquetas. Consulta el [entrenamiento](/docs/train) para los
datasets, el aumento de datos, el multi-GPU y los loggers.

## Validación

`val()` devuelve un diccionario normal de claves `metrics/`. El emparejamiento
usa IoU rotado, calculado entre rectángulos orientados y no entre las cajas
alineadas a los ejes que los contienen, así que una predicción con la posición
correcta y el ángulo equivocado cuenta como fallo.

<code-tabs name="val" />

`metrics/mAP50-95` es la precisión media promediada sobre umbrales de IoU de
0.50 a 0.95 en pasos de 0.05, y es la cifra principal. A diferencia del camino
COCO que usa la detección, esta tarea respeta `iou_thresholds` en la
configuración de validación, así que el barrido se puede cambiar.
`metrics/mAP50` y `metrics/mAP75` son las versiones de un solo umbral.
`metrics/precision` y `metrics/recall` son precisión y recall reales con IoU
0.50, leídos en el punto de operación más laxo: se cuenta cada predicción que
sobrevive al umbral de confianza, y ese umbral vale 0.001 por defecto durante la
validación. Subir `conf` por tanto los mueve, mientras que las cifras de mAP,
que usan la curva de precisión-recall entera, se quedan donde están. Cuatro de
estas se repiten con un sufijo `(OBB)`, `metrics/mAP50-95(OBB)`,
`metrics/mAP50(OBB)`, `metrics/precision(OBB)` y `metrics/recall(OBB)`, que es
como quien llama distingue un resultado orientado de uno alineado a los ejes
cuando ambos están en la misma tabla. `metrics/mAP75` no tiene gemela con
sufijo.

Dos opciones no hacen nada en esta tarea. `save_json` y `save_plots` se aceptan
y registran un aviso: los volcados de predicciones orientadas y las gráficas de
validación no están implementados.

## Exportación

<code-tabs name="export" />

Un artefacto exportado se vuelve a cargar con `LibreYOLO()` según el sufijo de
su archivo, así que un archivo `.onnx` o `.engine` se comporta como un
checkpoint y devuelve el mismo `Results`. La cobertura de formatos varía por
tarea dentro de una misma familia, y la matriz de la página del modelo se genera
a partir del conjunto validado y da el motivo por el que un destino no está
disponible. Consulta la [exportación y despliegue](/docs/export) para los
formatos, sus extras y sus restricciones.
