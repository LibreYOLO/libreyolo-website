---
title: Segmentación de instancias
seo_title: "Segmentación de instancias en LibreYOLO"
description: "Segmenta objetos individuales en LibreYOLO: las familias que cubren la tarea, el formato de etiquetas de polígonos y las llamadas de predicción, entrenamiento, validación y exportación."
lead: "La segmentación de instancias localiza cada instancia de objeto y devuelve una máscara por píxel para cada una, junto al box, la clase y la puntuación que devuelve un detector. La clave de la tarea es segment."
keywords: [segmentación de instancias python, máscaras de objetos python, entrenar modelo de segmentación, etiquetas de polígonos yolo, segmentación de instancias licencia MIT, mAP de máscaras]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # El sufijo -seg del nombre de archivo selecciona la cabeza de máscaras,
        # así que no hace falta ningún argumento de tarea.
        model = LibreYOLO("LibreDFINEn-seg.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.masks.data.shape)   # (N, H, W), una máscara por detección
        print(result.boxes.xyxy.shape)   # (N, 4), las mismas N filas
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreDFINEn-seg.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Contornos de las máscaras
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDFINEn-seg.pt")
        result = model(SAMPLE_IMAGE)

        # .xy es una lista de contornos (P, 2) en píxeles, .xyn los mismos normalizados.
        for name, contour in zip(result.boxes.cls, result.masks.xy):
            print(result.names[int(name)], contour.shape)
    - label: Otra familia, la misma llamada
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreRTMDets-seg.pt")
        result = model(SAMPLE_IMAGE)

        print(result.masks.data.shape)
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Continúa desde pesos de segmentación publicados, cabeza de máscaras incluida.
        # data debe apuntar a un dataset cuyas etiquetas lleven polígonos.
        model = LibreYOLO("LibreDFINEn-seg.pt")
        model.train(data="my-dataset.yaml", epochs=50, imgsz=640, batch=8, lr0=2e-4)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreDFINEn-seg.pt data=my-dataset.yaml \
          epochs=50 imgsz=640 batch=8 lr0=2e-4
    - label: Desde pesos de detección
      language: bash
      code: |
        # Los pesos de detección no llevan cabeza de máscaras, así que esto es
        # una transferencia explícita: la cabeza empieza sin entrenar. Pedir
        # task=segment es lo que la autoriza.
        libreyolo train model=LibreDFINEn.pt data=my-dataset.yaml \
          task=segment epochs=50 imgsz=640
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDFINEn-seg.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])       # máscaras
        print(metrics["metrics/mAP50-95(M)"])    # máscaras, explícito
        print(metrics["metrics/mAP50-95(B)"])    # boxes
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDFINEn-seg.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDFINEn-seg.pt")
        model.export(format="onnx", imgsz=640)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreDFINEn-seg.pt format=onnx imgsz=640
    - label: Usar el archivo exportado
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La factory decide según la extensión del archivo, así que un artefacto
        # exportado se carga como un checkpoint y devuelve el mismo objeto Results.
        model = LibreYOLO("LibreDFINEn-seg.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.masks.data.shape)
---

## Definición

La segmentación de instancias es detección más forma. Cada instancia de objeto
sigue recibiendo un box, una clase y una puntuación, y además recibe una máscara
binaria que cubre los píxeles que le pertenecen. Las máscaras pueden solaparse,
y los píxeles que no pertenecen a ningún objeto quedan sin asignar, que es lo
que separa la tarea de la
[segmentación semántica](/docs/tasks/semantic-segmentation) y la
[segmentación panóptica](/docs/tasks/panoptic-segmentation).

`segment` es la clave canónica de la tarea, y el sufijo `-seg` en el nombre de
archivo de un checkpoint la selecciona, así que no hace falta `task=` al cargar
pesos publicados.

`predict()` rellena `result.masks` junto a `result.boxes`. `.data` es una pila
`(N, H, W)` sobre el lienzo de la imagen original, alineada fila a fila con los
boxes, de modo que la máscara `i` pertenece al box `i`. `.xy` convierte cada
máscara en su mayor contorno exterior como un array de píxeles `(P, 2)`, y
`.xyn` da el mismo contorno normalizado.

## Modelos

Cuatro familias entrenan y predicen máscaras: [RF-DETR](/docs/models/rf-detr),
[EdgeCrafter](/docs/models/edgecrafter), [D-FINE](/docs/models/d-fine) y
[RTMDet](/docs/models/rtmdet). RF-DETR necesita su propio extra,
`pip install "libreyolo[rfdetr]"`; las otras tres funcionan con el paquete base.

[Mask R-CNN](/docs/models/mask-rcnn) predice, valida y exporta máscaras, pero su
`train()` lanza `NotImplementedError`.

[EoMT](/docs/models/eomt) predice y valida máscaras y tampoco puede entrenar, y
su exportación es aún más estrecha: `export()` solo acepta la tarea semántica, y
lanza `NotImplementedError` para `segment` y `panoptic`, porque el contrato de
runtime de máscaras por query que esas dos necesitan no está definido. Usa EoMT
para máscaras de instancia en Python, no a través de un grafo exportado.

Un grupo aparte segmenta a partir de un prompt en lugar de una lista de clases:
un clic, un box o una frase eligen el objeto, y el modelo devuelve su máscara.
[SAM](/docs/models/sam), [SAM 2](/docs/models/sam-2),
[SAM 3](/docs/models/sam-3), [MobileSAM](/docs/models/mobilesam),
[EdgeTAM](/docs/models/edgetam) y [PicoSAM3](/docs/models/picosam3) funcionan
así, igual que [SenseNova-Vision](/docs/models/sensenova-vision), cuya
segmentación es referring: toma una frase que nombra un objeto. Se cargan a
través de su propia factory y sus extras, y cada página de modelo lleva la
llamada exacta.

## Predicción

Los pesos se descargan de Hugging Face en el primer uso y se cachean localmente.

<code-tabs name="predict" />

`conf` y `max_det` moldean la salida igual que en detección, y las máscaras se
filtran junto con los boxes a los que pertenecen. Consulta la
[predicción](/docs/predict) para las fuentes, el streaming y el manejo de
resultados.

## Formato del dataset

La disposición es la de detección: un archivo de etiquetas `.txt` por imagen,
que se localiza sustituyendo `images` por `labels` en la ruta de la imagen y
cambiando la extensión.

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

Lo que cambia es la fila. Un segmento es un índice de clase seguido de un
polígono plano:

```text
<class_id> <x1> <y1> ... <xN> <yN>
```

Al menos tres puntos, así que el número de coordenadas después del índice de
clase es par y de al menos seis, y el polígono no puede ser degenerado. Las
coordenadas son floats en `[0, 1]`, relativas al ancho y alto de la imagen
original. Una fila de detección de cinco campos también se acepta en un dataset
de segmentación y se lee como un segmento rectangular, lo que permite cargar un
dataset solo de boxes sin una pasada de conversión.

El YAML es el YAML de detección:

```yaml
path: dataset
train: images/train
val: images/val
names:
  0: person
  1: bicycle
```

El JSON nativo de COCO también funciona: añade un mapeo `annotations` de nombre
de split a archivo JSON, y la ruta del split pasa a indicar la raíz de las
imágenes.

## Entrenamiento

<code-tabs name="train" />

El entrenamiento continúa por defecto desde un checkpoint `-seg` publicado.
Partir de pesos de detección es posible, pero es una transferencia deliberada:
esos pesos no llevan cabeza de máscaras, así que empieza sin entrenar, y pasar
`task=segment` es lo que autoriza el cambio. Consulta el
[entrenamiento](/docs/train) para los datasets, el aumento de datos, el
multi-GPU y los loggers.

## Validación

`val()` devuelve un diccionario plano de claves `metrics/`. Los boxes y las
máscaras se puntúan por separado, ambos con la evaluación COCO, y las cifras de
máscara son las principales.

<code-tabs name="val" />

Las claves sin sufijo contienen los resultados de máscara:
`metrics/mAP50-95`, `metrics/mAP50`, `metrics/mAP75`, luego `metrics/mAP_small`,
`metrics/mAP_medium` y `metrics/mAP_large` por área del objeto, y
`metrics/AR1`, `metrics/AR10`, `metrics/AR100`, `metrics/AR_small`,
`metrics/AR_medium` y `metrics/AR_large` para el recall promedio.
`metrics/AR_max_det` y `metrics/max_det` registran el límite de detecciones que
usó la ejecución.

Cuatro cifras se publican también bajo un sufijo explícito, `(M)` de máscara y
`(B)` de box, para que una comparación nunca dependa de qué número decidió
llamar principal la familia: `metrics/mAP50-95(M)` y `metrics/mAP50-95(B)`,
`metrics/mAP50(M)` y `metrics/mAP50(B)`, `metrics/precision(M)` y
`metrics/precision(B)`, `metrics/recall(M)` y `metrics/recall(B)`. En esta tarea
no existe `metrics/precision` ni `metrics/recall` sin sufijo.

Lee con cuidado las claves de precisión y recall. Se mantienen por
retrocompatibilidad y son alias, no un punto de operación:
`metrics/precision(M)` contiene el mismo valor que `metrics/mAP50-95(M)`, y
`metrics/recall(M)` el mismo valor que el AR de máscara con 100 detecciones, y
`(B)` se comporta igual para los boxes. Graficar un par de ellas reporta el
mismo número dos veces.

## Exportación

<code-tabs name="export" />

Un artefacto exportado se vuelve a cargar a través de `LibreYOLO()` por la
extensión del archivo, así que un archivo `.onnx` o `.engine` se comporta como
un checkpoint y devuelve el mismo `Results`. La cobertura de segmentación es más
estrecha que la de detección en la misma familia. La matriz de cada página de
modelo se genera a partir del conjunto validado y nombra el motivo por el que un
destino no está disponible. Consulta
[exportación y despliegue](/docs/export) para los formatos, sus extras y sus
restricciones.
