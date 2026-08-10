---
title: YOLO-NAS
families: [yolonas]
seo_title: "YOLO-NAS: predice, entrena y exporta en LibreYOLO"
description: "Usa YOLO-NAS en LibreYOLO para detección y pose. Los pesos de Deci.AI son propietarios y de uso no comercial, y LibreYOLO no publica ninguno."
lead: "Un detector convolucional cuyo backbone y cuello salieron de la búsqueda de arquitecturas de Deci.AI, construido con bloques RepVGG preparados para la cuantización. Sus pesos son de Deci.AI, con licencia solo para uso no comercial, y LibreYOLO no publica ninguno."
keywords: [YOLO-NAS, YOLONAS, Deci AI, SuperGradients, "detección de objetos", "estimación de pose", "detector cuantizable", AutoNAC]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Un nombre que no esté ya en disco se descarga de la CDN de Deci. La
        # descarga imprime antes los términos de licencia de Deci; quedarte con
        # el archivo implica aceptarlos.
        model = LibreYOLO("LibreYOLONASs.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLONASs.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Pose
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # El sufijo -pose selecciona la cabeza de pose y su propio juego de pesos.
        model = LibreYOLO("LibreYOLONASs-pose.pt")
        result = model(SAMPLE_IMAGE)

        print(result.keypoints.xy)
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLONASs.pt")
        model.train(data="my-dataset.yaml", epochs=100, imgsz=640, batch=16)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLONASs.pt data=my-dataset.yaml \
          epochs=100 imgsz=640 batch=16
    - label: Desde cero
      language: python
      code: |
        from libreyolo import LibreYOLONAS

        # No se toca ningún checkpoint de Deci: el modelo parte de pesos
        # aleatorios, así que lo que salga de la ejecución deriva solo de tus datos.
        model = LibreYOLONAS(None, size="s")
        model.train(data="my-dataset.yaml", imgsz=640, batch=16)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLONASs.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLONASs.pt data=my-dataset.yaml
    - label: Contra COCO
      language: bash
      code: |
        # El yaml de COCO incluido lleva un script de descarga embebido, así que
        # necesita permiso explícito salvo que el dataset ya esté en local.
        libreyolo val model=LibreYOLONASl.pt data=coco.yaml imgsz=640 \
          allow_download_scripts=True
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLONASs.pt")
        model.export(format="onnx", imgsz=640)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreYOLONASs.pt format=onnx imgsz=640
    - label: Usar el archivo exportado
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La factoría resuelve según el sufijo del archivo, así que un artefacto
        # exportado se carga como cualquier checkpoint y devuelve el mismo objeto Results.
        model = LibreYOLO("LibreYOLONASs.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
---

## Instalación

YOLO-NAS no necesita nada más allá del paquete base.

```bash
pip install libreyolo
```

## Predicción

Un nombre de checkpoint que no esté ya en disco se descarga de la CDN pública de
Deci, no de la organización de LibreYOLO, que no aloja ninguno de estos pesos.
Antes de que empiece la transferencia, la biblioteca imprime los términos de
licencia de Deci una vez por proceso, y antes de abrir el archivo descargado
comprueba su SHA-256 contra un valor fijado. Lo que esos términos permiten está
en [licencia](#licensing).

<code-tabs name="predict" />

El objeto `Results` devuelto es el mismo que devuelven todas las familias, así
que cambiar a otro detector es un cambio de una línea. `conf` fija el umbral de
confianza e `iou` el umbral de NMS. Consulta [predicción](/docs/predict) para
fuentes, streaming y manejo de resultados.

## Variantes

Detección y pose son la misma arquitectura bajo cabezas distintas, y aceptan los
mismos argumentos. Los tamaños de la tabla de abajo son los de detección; pose
se publica en esos y en un tamaño más pequeño. La cabeza de pose predice el
conjunto de keypoints de COCO.

<benchmark-table task="detect" />

<va-embed />

## Entrenamiento

<code-tabs name="train" />

`epochs`, `lr0` y `amp` se resuelven por tarea cuando los omites, así que una
ejecución de pose parte de valores por defecto distintos que una de detección.
El optimizador es AdamW por defecto. El número de clases sale del YAML del
dataset y la cabeza se reconstruye para él antes de la primera época; en la
cabeza de pose el número de keypoints se maneja igual, así que a un checkpoint
de pose de COCO se le puede hacer fine-tuning sobre un esqueleto de otro tamaño.

El fine-tuning parte de los pesos de Deci, que es lo que cubre la licencia de
Deci. Entrenar desde un modelo inicializado al azar no involucra ningún
checkpoint de Deci, y eso es el tercer snippet de arriba.

Consulta [entrenamiento](/docs/train) para datasets, aumento de datos
(data augmentation), multi-GPU y loggers.

## Validación

`val()` devuelve un diccionario de claves `metrics/` que cubren precisión,
recall, mAP 50 y mAP 50-95, medidos contra cualquier dataset en el formato con
el que entrenaste.

<code-tabs name="val" />

## Exportación

<export-matrix />

Un artefacto exportado se vuelve a cargar con `LibreYOLO()` según su sufijo de
archivo, así que un archivo `.onnx` o `.engine` se comporta como un checkpoint y
devuelve el mismo `Results`. Ejecutar el grafo en un runtime pelado, sin
LibreYOLO instalado, también está soportado, pero entonces el preprocesado y el
postprocesado corren de tu cuenta. Cada formato instala un extra distinto y
acepta unos cuantos argumentos propios. Ambas cosas están en la página de ese
formato.

Una exportación es otra copia de los mismos pesos en un contenedor distinto.
Exportar un checkpoint de Deci no cambia ni de dónde vienen los pesos ni la
licencia que los cubre.

<code-tabs name="export" />

## Checkpoints

No hay ninguno que listar. La licencia de Deci prohíbe la redistribución, así
que la organización de LibreYOLO no publica pesos de YOLO-NAS y la descarga se
resuelve en otro sitio: un nombre de la forma `LibreYOLONAS<size>.pt`, o
`LibreYOLONAS<size>-pose.pt` para pose, se corresponde con el objeto equivalente
en la CDN pública de Deci.

Solo se pueden descargar así los checkpoints cuyo SHA-256 tiene fijado la
biblioteca. Cualquier otra cosa falla en cerrado en lugar de abrir un pickle de
terceros sin verificar, y hay que descargarla a mano y pasarla como ruta. Un
archivo que ya esté en disco se carga desde su ruta, sin descarga y sin control
de checksum. Eso incluye un `.pth` de Deci con su nombre original, que el
cargador reconoce.

## Licencia

<provenance-box>

LibreYOLO ni aloja ni replica estos pesos: no existe nada de esta familia en la
organización de LibreYOLO en Hugging Face. En su lugar, cada descarga automática
va a la CDN pública de Deci, imprime los términos de Deci una vez por proceso
antes de empezar, y se comprueba contra un SHA-256 fijado antes de abrir el
archivo.

Entrenar desde un modelo inicializado al azar es la alternativa. La arquitectura
es Apache-2.0 upstream y MIT aquí, así que un modelo entrenado de esa forma con
tus propios datos no deriva de ningún checkpoint de Deci.

</provenance-box>

## Cita

YOLO-NAS se publicó sin paper. La entrada de abajo es la que piden sus autores,
y cubre SuperGradients, la biblioteca en la que se distribuyó.

<citation-block />
