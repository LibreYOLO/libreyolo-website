---
title: RT-DETR
families: [rtdetr]
seo_title: "RT-DETR, RT-DETRv2 y RT-DETRv4 en LibreYOLO"
description: "Usa RT-DETR, RT-DETRv2 y RT-DETRv4 en LibreYOLO para detección de objetos, más cajas orientadas en RT-DETRv2. Instala, predice, entrena, valida y exporta, con pesos Apache-2.0."
lead: "Un transformer de detección hecho para inferencia en tiempo real: decodifica un conjunto fijo de queries en vez de una rejilla densa, así que no ejecuta NMS. LibreYOLO incluye tres versiones, distinguidas por el checkpoint que cargas, y la versión 2 sirve además cajas orientadas."
keywords: [RT-DETR, RT-DETRv2, RT-DETRv4, "transformer de detección en tiempo real", DETR, "detección de objetos", "detección de cajas orientadas", OBB, DOTA]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreRTDETRr18.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreRTDETRr18.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Vídeo
      language: python
      code: |
        from libreyolo import LibreYOLO

        # La versión forma parte del nombre del archivo, y la factoría enruta
        # según el checkpoint, así que las tres se cargan igual.
        model = LibreYOLO("LibreRTDETRv4s.pt")

        # Cualquier fuente que acepta la biblioteca: archivo, carpeta, URL,
        # índice de webcam, stream RTSP o una lista .streams
        for result in model.predict("clip.mp4", stream=True, save=True):
            print(len(result.boxes))
    - label: Cajas orientadas
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Solo en la versión 2. El sufijo -obb selecciona la tarea, y el
        # checkpoint se reconoce como orientado a partir de sus propios
        # tensores, así que no hace falta el argumento task. Estos pesos son
        # DOTA v1.0, 15 clases aéreas a 1024 px.
        model = LibreYOLO("LibreRTDETRv2n-obb.pt")
        result = model("aerial.png", save=True)

        obb = result.obb
        print(obb.xywhr)     # (N, 5): cx, cy, w, h, radianes
        print(obb.xyxyxyxy)  # las mismas filas como cuatro esquinas
        print(result.boxes.xyxy)  # cajas alineadas con los ejes que las encierran
    - label: Cajas orientadas, CLI
      language: bash
      code: |
        libreyolo predict model=LibreRTDETRv2n-obb.pt source=aerial.png save=True
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTDETRr18.pt")

        # coco128.yaml descarga una muestra de 128 imágenes en el primer uso.
        # Apunta `data` al YAML de tu propio dataset para una ejecución real.
        model.train(data="coco128.yaml", epochs=50, batch=4, lr0=1e-4)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreRTDETRr18.pt data=coco128.yaml \
          epochs=50 batch=4 lr0=1e-4
    - label: LoRA
      language: python
      code: |
        # Necesita el extra lora: pip install "libreyolo[lora]"
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTDETRr18.pt")
        model.train(data="coco128.yaml", epochs=50, lora=True)
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreRTDETRr18.pt data=coco128.yaml \
          epochs=50 device=0,1
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTDETRr18.pt")

        # val() devuelve un dict plano, no un objeto
        metrics = model.val(data="coco128.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreRTDETRr18.pt data=coco128.yaml
    - label: Contra COCO
      language: bash
      code: |
        # coco-val-only.yaml descarga las 5000 imágenes de val2017 y se salta
        # el conjunto de entrenamiento. Lleva un script de descarga embebido,
        # así que necesita permiso explícito salvo que el dataset ya esté en
        # local.
        libreyolo val model=LibreRTDETRr18.pt data=coco-val-only.yaml \
          allow_download_scripts=True
    - label: Cajas orientadas
      language: python
      code: |
        from libreyolo import LibreYOLO

        # La validación orientada empareja con IoU rotado, así que una
        # predicción en el sitio correcto con el ángulo equivocado cuenta como
        # un fallo.
        model = LibreYOLO("LibreRTDETRv2n-obb.pt")
        metrics = model.val(data="my-obb-dataset.yaml")

        print(metrics["metrics/mAP50-95(OBB)"])
        print(metrics["metrics/mAP50(OBB)"])
  export:
    - label: Python
      language: python
      code: |
        # Necesita el extra onnx: pip install "libreyolo[onnx]"
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTDETRr18.pt")
        path = model.export(format="onnx")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreRTDETRr18.pt format=onnx
    - label: Cajas orientadas
      language: bash
      code: |
        # ONNX y TorchScript son los destinos validados para la tarea
        # orientada, a FP32, batch 1 y con un lienzo fijo de 1024 por 1024.
        libreyolo export model=LibreRTDETRv2n-obb.pt format=onnx imgsz=1024
        libreyolo export model=LibreRTDETRv2n-obb.pt format=torchscript imgsz=1024
    - label: Usar el archivo exportado
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La factoría enruta según la extensión del archivo, así que un
        # artefacto exportado se carga como cualquier checkpoint y devuelve el
        # mismo objeto Results.
        model = LibreYOLO("LibreRTDETRr18.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
---

## Instalación

RT-DETR no necesita ningún extra opcional. Todo lo que importa está en la
instalación base, y el extra `rtdetr` es un nombre estable que no le añade nada.

```bash
pip install libreyolo
```

El fine-tuning con adaptadores mediante `lora=True` es la excepción, y necesita
el extra `lora`.

```bash
pip install "libreyolo[lora]"
```

## Predicción

Los pesos se descargan de Hugging Face en el primer uso y se guardan en la caché
local.

<code-tabs name="predict" />

El objeto `Results` devuelto es el mismo que devuelven todas las familias, así
que cambiar a otro detector es un cambio de una línea. `conf` y `max_det`
filtran un decodificado top-k sobre queries y clases; no hay un paso de NMS que
ajustar, e `iou` se acepta pero no se usa. Un checkpoint orientado rellena
`result.obb` de forma nativa y rellena también `result.boxes` con los
rectángulos alineados con los ejes que los encierran. Consulta
[predicción](/docs/predict) para fuentes, streaming y manejo de resultados.

## Variantes

Tres versiones, dos tareas entre ellas, y los códigos de tamaño no siguen una
única serie. La versión 1 nombra sus tamaños según el backbone, ResNet o
HGNetv2. La versión 2 reutiliza solo los nombres de ResNet: la versión 1 ya
publica los dos tamaños HGNetv2, y los resultados de la versión 2 ahí quedaron
lo bastante cerca como para que LibreYOLO no publique pesos duplicados para
ellos. La versión 4 usa una serie de letras a secas, que choca con los nombres
HGNetv2 de la versión 1, así que un código de tamaño por sí solo no identifica
un modelo. La versión va escrita en el nombre del archivo del checkpoint.

<benchmark-table task="detect" />

<va-embed />

La versión 2 mantiene la arquitectura y la disposición del state dict de la
versión 1 y cambia cómo muestrea la atención deformable, y por eso las dos se
distinguen por los metadatos del checkpoint y no por la forma. La versión 4 es
un linaje distinto: reutiliza la arquitectura y el trainer de D-FINE, y sus
pesos vienen de destilar un modelo fundacional de visión DINOv3 como profesor en
un estudiante HGNetv2. En LibreYOLO `LibreRTDETRv4` es una subclase de
`LibreDFINE` con la cabeza de máscaras desactivada de forma fija, así que se
queda solo en detección.

### Cajas orientadas en la versión 2

La versión 2 es la única versión que lleva una segunda tarea. Sus tareas
soportadas son `detect` y `obb`, y las dos no comparten ni grafo ni serie de
tamaños. La detección usa los tamaños ResNet a 640 px; la detección orientada
usa una serie HGNetv2, n, s, m, l y x, a 1024 px, y el tamaño de entrada se
resuelve por tarea y no por familia. Un checkpoint se reconoce como orientado a
partir de sus propios tensores, por las cabezas de caja de cinco coordenadas y
los parámetros de muestreo de la versión 2, así que los pesos `-obb` se cargan
en el grafo orientado sin argumento `task`, y un desajuste entre ambos es un
error duro en lugar de una reinterpretación silenciosa.

Los archivos publicados van de `LibreRTDETRv2n-obb.pt` a
`LibreRTDETRv2x-obb.pt`. Son los checkpoints oficiales de DOTA v1.0 a escala
única convertidos al formato de LibreYOLO, 15 clases aéreas que van de avión y
barco a puerto y helicóptero, y sus nombres de clase están grabados en el
checkpoint. A diferencia del lado de detección, la tarea orientada es solo de
inferencia: la predicción, la validación y la exportación funcionan, y `train()`
sobre un modelo orientado lanza un error. El tracking y el aumento de datos en
test (data augmentation) tampoco soportan cajas orientadas.
[Detección orientada](/docs/tasks/oriented-detection) cubre la tarea, el formato
de etiquetas y las métricas.

## Entrenamiento

El entrenamiento parte de un checkpoint publicado. `pretrained` se acepta y
luego se descarta en las tres versiones, así que `pretrained=False` no te da un
modelo inicializado al azar. Todo lo de esta sección va de detección: la tarea
orientada de la versión 2 es solo de inferencia, y no hay ninguna vía de
transferencia desde pesos de detección hacia ella, porque las dos usan backbones
distintos.

<code-tabs name="train" />

El learning rate es el argumento que hay que acertar, y cada versión lleva el
suyo por defecto en lugar del de toda la biblioteca. La firma de `train()` en
Python lo lee de la configuración de entrenamiento de esa versión, y el CLI
resuelve el mismo valor cuando no se pasa `lr0`. Las versiones 1 y 2 aceptan
además `lr_backbone` y lo fijan por defecto a una veinteava parte de `lr0`,
siguiendo la receta original; la versión 4 corre a través del trainer de D-FINE,
que en su lugar escala el grupo de parámetros del backbone con
`backbone_lr_mult`.

Deja `imgsz` en el tamaño nativo del checkpoint salvo que tengas un motivo para
cambiarlo. La validación y la predicción a otros tamaños funcionan, con un
residuo: un tamaño rectangular cuyo número de tokens coincide con el del tamaño
nativo sigue reutilizando un embedding construido para una relación de aspecto
equivocada.

Consulta [entrenamiento](/docs/train) para datasets, aumento de datos
(data augmentation), multi-GPU y loggers.

## Validación

`val()` devuelve un diccionario de claves `metrics/` que cubren precisión,
recall, mAP 50 y mAP 50-95, medidas contra cualquier dataset en el formato con
el que entrenaste.

<code-tabs name="val" />

Las filas de la tabla de benchmarks de arriba salen del arnés de benchmarks de
LibreYOLO; la nota bajo esa tabla indica qué dataset las produjo y enlaza los
registros de las ejecuciones.

La validación orientada corre a través de la misma llamada y reporta las mismas
claves, más cuatro repetidas bajo un sufijo `(OBB)`. El emparejamiento usa IoU
rotado en lugar del IoU de los rectángulos que las encierran, así que un error
de ángulo es un fallo. `augment=True` se rechaza en esta tarea.

## Exportación

<export-matrix />

La matriz cubre todo el linaje en una sola página: donde las tres versiones no
coinciden sobre un formato, la celda muestra la más débil de las tres, así que
aquí nada queda sobrevendido para la versión que cargues. La fila de las
orientadas pertenece solo a la versión 2. ONNX y TorchScript están validados
ahí, a FP32, batch 1 y con un lienzo fijo de 1024 por 1024; OpenVINO, TensorRT y
ExecuTorch convierten y recargan, pero no han alcanzado la paridad de salida
cruda en todo el conjunto de queries, así que las cajas de arriba coinciden
hasta una fracción de píxel mientras que la cola se desvía.

Un artefacto exportado se vuelve a cargar con `LibreYOLO()` según su sufijo de
archivo, así que un archivo `.onnx` o `.engine` se comporta como un checkpoint y
devuelve el mismo `Results`.

<code-tabs name="export" />

## Checkpoints

Todos los archivos de pesos publicados de esta familia.

<checkpoint-table />

El nombre del archivo lleva la versión, luego el tamaño, luego la tarea. Los
pesos de detección son `LibreRTDETR<size>.pt`, `LibreRTDETRv2<size>.pt` y
`LibreRTDETRv4<size>.pt`, todos a 640 px. Los pesos orientados existen solo para
la versión 2 y añaden el sufijo de la tarea, de `LibreRTDETRv2n-obb.pt` a
`LibreRTDETRv2x-obb.pt`, todos a 1024 px y entrenados con DOTA v1.0 en lugar de
COCO.

## Licencia

<provenance-box></provenance-box>

## Cita

<citation-block />

El bloque de arriba es lo que publican los autores para la detección de las
versiones 1 y 2. Los pesos orientados de la versión 2 tienen un tercer upstream,
el repositorio RiO-DETR bajo Apache-2.0 en
[github.com/RicePasteM/RiO-DETR](https://github.com/RicePasteM/RiO-DETR), que es
de donde vienen los checkpoints de DOTA; cita ese proyecto si has usado uno. La
versión 4 es
un artículo aparte de otro grupo y tiene su propio bloque de cita en
[github.com/RT-DETRs/RT-DETRv4](https://github.com/RT-DETRs/RT-DETRv4#4-citation);
cita ese si has usado un checkpoint de la versión 4.
