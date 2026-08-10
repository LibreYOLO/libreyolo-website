---
title: EdgeCrafter
families: [ec]
seo_title: "EdgeCrafter: detección, pose y segmentación en LibreYOLO"
description: "Usa EdgeCrafter en LibreYOLO para detección, pose y segmentación de instancias. Instala, predice, valida y exporta, con código bajo licencia MIT."
lead: "Un vision transformer compacto para predicción densa en hardware edge, publicado upstream como tres modelos hermanos: ECDet, ECPose y ECSeg. LibreYOLO carga los tres como una sola familia, con la tarea determinada por el checkpoint."
keywords: [EdgeCrafter, ECDet, ECPose, ECSeg, "vision transformer compacto", "detección de objetos", "estimación de pose", "segmentación de instancias", "inferencia en dispositivos edge"]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreECs.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreECs.pt source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg save=True
    - label: Pose
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # El sufijo -pose del nombre de archivo selecciona la cabeza de
        # keypoints, así que aquí no hace falta argumento de tarea.
        model = LibreYOLO("LibreECs-pose.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.keypoints.xy)
        print(result.boxes.conf)
    - label: Segmentación de instancias
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreECs-seg.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.masks.data.shape)
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=50,
            imgsz=640,
            batch=8,
            lr0=5e-4,
        )
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreECs.pt data=my-dataset.yaml epochs=50 imgsz=640 batch=8 lr0=5e-4
    - label: Pose
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Necesita un dataset de keypoints de una sola clase cuyo data.yaml
        # declare kpt_shape, e imgsz al tamaño nativo del checkpoint.
        model = LibreYOLO("LibreECs-pose.pt")
        model.train(
            data="my-pose-dataset.yaml",
            epochs=50,
            imgsz=640,
        )
    - label: Segmentación de instancias
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Necesita etiquetas de polígonos, e imgsz al tamaño nativo del checkpoint.
        model = LibreYOLO("LibreECs-seg.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=50,
            imgsz=640,
        )
    - label: LoRA
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=50,
            lora=True,
        )
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreECs.pt data=my-dataset.yaml
    - label: Pose
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs-pose.pt")
        metrics = model.val(data="my-pose-dataset.yaml")

        print(metrics["metrics/keypoints_mAP50-95"])
        print(metrics["metrics/keypoints_mAP50"])
    - label: Segmentación de instancias
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs-seg.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95(M)"])   # máscaras
        print(metrics["metrics/mAP50-95(B)"])   # boxes
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs.pt")
        model.export(format="onnx", imgsz=640)
        model.export(format="tensorrt", imgsz=640, half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreECs.pt format=onnx imgsz=640
        libreyolo export model=LibreECs-pose.pt format=onnx imgsz=640
        libreyolo export model=LibreECs-seg.pt format=onnx imgsz=640
    - label: Usar el archivo exportado
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La factoría enruta según la extensión del archivo, así que un artefacto
        # exportado se carga como cualquier checkpoint y devuelve el mismo Results.
        model = LibreYOLO("LibreECs.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
---

## Instalación

EdgeCrafter no necesita ningún extra opcional. Todo lo que importa está en la
instalación base.

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

La tarea viene del nombre del archivo, así que un checkpoint `-pose` o `-seg`
selecciona su propia cabeza y no acepta ningún argumento de tarea. Los tres
devuelven el objeto `Results` que devuelven todas las familias, con
`result.keypoints` añadido para pose y `result.masks` para segmentación. Pose
cubre una sola clase, persona, con los 17 keypoints de COCO, y el número queda
fijado al construir el modelo. No tiene cabeza de boxes, así que cada box de
pose es la extensión que envuelve a sus propios keypoints, y el tercer canal del
keypoint es una constante en lugar de una puntuación por punto.

`conf` y `max_det` filtran la selección de queries; `iou` se acepta por paridad
de API pero no tiene efecto, porque las tres cabezas decodifican un conjunto de
queries sin paso de NMS. Consulta [predicción](/docs/predict) para fuentes,
streaming y manejo de resultados.

## Variantes

Cuatro tamaños. Todos funcionan a la misma resolución de entrada, así que la
tabla los separa por número de parámetros y precisión.

<benchmark-table task="detect" />

<va-embed />

Upstream publica ECDet, ECPose y ECSeg como tres modelos separados en lugar de
un modelo con tres cabezas. Comparten el backbone ECViT y el encoder híbrido y
solo se diferencian en la cabeza, así que LibreYOLO los agrupa en una sola
familia y deja que el nombre del archivo del checkpoint indique la tarea. Por
tanto, una letra de tamaño significa el mismo backbone y el mismo encoder en los
tres, y predicción, validación y exportación aceptan los mismos argumentos sea
cual sea el que cargues.

## Entrenamiento

Las tres tareas se entrenan con `train()`, que lee la tarea del checkpoint
cargado y elige el trainer correspondiente.

<code-tabs name="train" />

Lo que sí se ha comprobado para detección y segmentación: paridad de inferencia
con upstream a 1e-5, capa a capa y por tamaño, y que la loss (la función de
pérdida) y un único paso de entrenamiento se ejecutan sobre entrada sintética.
Lo que no, según el propio docstring de `train()`: la convergencia de un
fine-tune completo, el entrenamiento multi-GPU, el paso de recarga del mejor
checkpoint al detener el aumento de datos, y el remapeo de clases de Objects365
a COCO. La ruta de pose sigue la receta publicada de DETRPose, un matcher
húngaro sobre costes de clase, L1 de keypoints y OKS con denoising contrastivo
de keypoints, y su convergencia tampoco se ha comprobado de extremo a extremo.

Si no lo tocas, el trainer ejecuta 74 epochs con `lr0=5e-4` y precisión mixta
activada, siguiendo la receta de upstream: AdamW, un schedule coseno plano, EMA
a 0.9999 y entradas normalizadas según ImageNet. Pose y segmentación requieren
ambas `imgsz` al tamaño nativo del checkpoint, porque su rejilla de anclas de
evaluación se construye al construir el modelo; un valor distinto lanza un error
antes de que empiece la ejecución. Pose además requiere un dataset de una sola
clase cuyo `data.yaml` declare `kpt_shape`, con un número de keypoints que
coincida con el de la cabeza.

`lora=True` solo se aplica a detección; pose y segmentación lanzan un
`ValueError` con él. En Apple silicon el trainer mantiene la ejecución en la GPU
y envía una operación a CPU, el backward de grid-sample dentro de la atención
deformable, que PyTorch no implementa en Metal.

Consulta [entrenamiento](/docs/train) para datasets, aumento de datos, multi-GPU
y loggers.

## Validación

`val()` devuelve un diccionario indexado por nombre de métrica, e imprime
resultados por clase si dejas `verbose` activado.

<code-tabs name="val" />

Pose reporta métricas OKS de keypoints bajo `metrics/keypoints_*`. Segmentación
reporta las máscaras bajo la clave `metrics/mAP50-95` a secas y repite ambas
vistas en una sola pasada, los boxes bajo `(B)` y las máscaras bajo `(M)`.

## Exportación

<export-matrix />

Un artefacto exportado se vuelve a cargar con `LibreYOLO()` según su extensión
de archivo, así que un archivo `.onnx` o `.engine` se comporta como un
checkpoint y devuelve el mismo `Results`. Pose y segmentación se exportan con
una entrada fija de 640 por 640 en lugar de formas dinámicas, y varios destinos
de detección también tienen lienzo fijo, incluidos OpenVINO, Paddle, MNN,
ExecuTorch y Core AI. [Exportación](/docs/export) lista los argumentos que
acepta cada formato y los extras que añaden unos pocos.

<code-tabs name="export" />

## Checkpoints

Todos los archivos de pesos publicados de esta familia.

<checkpoint-table />

## Licencia

<provenance-box></provenance-box>

## Cita

<citation-block />
