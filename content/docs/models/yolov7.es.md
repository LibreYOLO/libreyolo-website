---
title: YOLOv7
families: [yolo7]
seo_title: "YOLOv7 en LibreYOLO: predecir, entrenar y exportar bajo MIT"
description: "Ejecuta YOLOv7 en LibreYOLO para detección de objetos: instala, predice, entrena, valida y exporta, con código y pesos bajo licencia MIT."
lead: "YOLOv7 es un detector de una sola etapa basado en anchors cuya cabeza añade offsets de conocimiento implícito aprendidos antes de la convolución final. LibreYOLO incluye su único tamaño publicado para detección."
keywords: [YOLOv7, "detección de objetos", "detector basado en anchors", "yolov7 python", "entrenar yolov7", "conocimiento implícito", ImplicitA, "detección de objetos en tiempo real"]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO7b.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLO7b.pt source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg save=True
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO7b.pt")
        model.train(data="my-dataset.yaml", epochs=300, imgsz=640, batch=16, lr0=0.01)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO7b.pt data=my-dataset.yaml \
          epochs=300 imgsz=640 batch=16 lr0=0.01
    - label: Arranque en caliente desde un modelo nuevo
      language: python
      code: |
        from libreyolo import LibreYOLO7

        # pretrained=True siempre carga el checkpoint publicado LibreYOLO7b.pt,
        # sin importar con qué se construyó esta instancia. Construir la clase
        # directamente, en lugar de a través de LibreYOLO(), arranca sin ningún
        # peso cargado.
        model = LibreYOLO7(None, size="b")
        model.train(data="my-dataset.yaml", epochs=300, pretrained=True)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO7b.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLO7b.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO7b.pt")
        model.export(format="onnx", imgsz=640)
        model.export(format="tensorrt", imgsz=640, half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreYOLO7b.pt format=onnx imgsz=640
        libreyolo export model=LibreYOLO7b.pt format=tensorrt imgsz=640 half=True
    - label: Usar el archivo exportado
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La factoría enruta según el sufijo del archivo, así que un artefacto
        # exportado se carga como cualquier checkpoint y devuelve el mismo objeto Results.
        model = LibreYOLO("LibreYOLO7b.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
---

## Instalación

YOLOv7 no necesita ningún extra más allá del paquete base.

```bash
pip install libreyolo
```

## Predicción

Los pesos se descargan de Hugging Face en el primer uso y se guardan en la caché
local.

<code-tabs name="predict" />

El objeto `Results` devuelto es el mismo que devuelve cada familia, así que
cambiar a otro detector es un cambio de una línea. `conf` fija el umbral de
confianza e `iou` el umbral de NMS que se aplica después de decodificar la
cabeza basada en anchors. Consulta [predicción](/docs/predict) para fuentes,
streaming y manejo de resultados.

## Variantes

LibreYOLO incluye un único tamaño, `b`. Upstream publica un solo modelo YOLOv7,
así que no hay tamaño entre el que elegir.

## Entrenamiento

<code-tabs name="train" />

`pretrained` sí se lee, a diferencia del argumento del mismo nombre que no hace
nada en algunas otras familias de aquí: pasa `True` para arrancar en caliente
desde el checkpoint publicado `LibreYOLO7b.pt` (se descarga solo), o una ruta o
un nombre para cualquier otra cosa. Ese checkpoint publicado es COCO de 80
clases, así que pedirlo sobre un modelo ya reconstruido para otro número de
clases primero lo reconstruye de vuelta a 80, lo carga y después transfiere
cada tensor cuya forma coincide al número de clases de la cabeza de destino en
cuanto se lee el número de clases del dataset. `resume=True` no se puede
combinar con `pretrained`. Si se deja en el valor por defecto `None`, el
entrenamiento continúa desde aquello con lo que se construyó el modelo, o desde
una inicialización aleatoria si no se cargó nada.

Si no se toca nada más, el trainer ejecuta 300 épocas con `lr0=0.01`, SGD con
momentum 0.937, un warmup de 3 épocas, y la misma asignación SimOTA y la misma
fase final de 15 épocas sin aumento de datos que usa YOLOX, adaptadas a la
cabeza basada en anchors. La única diferencia: YOLOX añade durante esas épocas
finales un refinamiento L1 de la regresión de boxes que v7 se salta, porque la
función de pérdida SimOTA de v7 no lleva una rama L1 de offsets crudos que
refinar.

Consulta [entrenamiento](/docs/train) para datasets, aumento de datos,
multi-GPU y loggers.

## Validación

`val()` devuelve un diccionario de claves `metrics/` que cubren precisión,
recall, mAP 50 y mAP 50-95, medidas contra cualquier dataset en el formato con
el que entrenaste.

<code-tabs name="val" />

## Exportación

<export-matrix />

Un artefacto exportado se vuelve a cargar a través de `LibreYOLO()` por su
sufijo de archivo, así que un archivo `.onnx` o `.engine` se comporta como un
checkpoint y devuelve el mismo `Results`. Ejecutar el grafo en un runtime
desnudo, sin LibreYOLO instalado, también está soportado, pero entonces el
preprocesamiento y el postprocesamiento corren de tu cuenta.

<code-tabs name="export" />

## Checkpoints

Todos los archivos de pesos publicados de esta familia.

<checkpoint-table />

## Licencia

<provenance-box></provenance-box>

## Cita

<citation-block />
