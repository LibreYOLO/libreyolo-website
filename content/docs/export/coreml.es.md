---
title: Core ML
seo_title: "Exportar a Core ML desde LibreYOLO"
description: "Exporta un detector LibreYOLO a un .mlpackage de Core ML: el contrato de entrada ImageType, FP16, las compute units, el NMS embebido y las cuatro familias soportadas."
lead: "Core ML es el formato de modelos on-device de Apple. LibreYOLO traza el detector detrás de un wrapper de preprocesado propio de cada familia, de modo que el grafo convertido siempre recibe una entrada de imagen RGB canónica, y después escribe un .mlpackage en formato ML Program con los metadatos del modelo adjuntos."
keywords:
  - exportar yolo coreml
  - mlpackage
  - coremltools
  - ct.ImageType
  - apple neural engine
  - compute_units
  - nms embebido coreml
  - yolo en ios
last_verified: "1.5.0"
meta:
  - label: Flag
    value: 'export(format="coreml")'
    mono: true
  - label: Escribe
    value: "Un bundle .mlpackage (un directorio) en formato ML Program"
  - label: Extra
    value: 'pip install "libreyolo[coreml]"'
    mono: true
  - label: Se recarga con
    value: 'LibreYOLO("weights/LibreYOLO9t.mlpackage") en macOS'
    mono: true
  - label: Formas
    value: "Fijas. La entrada es un ct.ImageType de forma rígida."
  - label: Precisión
    value: "FP32, FP16 (half=True). Sin INT8."
  - label: Familias
    value: "Solo detección, para yolox, yolo9, rtdetr y rfdetr"
verification: "Leído de libreyolo/export/coreml.py, libreyolo/export/exporter.py, libreyolo/export/support.py, libreyolo/backends/coreml.py y pyproject.toml en la rama dev."
snippets:
  install:
    - label: Instalación
      language: bash
      code: |
        pip install "libreyolo[coreml]"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Escribe el bundle weights/LibreYOLO9t.mlpackage
        path = model.export(format="coreml")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format coreml
    - label: Argumentos
      language: python
      code: |
        model.export(
            format="coreml",
            imgsz=640,
            batch=1,
            half=False,           # True convierte con precisión de cómputo FLOAT16
            compute_units="all",  # all | cpu_and_gpu | cpu_and_ne | cpu_only
            output_path=None,     # None escribe weights/<stem>.mlpackage
        )

        # dynamic se acepta, pero la entrada es un ct.ImageType de forma fija,
        # y los metadatos embebidos registran dynamic=False en cualquier caso.
  nms:
    - label: Embeber la capa NMS de Apple
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Solo detección con YOLOX y YOLO9, batch 1.
        LibreYOLO("LibreYOLO9t.pt").export(
            format="coreml",
            nms=True,
            conf=0.25,
            iou=0.45,
        )
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format coreml --nms \
          --conf 0.25 --iou 0.45
  run:
    - label: A través de LibreYOLO, en macOS
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO(
            "weights/LibreYOLO9t.mlpackage",
            compute_units="all",   # o cpu_and_ne para fijar el Neural Engine
        )
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: coremltools a secas
      language: python
      code: |
        import coremltools as ct
        from PIL import Image

        mlmodel = ct.models.MLModel("weights/LibreYOLO9t.mlpackage")
        print(mlmodel.user_defined_metadata["model_family"])
        print(mlmodel.user_defined_metadata["names"])

        # La entrada es una imagen llamada "image" con el tamaño fijo de exportación.
        image = Image.open(SAMPLE_IMAGE).convert("RGB").resize((640, 640))
        out = mlmodel.predict({"image": image})
        print({name: value.shape for name, value in out.items()})

        # El letterboxing y el postprocesado corren de tu cuenta en esta ruta.
  support:
    - label: Comprobar una familia y una tarea antes de exportar
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
---

## Instalación

<code-tabs name="install" />

La predicción necesita macOS. `LibreYOLO()` rechaza un `.mlpackage` en cualquier otra
plataforma con un mensaje que nombra la actual, y la matriz de soporte registra estas
combinaciones como disponibles porque la paridad en tiempo de ejecución necesita un runner macOS.

## Exportación

<code-tabs name="export" />

El bundle se escribe en `weights/` con el stem del checkpoint, añadiendo `_fp16`
cuando `half=True`. Un `.mlpackage` es un directorio, así que copia el árbol entero.

Todas las familias se trazan detrás de un wrapper de preprocesado, de modo que el grafo
convertido recibe una única entrada canónica: RGB, `scale=1/255`, sin bias, declarada como
`ct.ImageType`. El wrapper absorbe la convención propia de cada familia, que es BGR en
el rango de 0 a 255 para YOLOX, media y desviación estándar de ImageNet para RF-DETR,
e identidad para YOLO9 y RT-DETR. Por eso un consumidor de Core ML alimenta una
imagen normal en lugar de un tensor específico de la familia.

La conversión apunta a ML Program con un deployment target mínimo de iOS 15.
`compute_units` se almacena en el modelo convertido y se puede sobrescribir de nuevo
al cargar el artefacto.

Los metadatos del modelo van a `user_defined_metadata` como cadenas, que es de donde el
backend lee la familia, la tarea, los nombres de clase, el tamaño de entrada y el esquema de pose.

### NMS embebido

<code-tabs name="nms" />

`nms=True` envuelve el modelo en un pipeline de Core ML que termina en la capa
`NonMaximumSuppression` de Apple. El resultado tiene dos salidas: `confidence`, de forma
`N` por el número de clases, y `coordinates`, de forma `N` por 4 como `xywh` normalizado.

Se aplica solo a la detección con YOLOX y YOLO9, y exige batch 1. Las familias de
estilo DETR se rechazan por nombre, porque la predicción de conjuntos hace un top-k sobre
queries y clases sin paso de IoU y no puede usar esa capa. `max_det` tampoco se expone
aquí; cuando el tope de detecciones importa, usa el
[NMS embebido de ONNX](/docs/export/onnx) en su lugar.

## Ejecutar el artefacto

<code-tabs name="run" />

`LibreYOLO()` reconoce un directorio con el sufijo `.mlpackage` y devuelve el
mismo objeto `Results` que el checkpoint. `compute_units` es el único argumento que la
factoría pasa a través para este formato, y acepta `all`, `cpu_and_gpu`,
`cpu_and_ne` y `cpu_only`. El argumento `device` se ignora, porque Core ML
enruta a través de las compute units.

El segundo snippet es la ruta del runtime a secas. Ahí el letterboxing, el decodificado, el NMS y
el reescalado de coordenadas corren de tu cuenta, y los nombres de clase viven en
`user_defined_metadata`.

## Restricciones

Cuatro familias, solo detección: `yolox`, `yolo9`, `rtdetr` y `rfdetr`. Cualquier otra
cosa se rechaza en el preflight, porque el wrapper de preprocesado consciente de la familia es
lo que hace correcto el contrato de entrada de imagen fija, y una familia fuera de él se
convertiría con la normalización equivocada. El error nombra ONNX y TorchScript como
alternativas.

La forma de entrada queda fijada de manera rígida por `ct.ImageType`, así que `dynamic=True` no cambia nada
y los metadatos registran `dynamic=False`. Exporta un segundo bundle para una segunda
resolución.

`half=True` convierte con precisión de cómputo FP16. No hay ruta a INT8 desde este
exportador.

Para la rejilla completa de familias y tareas, consulta
[la matriz de exportación](/docs/reference/export-matrix). Para el formato on-device más
reciente de Apple, consulta [Core AI](/docs/export/coreai). Para una sola combinación:

<code-tabs name="support" />
