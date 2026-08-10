---
title: OpenVINO
seo_title: "Exportar a OpenVINO IR desde LibreYOLO"
description: "Convierte un modelo LibreYOLO a OpenVINO IR: el par model.xml y model.bin, la compresión de pesos en FP16, INT8 con NNCF e inferencia en CPU, GPU o NPU."
lead: "OpenVINO IR es el formato de runtime de Intel: un grafo model.xml junto a un blob de pesos model.bin. LibreYOLO exporta un ONNX intermedio, lo convierte con ov.convert_model y escribe un metadata.yaml en el mismo directorio."
keywords:
  - exportar yolo openvino
  - openvino ir
  - model.xml model.bin
  - ov.convert_model
  - cuantización int8 nncf
  - openvino npu
  - compress_to_fp16
last_verified: "1.5.0"
meta:
  - label: Flag
    value: 'export(format="openvino")'
    mono: true
  - label: Escribe
    value: "Un directorio con model.xml, model.bin y metadata.yaml"
  - label: Extra
    value: 'pip install "libreyolo[onnx,openvino]"'
    mono: true
  - label: Se recarga con
    value: 'LibreYOLO("weights/LibreYOLO9t_openvino")'
    mono: true
  - label: Formas
    value: "Sigue al ONNX intermedio: batch dinámico cuando dynamic=True"
  - label: Precisión
    value: "FP32, compresión de pesos en FP16 (half=True), INT8 vía NNCF (int8=True con data=)"
verification: "Leído de libreyolo/export/openvino.py, libreyolo/export/exporter.py, libreyolo/export/support.py, libreyolo/backends/openvino.py y pyproject.toml en la rama dev."
snippets:
  install:
    - label: Instalación
      language: bash
      code: |
        # El IR se convierte desde un ONNX intermedio, así que hacen falta ambos extras.
        pip install "libreyolo[onnx,openvino]"
    - label: INT8 necesita además NNCF
      language: bash
      code: |
        pip install nncf
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Escribe el directorio weights/LibreYOLO9t_openvino
        path = model.export(format="openvino")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format openvino
    - label: Argumentos
      language: python
      code: |
        model.export(
            format="openvino",
            imgsz=640,
            batch=1,
            dynamic=False,    # True mantiene un eje de batch dinámico a lo largo del IR
            half=False,       # True guarda los pesos en FP16
            int8=False,       # True ejecuta la cuantización post-entrenamiento de NNCF
            data=None,        # obligatorio cuando int8=True
            output_path=None, # None escribe weights/<stem>_openvino
        )
  int8:
    - label: INT8 con datos de calibración
      language: python
      code: |
        from libreyolo import LibreYOLO

        LibreYOLO("LibreYOLO9t.pt").export(
            format="openvino",
            int8=True,
            data="coco128.yaml",   # obligatorio: este formato no tiene valor por defecto
            fraction=1.0,
        )
  run:
    - label: A través de LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t_openvino")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: Seleccionar el dispositivo
      language: python
      code: |
        from libreyolo import LibreYOLO

        # "auto" y "cpu" se mapean a CPU, "gpu" y "cuda" se mapean a GPU,
        # cualquier otro valor se pasa en mayúsculas, por ejemplo "npu" -> NPU.
        model = LibreYOLO("weights/LibreYOLO9t_openvino", device="gpu")
    - label: OpenVINO puro
      language: python
      code: |
        import numpy as np
        import openvino as ov
        import yaml

        core = ov.Core()
        print(core.available_devices)

        compiled = core.compile_model("weights/LibreYOLO9t_openvino/model.xml", "CPU")
        outputs = compiled(np.zeros((1, 3, 640, 640), dtype=np.float32))
        print([tensor.shape for tensor in outputs.values()])

        # Los nombres de clase, la tarea y el tamaño de entrada viven en metadata.yaml junto al IR.
        meta = yaml.safe_load(open("weights/LibreYOLO9t_openvino/metadata.yaml"))
        print(meta["model_family"], meta["task"], meta["names"])

        # En esta vía, el preprocesado y el postprocesado corren de tu cuenta.
  support:
    - label: Comprobar una familia y tarea antes de exportar
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
---

## Instalación

<code-tabs name="install" />

La conversión pasa por un ONNX intermedio, así que el extra `onnx` forma parte del
requisito y no es un acompañante opcional. NNCF se instala aparte y solo hace
falta para `int8=True`.

## Exportación

<code-tabs name="export" />

El artefacto es un directorio, no un archivo. `weights/LibreYOLO9t_openvino`
contiene `model.xml`, `model.bin` y `metadata.yaml`, y se inserta `_fp16` antes
del sufijo cuando `half=True`. Mueve o copia el directorio entero; los tres
archivos son un único artefacto.

`half=True` activa `compress_to_fp16` al guardar. Eso es compresión de los pesos
dentro del IR, no un cambio en la precisión de inferencia que el dispositivo
elige en tiempo de ejecución.

### INT8

<code-tabs name="int8" />

`int8=True` ejecuta la cuantización post-entrenamiento de NNCF sobre un loader de
calibración de LibreYOLO con el preset mixed, y `data` es obligatorio: este
formato no tiene un fallback de ocho imágenes. Si falta NNCF se lanza un
`ImportError` que nombra el comando de instalación.

## Ejecutar el artefacto

<code-tabs name="run" />

`LibreYOLO()` reconoce cualquier directorio que contenga `model.xml` y devuelve el
mismo objeto `Results` que el checkpoint, leyendo los nombres de clase, la tarea,
el tamaño de entrada y el esquema de pose de `metadata.yaml`.

La cadena del dispositivo se mapea en lugar de pasarse tal cual. `auto` y `cpu`
compilan ambas para CPU, `gpu` y `cuda` compilan ambas para GPU, y cualquier otro
valor se pasa en mayúsculas a OpenVINO, que es la forma de llegar a un destino
NPU.

El tercer snippet es para quien no tenga LibreYOLO instalado. Ahí el
preprocesado, la decodificación, el NMS y el reescalado de coordenadas corren de
tu cuenta, y los nombres de clase solo existen en `metadata.yaml`.

## Restricciones

Un IR sin su `metadata.yaml` se carga igualmente, pero entonces el backend recurre
a 80 clases y a la tarea de detección, lo cual es incorrecto para cualquier otra
cosa. Mantén el directorio intacto.

Bloqueadas antes del trazado: la segmentación YOLO9, la segmentación RTMDet-Ins,
la detección con SSD, Faster R-CNN y RetinaNet, y el matting con BiRefNet o
FeyNobg, donde OpenVINO 2026.2 no consigue traducir la operación estándar de ONNX
`DeformConv-19` del decodificador de mattes compartido.

Cuando una combinación no está ni validada ni bloqueada, la vía del conversor está
disponible y el proyecto no ha registrado paridad de runtime en OpenVINO para
ella. Varias combinaciones están validadas con un contexto explícito adjunto, por
ejemplo la segmentación semántica DeepLabV3 con una entrada fija de 520 por 520 en
OpenVINO 2026.2 con la precisión de inferencia por defecto de la CPU, y la
estimación de mirada L2CS con un recorte de cara fijo de 448 por 448.
`libreyolo formats` imprime ese contexto para cada combinación.

Para la tabla completa de familias y tareas, consulta
[la matriz de exportación](/docs/reference/export-matrix). Para una combinación
concreta:

<code-tabs name="support" />
