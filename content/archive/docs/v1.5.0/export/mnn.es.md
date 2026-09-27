---
title: MNN
seo_title: Exportar a MNN desde LibreYOLO
description: >-
  Exporta un detector LibreYOLO a MNN a través de ONNX y mnnconvert: una forma
  NCHW fija, FP32 en CPU y un sidecar de metadatos que el contrato de runtime
  exige.
lead: >-
  MNN es el motor de inferencia ligero de Alibaba. LibreYOLO exporta un grafo
  ONNX estático, lo convierte con la herramienta mnnconvert que trae el paquete
  MNN y escribe un sidecar JSON que registra los nombres de entrada y salida, la
  forma de entrada fija y los nombres de clase.
keywords:
  - exportar yolo mnn
  - mnnconvert
  - inferencia mnn
  - inferencia detector en móvil
  - forma nchw fija
last_verified: 1.5.0
meta:
  - label: Flag
    value: export(format="mnn")
    mono: true
  - label: Escribe
    value: Un archivo .mnn más un sidecar de metadatos .mnn.json
  - label: Extra
    value: 'pip install "libreyolo[mnn]"'
    mono: true
  - label: Se recarga con
    value: LibreYOLO("weights/LibreYOLO9t.mnn")
    mono: true
  - label: Formas
    value: NCHW fija. dynamic=True se rechaza.
  - label: Precisión
    value: 'Solo FP32, solo CPU.'
  - label: Tareas
    value: Solo detección en esta versión
verification: >-
  Leído de libreyolo/export/mnn.py, libreyolo/export/exporter.py,
  libreyolo/export/support.py, libreyolo/backends/mnn.py y pyproject.toml en la
  rama dev.
snippets:
  install:
    - label: Instalación
      language: bash
      code: >
        # El extra incluye libreyolo[onnx]: MNN convierte desde un intermedio
        ONNX.

        pip install "libreyolo[mnn]"
    - label: Comprobar que el conversor está en el PATH
      language: bash
      code: |
        mnnconvert --version
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Escribe weights/LibreYOLO9t.mnn y weights/LibreYOLO9t.mnn.json
        path = model.export(format="mnn", imgsz=640)
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format mnn --imgsz 640
    - label: Argumentos
      language: python
      code: |
        model.export(
            format="mnn",
            imgsz=640,        # int, o (alto, ancho)
            batch=1,          # queda fijado en el artefacto
            simplify=True,    # pasa onnxsim sobre el intermedio ONNX
            output_path=None, # None escribe weights/<stem>.mnn
            verbose=False,    # True muestra el log de mnnconvert
        )

        # dynamic=True lanza ValueError. half=True e int8=True se rechazan.
  run:
    - label: A través de LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t.mnn")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: MNN puro
      language: python
      code: >
        import json


        import MNN

        import numpy as np


        meta = json.load(open("weights/LibreYOLO9t.mnn.json"))

        print(meta["mnn_input_names"], meta["mnn_output_names"],
        meta["mnn_input_shape"])


        runtime = MNN.nn.create_runtime_manager(
            ({"backend": 0, "precision": 1, "numThread": 4},)
        )

        module = MNN.nn.load_module_from_file(
            "weights/LibreYOLO9t.mnn",
            meta["mnn_input_names"],
            meta["mnn_output_names"],
            runtime_manager=runtime,
            dynamic=False,
            shape_mutable=False,
        )


        blob = np.zeros(meta["mnn_input_shape"], dtype=np.float32)

        input_var = MNN.expr.const(
            blob, list(blob.shape), MNN.expr.NCHW, MNN.expr.float
        )

        outputs = module.forward([input_var])

        for out in outputs:
            print(np.array(MNN.expr.convert(out, MNN.expr.NCHW).read()).shape)

        # En esta vía, el preprocesado y el postprocesado corren de tu cuenta.
  support:
    - label: Comprobar una familia y tarea antes de exportar
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: 68fad34d07aea149
---

## Instalación

<code-tabs name="install" />

El extra incluye `libreyolo[onnx]`, porque la conversión se ejecuta sobre un
intermedio ONNX. También trae el ejecutable `mnnconvert`, que el exportador busca
primero junto al intérprete de Python activo y después en el `PATH`. Si falta el
conversor, se lanza un `ImportError` que nombra el comando de instalación, en
lugar de fallar a mitad de la conversión.

## Exportación

<code-tabs name="export" />

Antes de entregar el grafo, el exportador lee el contrato de entrada de ONNX y
rechaza todo lo que no puede expresar: más de una entrada de imagen, o una forma
de entrada con una dimensión simbólica. MNN en esta versión exige una forma NCHW
totalmente fija, y `batch` queda fijado en el artefacto en lugar de negociarse al
cargarlo.

El sidecar no es papeleo opcional. `weights/LibreYOLO9t.mnn.json` registra los
nombres de entrada y salida, la forma de entrada fija, el batch, los nombres de
clase, la versión de MNN utilizada y el backend para el que se construyó el
artefacto, y el runtime valida cada uno de esos campos al cargar.

En Windows, MNN 3.6.1 a veces completa la conversión y luego termina durante el
cierre del proceso con una violación de acceso o un estado de fail-fast. El
exportador reconoce esos códigos de salida concretos y trata la conversión como
correcta cuando el archivo de salida está presente.

## Ejecutar el artefacto

<code-tabs name="run" />

`LibreYOLO()` despacha por el sufijo `.mnn` y devuelve el mismo objeto `Results`
que el checkpoint. La carga es estricta por diseño: el sidecar tiene que declarar
`format=mnn`, `mnn_backend=cpu`, `dynamic=false`, `precision=fp32`, un tamaño, una
tarea de detección, una forma NCHW fija y positiva que concuerde con el tamaño de
imagen registrado, y nombres de clase que cubran todos los índices de 0 a `nc - 1`.
Cualquier discrepancia lanza un error en lugar de adivinar.

Predecir con un `imgsz` distinto de aquel para el que se construyó el artefacto
también lanza un error, y `device` se ignora con un aviso, porque aquí las
exportaciones a MNN se ejecutan en CPU.

El segundo snippet es la vía del runtime puro. El preprocesado, la decodificación,
el NMS y el reescalado de coordenadas corren de tu cuenta ahí, y los nombres de
entrada y salida salen del sidecar porque el cargador de módulos de MNN los quiere
de forma explícita.

## Restricciones

Solo detección. El backend rechaza cualquier otra tarea al cargar, y el lado de la
exportación hace lo mismo: fuera de las combinaciones registradas, la comprobación
previa lanza un error con "MNN v1 has no implemented runtime contract for this
family and task."

FP32, CPU, forma fija. `dynamic=True` lanza `ValueError`, y `half=True` e
`int8=True` se rechazan durante la validación.

Las familias de detección validadas son YOLO9, YOLO9-E2E, YOLO9-P2, RF-DETR, EC,
RT-DETR, RT-DETRv2, RT-DETRv4, D-FINE, DEIM y YOLO-NAS, cada una cubierta por la
conversión, una recarga del artefacto recién escrito, la ejecución en CPU con MNN,
las comprobaciones de metadatos y la paridad de detecciones post-NMS emparejadas
contra el modelo de PyTorch. DEIMv2 convierte, recarga, ejecuta y preserva las
detecciones post-NMS, pero su ruta ONNX intermedia tiene una paridad incompleta de
puntuaciones a nivel de query, así que se registra como disponible en lugar de
validada.

Para la tabla completa de familias y tareas, consulta
[la matriz de exportación](/docs/reference/export-matrix). Para una combinación
concreta:

<code-tabs name="support" />
