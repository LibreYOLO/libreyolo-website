---
title: TFLite
seo_title: Exportar a TFLite (LiteRT) desde LibreYOLO
description: >-
  Exporta un modelo LibreYOLO a un FlatBuffer .tflite a través de onnx2tf:
  formas estáticas, solo FP32, entradas NHWC y las familias que convierten sin
  problemas.
lead: >-
  TFLite es el formato FlatBuffer que LiteRT ejecuta en objetivos móviles y
  embebidos. LibreYOLO exporta un grafo ONNX estático, lo convierte con onnx2tf
  en modo flatbuffer-direct y escribe los metadatos del modelo junto al
  artefacto en forma de sidecar JSON.
keywords:
  - exportar yolo tflite
  - litert
  - onnx2tf
  - ai-edge-litert
  - flatbuffer tflite
  - entrada nhwc tflite
  - inferencia en el edge
last_verified: 1.5.0
meta:
  - label: Flag
    value: export(format="tflite")
    mono: true
  - label: Escribe
    value: Un archivo .tflite más un sidecar de metadatos .tflite.json
  - label: Extra
    value: 'pip install "libreyolo[tflite]"'
    mono: true
  - label: Se recarga con
    value: LibreYOLO("weights/LibreYOLO9t.tflite")
    mono: true
  - label: Formas
    value: Solo estáticas. dynamic=True se rechaza.
  - label: Precisión
    value: Solo FP32. half=True e int8=True se rechazan.
  - label: Requiere
    value: >-
      Python 3.12 o superior, porque onnx2tf 2.4.x no publica wheels más
      antiguas
verification: >-
  Leído de libreyolo/export/tflite.py, libreyolo/export/exporter.py,
  libreyolo/export/support.py, libreyolo/backends/tflite.py y pyproject.toml en
  la rama dev.
snippets:
  install:
    - label: Instalación
      language: bash
      code: >
        # LiteRT es el nombre actual que Google da a TensorFlow Lite. Ambos
        extras

        # instalan la misma toolchain y producen la misma salida .tflite.

        pip install "libreyolo[tflite]"
    - label: Comprobar antes la versión de Python
      language: bash
      code: |
        python -c "import sys; print(sys.version_info >= (3, 12))"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Escribe weights/LibreYOLO9t.tflite y weights/LibreYOLO9t.tflite.json
        path = model.export(format="tflite", imgsz=640)
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format tflite --imgsz 640

        # "litert" se acepta como alias y resuelve al mismo exportador.
        libreyolo export --model LibreYOLO9t.pt --format litert --imgsz 640
    - label: Argumentos
      language: python
      code: |
        model.export(
            format="tflite",
            imgsz=640,        # int, o (alto, ancho)
            batch=1,
            simplify=True,    # onnxsim sobre el intermedio ONNX
            output_path=None, # None escribe weights/<stem>.tflite
            verbose=False,    # True muestra el log de onnx2tf
        )

        # dynamic=True lanza ValueError: el conversor necesita formas estáticas.
        # half=True e int8=True se rechazan antes del trazado.
  run:
    - label: A través de LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t.tflite")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: LiteRT puro
      language: python
      code: >
        import json


        import numpy as np

        from ai_edge_litert.interpreter import Interpreter


        interpreter = Interpreter(model_path="weights/LibreYOLO9t.tflite")

        interpreter.allocate_tensors()

        detail = interpreter.get_input_details()[0]

        print(detail["shape"], detail["dtype"])   # NHWC, no NCHW


        interpreter.set_tensor(detail["index"], np.zeros(detail["shape"],
        np.float32))

        interpreter.invoke()

        for output in interpreter.get_output_details():
            print(output["name"], interpreter.get_tensor(output["index"]).shape)

        # Los nombres de clase, la tarea y el tamaño de entrada viven en el
        sidecar.

        meta = json.load(open("weights/LibreYOLO9t.tflite.json"))

        print(meta["model_family"], meta["task"], meta["names"])


        # El preprocesado, la transposición de NCHW a NHWC y el postprocesado
        corren de tu cuenta.
  support:
    - label: Comprobar una familia y tarea antes de exportar
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: fa2deaa0ef6d9978
---

## Instalación

<code-tabs name="install" />

El extra trae `onnx2tf` para la conversión y `ai-edge-litert` para ejecutar el
resultado, ambos detrás de un marcador de Python 3.12. En un intérprete más
antiguo, la exportación lanza un `ImportError` que nombra el requisito de versión
en lugar de fallar dentro del conversor.

`libreyolo[litert]` instala exactamente lo mismo. La cadena de formato `litert` es
un alias de `tflite`, y el archivo de salida es un `.tflite` en cualquier caso.

## Exportación

<code-tabs name="export" />

La familia y la tarea se comprueban antes que nada, así que una combinación no
soportada falla de inmediato con el error concreto del conversor o del runtime que
la dejó fuera, no con un mensaje genérico. La conversión en sí es una llamada a
`onnx2tf` como subproceso, en modo `flatbuffer_direct`, sobre un intermedio ONNX
estático.

Los metadatos son un sidecar. `weights/LibreYOLO9t.tflite.json` lleva la familia,
la tarea, los nombres de clase, el tamaño de entrada y el esquema de pose; el
FlatBuffer en sí no tiene campo de metadatos de LibreYOLO, así que los dos
archivos viajan juntos.

## Ejecutar el artefacto

<code-tabs name="run" />

`LibreYOLO()` despacha por el sufijo `.tflite` y devuelve el mismo objeto `Results`
que el checkpoint. El backend lee el sidecar, transpone el blob NCHW a NHWC cuando
el intérprete pide una entrada channels-last, aplica la escala de cuantización y el
zero point del intérprete donde estén presentes, y transpone las salidas de vuelta
al layout que espera el postprocesado de LibreYOLO.

El segundo snippet es la vía del runtime puro. El preprocesado, la transposición de
layout, la decodificación, el NMS y el reescalado de coordenadas corren todos de tu
cuenta ahí, y el detalle del layout es el que más fácil se pasa por alto: onnx2tf
emite entradas channels-last, así que un blob con forma `(1, 3, 640, 640)` no
encajará.

## Restricciones

Solo formas estáticas. `dynamic=True` lanza `ValueError` antes del trazado, y el
lienzo de exportación queda fijado en el valor al que se resolvió `imgsz`.

Solo FP32. `half=True` e `int8=True` se rechazan ambos durante la validación, así
que hoy por hoy el despliegue cuantizado no es alcanzable desde este exportador.

La cobertura aquí es más estrecha que la de los formatos de grafo, y la decide la
medición en lugar de la familia. Entre las combinaciones validadas están la
detección con YOLO9, YOLOX y YOLO-NAS, la segmentación semántica con PIDNet, las
cuatro familias de clasificación CNN, el embedding con DINOv2 y SigLIP2, la
clasificación con SigLIP2, los bordes con TEED y DexiNed, y la restauración con
Real-ESRGAN y SwinIR. SwinIR arrastra una salvedad extra: la paridad se mantiene
cuando las dimensiones de origen coinciden exactamente con el lienzo de
exportación, y los orígenes más pequeños se rellenan hasta el lienzo antes de que
corra el transformer, lo que puede divergir de la inferencia nativa a tamaño
variable.

Las entradas bloqueadas nombran el fallo exacto, y conviene leerlo antes de
intentar un rodeo. Unos cuantos ejemplos: la detección con RF-DETR convierte en su
lienzo nativo de 384 pero LiteRT no puede reservarlo porque `STRIDED_SLICE` recibe
una entrada por encima del rango 5-D que soporta; PicoDet se rechaza porque un
`RESHAPE` mapea 19.200 elementos de entrada a 9.600 de salida; D-FINE hace caer al
conversor en el manejo de formas de `GatherElements`; RTMDet exporta y recarga con
la paridad en crudo intacta, pero los boxes públicos bajan a 0,911 de IoU con 29,9
px de deriva de coordenadas.

Para la tabla completa de familias y tareas, consulta
[la matriz de exportación](/docs/reference/export-matrix). Para una combinación
concreta, incluida la cadena de motivo detrás de un bloqueo:

<code-tabs name="support" />
