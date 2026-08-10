---
title: ncnn
seo_title: Exportar a ncnn desde LibreYOLO
description: >-
  Exporta un modelo LibreYOLO a ncnn a través de PNNX: el par param y bin, el
  lienzo de exportación fijo, la reescritura del Focus de YOLOX y qué familias
  convierten.
lead: >-
  ncnn es la biblioteca de inferencia en CPU de Tencent para targets móviles.
  LibreYOLO convierte a través de PNNX, escribiendo un grafo model.ncnn.param
  junto a un archivo de pesos model.ncnn.bin y un metadata.yaml que lleva la
  familia, la tarea y los nombres de clase.
keywords:
  - exportar yolo ncnn
  - pnnx
  - model.ncnn.param
  - inferencia cpu movil
  - ncnn extractor
  - focus pixel_unshuffle
last_verified: 1.5.0
meta:
  - label: Flag
    value: export(format="ncnn")
    mono: true
  - label: Escribe
    value: 'Un directorio con model.ncnn.param, model.ncnn.bin y metadata.yaml'
  - label: Extra
    value: 'pip install "libreyolo[ncnn]"'
    mono: true
  - label: Se recarga con
    value: LibreYOLO("weights/LibreYOLO9t_ncnn")
    mono: true
  - label: Formas
    value: Fijas. Los metadatos registran dynamic=False independientemente del flag.
  - label: Precisión
    value: Solo FP32. half=True e int8=True se rechazan.
verification: >-
  Leído de libreyolo/export/ncnn.py, libreyolo/export/exporter.py,
  libreyolo/export/support.py, libreyolo/backends/ncnn.py y pyproject.toml en la
  rama dev.
snippets:
  install:
    - label: Instalación
      language: bash
      code: |
        # pnnx convierte, ncnn ejecuta el resultado.
        pip install "libreyolo[ncnn]"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Escribe el directorio weights/LibreYOLO9t_ncnn
        path = model.export(format="ncnn", imgsz=640)
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format ncnn --imgsz 640
    - label: Argumentos
      language: python
      code: |
        model.export(
            format="ncnn",
            imgsz=640,        # int, o (alto, ancho)
            batch=1,
            simplify=True,    # solo se aplica a la ruta de respaldo por ONNX
            opset=None,       # automático; solo se aplica a la ruta de respaldo por ONNX
            output_path=None, # None escribe weights/<stem>_ncnn
        )

        # half=True e int8=True se rechazan durante la validación.
  run:
    - label: A través de LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t_ncnn")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: ncnn a secas
      language: python
      code: |
        import ncnn
        import numpy as np
        import yaml

        directory = "weights/LibreYOLO9t_ncnn"
        net = ncnn.Net()
        net.load_param(f"{directory}/model.ncnn.param")
        net.load_model(f"{directory}/model.ncnn.bin")

        # ncnn toma una única imagen CHW, no un batch.
        mat_in = ncnn.Mat(np.zeros((3, 640, 640), dtype=np.float32))
        extractor = net.create_extractor()
        extractor.input("in0", mat_in)
        ret, mat_out = extractor.extract("out0")
        print(ret, np.array(mat_out).shape)

        meta = yaml.safe_load(open(f"{directory}/metadata.yaml"))
        print(meta["model_family"], meta["task"], meta["names"])

        # El preprocesado y el postprocesado corren de tu cuenta en esta ruta.
  support:
    - label: Comprobar una familia y una tarea antes de exportar
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: 9a849a16a3b32334
---

## Instalación

<code-tabs name="install" />

El extra trae las dos mitades del toolchain: `pnnx` realiza la conversión y
`ncnn` ejecuta el resultado. Ninguna de las dos pasa por ONNX en la ruta principal.

## Exportación

<code-tabs name="export" />

El artefacto es un directorio. `weights/LibreYOLO9t_ncnn` contiene
`model.ncnn.param`, `model.ncnn.bin` y `metadata.yaml`; los tres son un único
artefacto y se mueven juntos.

La conversión intenta primero PNNX directamente desde PyTorch. Si eso falla, exporta
un grafo ONNX estático a un directorio temporal y llama sobre él a la herramienta de
línea de comandos `pnnx`, y la exportación solo lanza un error cuando fallan ambas
rutas, informando de los dos errores. Por tanto, `opset` y `simplify` solo afectan a
la ruta de respaldo.

YOLOX necesita una reescritura para poder convertirse siquiera. Su capa Focus usa
slicing con stride, que PNNX no puede bajar de nivel, así que la exportación la
sustituye por `pixel_unshuffle` y permuta los canales de entrada de la convolución
siguiente para compensar el distinto orden de canales. La salida es numéricamente
idéntica, y los pesos originales se restauran después de la exportación.

## Ejecutar el artefacto

<code-tabs name="run" />

`LibreYOLO()` reconoce cualquier directorio que contenga `model.ncnn.param` y
`model.ncnn.bin`, lee `metadata.yaml` y devuelve el mismo objeto `Results` que el
checkpoint.

El segundo snippet es la ruta del runtime a secas, y hay dos detalles que difieren
de todos los demás formatos aquí. ncnn trabaja sobre una única imagen CHW en lugar
de sobre un batch, así que no hay eje de batch inicial. Los nombres de los blobs
salen del archivo `.param`; PNNX escribe `in0` y `out0` por convención, y el backend
parsea el archivo en lugar de darlos por supuestos. El preprocesado, la
decodificación, el NMS y el reescalado de coordenadas corren de tu cuenta en esa
ruta.

## Restricciones

FP32 sobre un lienzo fijo. `half=True` e `int8=True` se rechazan ambos durante la
validación, y los metadatos exportados registran `dynamic=False` dijera lo que
dijera el flag, de modo que ningún backend asume un eje que el grafo no tiene.

Todas las familias de tipo DETR se rechazan en el preflight: `detr`,
`deformable_detr`, `dinodetr`, `dfine`, `lwdetr`, `deim`, `deimv2`, `rtdetr`,
`rtdetrv2`, `rtdetrv4`, `rfdetr` y `ec`. El mensaje es el mismo para todas: que el
modelo necesita operaciones de decoder o de sampling no disponibles en ncnn, y
apunta en su lugar a ONNX, OpenVINO, TorchScript o TensorRT.

Lo que sí convierte es amplio por el lado convolucional: YOLO9 y YOLO9-E2E, YOLOX,
PicoDet, YOLO-NAS en detección y pose, los detectores más antiguos YOLO1, YOLO3,
YOLO4 y YOLO7, las cuatro familias de clasificación CNN, la segmentación semántica
PIDNet, la detección de puntos FOMO a 96 por 96 fijos, ZipDepth, NAFNet y
Real-ESRGAN.

Las entradas bloqueadas nombran el fallo concreto. Los grafos transformer suelen
dejar atrás nodos `pnnx.Expression` no soportados, lo que produce una red sin blob
de entrada ejecutable, y eso es lo que detiene a DINOv2, CLIP, SigLIP2 y SegFormer.
BiRefNet necesita la convolución deformable de torchvision, que PNNX no puede bajar
de nivel. El grafo convertido de YOLO2 termina el runtime de ncnn en Windows con una
división entera por cero nativa durante la extracción de la salida.

Para la rejilla completa de familias y tareas, consulta
[la matriz de exportación](/docs/reference/export-matrix). Para una combinación
concreta:

<code-tabs name="support" />
