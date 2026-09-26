---
title: TensorRT
seo_title: Exportar a TensorRT desde LibreYOLO
description: >-
  Construye un engine de TensorRT a partir de un modelo LibreYOLO: el intermedio
  ONNX, las builds FP16 e INT8, los perfiles de batch dinámico y los límites de
  portabilidad del engine.
lead: >-
  TensorRT compila un grafo en un engine ajustado a una GPU concreta. LibreYOLO
  exporta primero un intermedio ONNX, lo parsea con el parser de ONNX de
  TensorRT, construye el engine y escribe los metadatos del modelo a su lado
  como un sidecar JSON.
keywords:
  - exportar yolo tensorrt
  - engine tensorrt
  - trt fp16
  - calibración int8 tensorrt
  - perfil de optimización
  - batch dinámico tensorrt
  - hardware compatibility level
last_verified: 1.5.0
meta:
  - label: Flag
    value: export(format="tensorrt")
    mono: true
  - label: Escribe
    value: Un archivo .engine más un sidecar de metadatos .engine.json
  - label: Extra
    value: 'pip install "libreyolo[onnx,tensorrt]"'
    mono: true
  - label: Se recarga con
    value: LibreYOLO("weights/LibreYOLO9t.engine")
    mono: true
  - label: Formas
    value: >-
      Estáticas por defecto; dynamic=True añade un perfil de optimización sobre
      el eje de batch
  - label: Precisión
    value: 'FP32, FP16 (half=True), INT8 (int8=True con data=)'
  - label: Requiere
    value: >-
      Una GPU NVIDIA al construir y al ejecutar. Los engines no se mueven entre
      arquitecturas de GPU.
verification: >-
  Leído de libreyolo/export/tensorrt.py, libreyolo/export/exporter.py,
  libreyolo/export/support.py, libreyolo/backends/tensorrt.py y pyproject.toml
  en la rama dev.
snippets:
  install:
    - label: Instalación
      language: bash
      code: >
        # El engine se construye a partir de un intermedio ONNX, así que hacen
        falta ambos extras.

        pip install "libreyolo[onnx,tensorrt]"
    - label: Comprobar el toolchain antes de construir
      language: bash
      code: >
        python -c "import tensorrt, torch; print(tensorrt.__version__,
        torch.cuda.is_available())"
  export:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO9t.pt")


        # Escribe weights/LibreYOLO9t_fp16.engine y
        weights/LibreYOLO9t_fp16.engine.json

        path = model.export(format="tensorrt", half=True)

        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format tensorrt --half
    - label: Argumentos
      language: python
      code: |
        model.export(
            format="tensorrt",
            imgsz=640,
            batch=1,
            half=False,
            int8=False,
            data=None,                      # obligatorio cuando int8=True
            dynamic=False,
            workspace=4.0,                  # GiB de memoria de trabajo al construir
            min_batch=1,                    # límites del perfil dinámico
            opt_batch=1,
            max_batch=8,
            hardware_compatibility="none",  # o "ampere_plus"
            gpu_device=0,                   # dispositivo de construcción en un host multi-GPU
            verbose=False,
        )
  dynamic:
    - label: Engine con batch dinámico
      language: python
      code: >
        from libreyolo import LibreYOLO


        # El intermedio ONNX necesita el eje de batch dinámico para que el
        perfil

        # tenga algo a lo que enlazarse.

        LibreYOLO("LibreYOLO9t.pt").export(
            format="tensorrt",
            dynamic=True,
            min_batch=1,
            opt_batch=4,
            max_batch=8,
            half=True,
        )
  int8:
    - label: INT8 con datos de calibración
      language: python
      code: |
        from libreyolo import LibreYOLO

        LibreYOLO("LibreYOLO9t.pt").export(
            format="tensorrt",
            int8=True,
            data="coco128.yaml",   # obligatorio: no hay valor por defecto en este formato
            fraction=1.0,
        )
  run:
    - label: A través de LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t_fp16.engine")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: TensorRT a secas
      language: python
      code: >
        import json


        import tensorrt as trt


        path = "weights/LibreYOLO9t_fp16.engine"

        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))

        with open(path, "rb") as handle:
            engine = runtime.deserialize_cuda_engine(handle.read())

        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            print(engine.get_tensor_mode(name), name, engine.get_tensor_shape(name))

        # Los nombres de clase, la tarea y el tamaño de entrada viven en el
        sidecar, no en el engine.

        # La reserva de buffers, el preprocesado y el posprocesado corren de tu
        cuenta aquí.

        print(json.load(open(path + ".json"))["names"])
  support:
    - label: Comprobar una familia y una tarea antes de construir
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: cb90fc98ab735233
---

## Instalación

Tanto la construcción como la ejecución necesitan una GPU NVIDIA con un stack de
CUDA funcional. No hay respaldo por CPU para este formato.

<code-tabs name="install" />

El extra `tensorrt` fija `tensorrt-cu12` y `pycuda`, y el marcador descarta ambos
en macOS. En una Jetson, no uses ese extra: fija una build de CUDA 12 contra una
plataforma CUDA 13. Usa en su lugar el TensorRT que instala JetPack, como se
describe en [NVIDIA Jetson](/docs/export/jetson).

## Exportación

<code-tabs name="export" />

La exportación se ejecuta en dos pasos. El primero escribe un intermedio ONNX en
una ruta temporal, el segundo lo parsea y construye el engine, y después se elimina
el intermedio. `workspace` es memoria de trabajo en tiempo de construcción, en GiB;
un valor mayor deja al builder probar más kernels y no afecta a la memoria de
inferencia.

El sidecar de metadatos se escribe junto al engine como `<engine>.json` y registra
la precisión que la build consiguió realmente. Cuando la GPU no tiene FP16 rápido
ni INT8 rápido, el builder avisa y recurre a otra precisión, y el sidecar informa
de la precisión que salió, no de la que se pidió.

Con FP16, si hay un backbone ViT en el grafo se detecta y sus capas float se fijan
a FP32. Los backbones de tipo DINOv2 desbordan en FP16 y producen NaN, así que la
build activa `OBEY_PRECISION_CONSTRAINTS` e informa de `FP16 (FP32 ViT backbone)`.
La pasada no hace nada sobre backbones CNN.

### Batch dinámico

<code-tabs name="dynamic" />

`dynamic=True` añade un perfil de optimización que va de `min_batch` a `max_batch`,
optimizado en `opt_batch`, y registra esos tres valores en el sidecar. El perfil
solo se añade cuando el intermedio ONNX lleva realmente una dimensión de batch
dinámica; en caso contrario, la build registra que está usando optimización
estática y continúa.

### INT8

<code-tabs name="int8" />

INT8 usa el calibrador de entropía de TensorRT sobre un loader de calibración de
LibreYOLO, y `data` es obligatorio: este formato no tiene el respaldo de ocho
imágenes. La calibración necesita `cuda-python` o `pycuda` para el buffer del
dispositivo. La caché de calibración se indexa por un hash de los bytes del ONNX,
así que las escalas de un modelo nunca se reutilizan para otro que dé la casualidad
de escribir en la misma ruta de salida.

`half=True` e `int8=True` juntos avisan y construyen INT8, que mantiene un respaldo
FP16 para las capas que TensorRT no puede cuantizar.

## Ejecutar el artefacto

<code-tabs name="run" />

`LibreYOLO()` despacha según el sufijo `.engine`, lee del sidecar los nombres de
clase, la tarea y el esquema de pose, y devuelve el mismo objeto `Results` que el
checkpoint. Lanza un error de inmediato cuando no hay ningún dispositivo CUDA
presente.

El segundo snippet es la ruta del runtime a secas. La reserva de buffers de host y
de dispositivo, el preprocesado, la decodificación, el NMS y el reescalado de
coordenadas corren todos de tu cuenta, y el engine en sí no lleva nombres de clase,
así que el sidecar tiene que viajar con él.

## Restricciones

Un engine serializado queda atado a la arquitectura de GPU, al stack de drivers y a
la versión de TensorRT que lo construyó. Un engine construido en una estación de
trabajo no cargará en una arquitectura distinta, y por eso el paso de construcción
se ejecuta en la máquina de despliegue. `hardware_compatibility="ampere_plus"` cede
algo de rendimiento a cambio de portabilidad entre Ampere y posteriores. El valor
`"same_compute_capability"` se mapea a `NONE` y avisa: el engine está optimizado
solo para la GPU actual, y la exportación lo dice en lugar de afirmar una
portabilidad que no aplicó.

Solo se perfila el eje de batch. Una build con dimensiones espaciales dinámicas no
forma parte de este contrato, y por eso FCOS está bloqueado: necesita alto y ancho
con padding dinámico para conservar su transformación de aspecto de 800 por 1333.

Bloqueados antes del trazado: la segmentación de YOLO9, la segmentación de
RTMDet-Ins, la detección de SSD, Faster R-CNN y RetinaNet, y el matting de BiRefNet
o FeyNobg, donde TensorRT 10.16 llega al nodo ONNX compartido `DeformConv` y no
puede parsearlo porque `ModulatedDeformConv2d` no está en el registro de plugins.

Cuando una combinación no está ni validada ni bloqueada, la ruta del conversor está
disponible y el proyecto no ha registrado paridad de runtime en TensorRT para ella.
Eso es una afirmación sobre la evidencia, no sobre si la build tiene éxito.

Para la rejilla completa de familias y tareas, consulta
[la matriz de exportación](/docs/reference/export-matrix). Para una combinación
concreta:

<code-tabs name="support" />
