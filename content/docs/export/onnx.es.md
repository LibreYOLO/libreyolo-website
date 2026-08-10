---
title: ONNX
seo_title: "Exportar a ONNX desde LibreYOLO"
description: "Exporta un modelo LibreYOLO a ONNX: el opset que LibreYOLO elige por familia, los ejes dinámicos, el NMS embebido, INT8 y cómo se recarga el grafo."
lead: "ONNX es un formato de grafo portable. LibreYOLO traza el modelo con torch.onnx.export, opcionalmente simplifica el grafo y escribe la familia, la tarea, los nombres de clase y el tamaño de entrada en los metadatos del propio archivo, de modo que cualquier backend de LibreYOLO pueda reconstruir el postprocesado."
keywords:
  - exportar yolo onnx
  - onnxruntime
  - torch.onnx.export
  - onnx opset
  - ejes dinámicos onnx
  - nms embebido onnx
  - onnx int8 qdq
  - onnx metadata_props
last_verified: "1.5.0"
meta:
  - label: Flag
    value: 'export(format="onnx")'
    mono: true
  - label: Escribe
    value: "Un archivo .onnx, con los metadatos embebidos en el grafo"
  - label: Extra
    value: 'pip install "libreyolo[onnx]"'
    mono: true
  - label: Se recarga con
    value: 'LibreYOLO("weights/LibreYOLO9t.onnx")'
    mono: true
  - label: Formas
    value: "Batch dinámico por defecto en Python; excepciones por tarea más abajo"
  - label: Precisión
    value: "FP32, FP16 (half=True), INT8 (int8=True, detección YOLO9)"
verification: "Leído de libreyolo/export/onnx.py, libreyolo/export/exporter.py, libreyolo/export/support.py, libreyolo/backends/onnx.py y libreyolo/cli/commands/export.py en la rama dev."
snippets:
  install:
    - label: Instalación
      language: bash
      code: |
        pip install "libreyolo[onnx]"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Escribe weights/LibreYOLO9t.onnx
        path = model.export(format="onnx")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format onnx
    - label: Argumentos
      language: python
      code: |
        model.export(
            format="onnx",
            imgsz=640,        # int, o (alto, ancho)
            batch=1,
            dynamic=True,     # valor por defecto en Python; en la CLI es False
            simplify=True,    # pasa onnxsim sobre el grafo
            opset=None,       # None elige 13, o 17 para las familias estilo DETR
            half=False,       # pesos y activaciones en FP16
            int8=False,       # INT8 QDQ, solo detección YOLO9
            data=None,        # data.yaml de calibración, solo INT8
            device=None,      # dispositivo de trazado; None usa el del modelo
            output_path=None, # None escribe weights/<stem>.onnx
        )
  nms:
    - label: Incrustar NMS en el grafo
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Solo detección YOLO9, batch 1. dynamic se fuerza a False.
        LibreYOLO("LibreYOLO9t.pt").export(
            format="onnx",
            nms=True,
            conf=0.25,
            iou=0.45,
            max_det=300,
        )
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format onnx --nms \
          --conf 0.25 --iou 0.45 --max-det 300
  int8:
    - label: INT8 con datos de calibración
      language: python
      code: |
        from libreyolo import LibreYOLO

        LibreYOLO("LibreYOLO9t.pt").export(
            format="onnx",
            int8=True,
            data="coco128.yaml",   # unos cientos de imágenes representativas
            fraction=1.0,
        )
  run:
    - label: A través de LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t.onnx")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: ONNX Runtime puro
      language: python
      code: |
        import numpy as np
        import onnx
        import onnxruntime as ort

        session = ort.InferenceSession(
            "weights/LibreYOLO9t.onnx",
            providers=["CPUExecutionProvider"],
        )

        # En esta vía, el preprocesado y el postprocesado corren de tu cuenta.
        batch = np.zeros((1, 3, 640, 640), dtype=np.float32)
        outputs = session.run(None, {session.get_inputs()[0].name: batch})
        print([out.shape for out in outputs])

        # El grafo lleva la familia, la tarea, los nombres de clase y el tamaño de entrada.
        meta = {p.key: p.value for p in onnx.load("weights/LibreYOLO9t.onnx").metadata_props}
        print(meta["model_family"], meta["task"], meta["imgsz"])
  support:
    - label: Comprobar una familia y tarea antes de exportar
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
---

## Instalación

<code-tabs name="install" />

El extra instala `onnx`, `onnxsim` y `onnxruntime`. Con `onnx` solo basta para
escribir el archivo; `onnxsim` ejecuta la pasada de simplificación y `onnxruntime`
ejecuta el artefacto y realiza la calibración INT8.

## Exportación

<code-tabs name="export" />

Sin `output_path`, el archivo se guarda en `weights/` con el stem del checkpoint,
añadiendo `_fp16` o `_int8` cuando se pidió esa precisión.

`dynamic` es `True` por defecto en Python y `False` en la CLI. Cuando está
activo, el eje de batch se vuelve simbólico y algunas tareas se abren más: la
segmentación semántica abre también el alto y el ancho de la máscara, la
restauración con Real-ESRGAN abre los ejes espaciales, y los detectores de dos
etapas mantienen dinámicos el alto y el ancho de origen porque su
redimensionado ocurre dentro del grafo.

`opset` se elige por familia cuando se omite. Las familias estilo DETR (`detr`,
`deformable_detr`, `dinodetr`, `dfine`, `deim`, `deimv2`, `ec`, `lwdetr`,
`rfdetr`, `rtdetr`, `rtdetrv2`, `rtdetrv4`) más `deit`, `midas` y `moge2`
reciben el opset 17, que es donde baja `aten::scaled_dot_product`. Todo lo demás
recibe 13. El matting se eleva a 19 en cualquier caso, porque el decodificador de
BiRefNet necesita el operador `DeformConv`, que ONNX define a partir del opset 19.

`simplify=True` ejecuta `onnxsim` y conserva el grafo original si la pasada
falla, así que un error de simplificación es un aviso y no un fallo de
exportación. En macOS arm64 con `onnx` 1.22 o más reciente y `onnxsim` 0.6.5 o
más antiguo, la pasada se omite por completo, porque esa combinación puede
abortar el proceso de Python.

### NMS embebido

<code-tabs name="nms" />

`nms=True` es solo para detección YOLO9 y requiere batch 1; pedirlo con
`dynamic=True` registra un aviso y desactiva dynamic. El grafo pasa entonces a
tener dos salidas: `output`, con forma `(batch, max_det, 6)`, y `raw`, el tensor
del detector sin decodificar que el propio backend de LibreYOLO usa para que el
postprocesado sea idéntico al de la vía PyTorch.

### DeepStream

`deepstream=True` es una opción exclusiva de ONNX. Exporta el grafo en la
disposición que espera el parser de NVIDIA DeepStream y escribe dos archivos
adjuntos a su lado, `config_infer_primary_<stem>.txt` y `<stem>_labels.txt`,
para que el artefacto encaje en un pipeline sin configuración escrita a mano.

Es mutuamente excluyente con `nms=True`, y pedir ambos lanza un `ValueError`:
DeepStream ejecuta la supresión en su propia etapa de clustering. Pasarlo a
cualquier formato distinto de ONNX también lanza un error. Consulta
[DeepStream](/docs/export/deepstream) para la tabla de familias y tareas
soportadas y la compilación del parser.

### INT8

<code-tabs name="int8" />

`int8=True` ejecuta la cuantización estática de ONNX Runtime y escribe un grafo
QDQ con entradas y salidas en float32. Solo se cuantizan los nodos `Conv` y
`Gemm`. Dejar en float32 la decodificación de la cabeza de detección es
deliberado: esa concatenación mezcla coordenadas de box a escala de píxel con
puntuaciones de clase en el rango de 0 a 1, y una única escala de activación por
tensor dominada por la magnitud de los boxes llevaría todas las puntuaciones a
cero.

Este flag actualmente aplica solo a detección YOLO9, y cualquier otra cosa lanza
`NotImplementedError` en la comprobación previa. Omitir `data` recurre a
`coco8.yaml` con un aviso; ocho imágenes no son un conjunto de calibración
representativo. Un modelo que ya fue cuantizado en PyTorch sigue una ruta
distinta, descrita en [Cuantización](/docs/export/quantization).

## Ejecutar el artefacto

<code-tabs name="run" />

`LibreYOLO()` despacha por el sufijo `.onnx` y devuelve el mismo objeto
`Results` que un checkpoint `.pt`, porque los nombres de clase, la tarea, el
tamaño de entrada y el esquema de pose se escribieron en los `metadata_props`
del grafo en el momento de la exportación. Con `device="auto"` la sesión toma
`CUDAExecutionProvider` cuando ONNX Runtime lo reporta y recurre a la CPU en
caso contrario.

El segundo snippet es para lectores sin LibreYOLO instalado. El preprocesado, la
decodificación, el NMS y el reescalado de coordenadas corren de tu cuenta en esa
vía; el bloque de metadatos sigue ahí para leerlo.

## Restricciones

Los nombres de los tensores de salida son fijos por tarea, y son lo que el
consumidor sin metadatos tiene que igualar:

| Tarea | Nombres de salida |
|---|---|
| Detección, cabezas de rejilla y de anchors | `output` |
| Detección, estilo DETR | `pred_logits`, `pred_boxes` |
| Detección, RF-DETR | `dets`, `labels` |
| Clasificación | `output` |
| Segmentación semántica | `semantic_logits` |
| Profundidad | `depth` |
| Normales de superficie | `normal` |
| Bordes | `edges` |
| Restauración | `restored` |
| Matting | `matte` |
| Mirada | `yaw_logits`, `pitch_logits` |

RF-DETR es también la única familia cuyo tensor de entrada se llama `input` en
lugar de `images`.

Varias tareas llevan en esta versión un contrato de resolución fija en tiempo de
ejecución. Profundidad, normales de superficie y bordes rechazan `batch != 1` y
fuerzan `dynamic=False`. El matting fuerza el cuadrado nativo de 1024, porque
las tablas de posición relativa del Swin de BiRefNet están ligadas a su
resolución. La restauración fuerza un lienzo fijo para todas las familias salvo
Real-ESRGAN, cuyo generador es totalmente convolucional.

Un `imgsz` rectangular funciona para las familias YOLO9, HRNet, NAFNet y
Real-ESRGAN. Las familias con contrato de cuadrado fijo (`clip`,
`deformable_detr`, `detr`, `dinodetr`, `dfine`, `deim`, `deimv2`, `ec`,
`lwdetr`, `moge2`, `rtdetr`, `rtdetrv2`, `rtdetrv4`, `rfdetr`, `siglip2`,
`ssd`) lo rechazan de plano.

Dos combinaciones se rechazan antes del trazado: la segmentación YOLO9, porque
YOLO9 es solo detección en LibreYOLO, y la segmentación RTMDet-Ins, cuya
decodificación de máscaras con kernels dinámicos no tiene contrato de runtime
exportado.

Para la tabla completa de familias y tareas, consulta
[la matriz de exportación](/docs/reference/export-matrix). Para una combinación
concreta, pregunta directamente a la librería:

<code-tabs name="support" />
