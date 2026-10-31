---
title: Paddle
seo_title: Exportar a PaddlePaddle desde LibreYOLO
description: >-
  Convierte un detector de LibreYOLO en un modelo de inferencia de PaddlePaddle
  a través de X2Paddle: el toolchain fijado, los grafos estáticos FP32 con batch
  1 y la inferencia en CPU.
lead: >-
  Los modelos de inferencia de PaddlePaddle son un grafo model.pdmodel junto a
  un archivo de pesos model.pdiparams. LibreYOLO exporta un grafo ONNX estático
  con opset 15, lo convierte con X2Paddle y empaqueta el resultado con un
  metadata.yaml para que se cargue a través de la misma factoría que cualquier
  otro runtime.
keywords:
  - exportar yolo paddle
  - inferencia paddlepaddle
  - x2paddle
  - model.pdmodel
  - model.pdiparams
  - onnx opset 15
last_verified: 1.5.0
meta:
  - label: Flag
    value: export(format="paddle")
    mono: true
  - label: Escribe
    value: 'Un directorio con model.pdmodel, model.pdiparams y metadata.yaml'
  - label: Extra
    value: 'pip install "libreyolo[paddle]"'
    mono: true
  - label: Se recarga con
    value: 'LibreYOLO("weights/LibreYOLO9t_paddle", device="cpu")'
    mono: true
  - label: Backend
    value: libreyolo.backends.paddle.PaddleBackend
    mono: true
  - label: Formas
    value: 'Estáticas, batch 1, opset 15. Las tres se imponen.'
  - label: Precisión
    value: 'Solo FP32, solo CPU.'
  - label: Toolchain
    value: >-
      PaddlePaddle 2.6.2, X2Paddle 1.6.0, ONNX 1.17 o anterior, comprobados de
      forma exacta
verification: >-
  Leído de libreyolo/export/paddle.py, libreyolo/export/exporter.py,
  libreyolo/export/support.py, libreyolo/backends/paddle.py, docs/paddle.md y
  pyproject.toml en la rama dev.
snippets:
  install:
    - label: Instalación
      language: bash
      code: >
        # Python 3.10 a 3.12. WSL2 con Ubuntu 22.04 es la ruta validada en
        Windows.

        pip install "libreyolo[paddle]"
    - label: Confirmar las versiones fijadas
      language: bash
      code: >
        python -c "from importlib.metadata import version;
        print(version('paddlepaddle'), version('x2paddle'), version('onnx'))"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Escribe el directorio weights/LibreYOLO9t_paddle
        path = model.export(format="paddle")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format paddle
    - label: Argumentos
      language: python
      code: |
        model.export(
            format="paddle",
            imgsz=640,        # int; el lienzo cuadrado de esta familia
            batch=1,          # cualquier otro valor lanza ValueError
            dynamic=False,    # True lanza ValueError
            simplify=True,    # False lanza ValueError
            opset=15,         # cualquier otro valor lanza ValueError
            output_path=None, # None escribe weights/<stem>_paddle
        )
  run:
    - label: A través de LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t_paddle", device="cpu")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: CLI
      language: bash
      code: |
        libreyolo predict --model weights/LibreYOLO9t_paddle \
          --source https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg --device cpu --save
    - label: El backend directamente
      language: python
      code: >
        from libreyolo.backends.paddle import PaddleBackend


        # Lo que construye LibreYOLO() para un directorio Paddle. El mismo
        objeto

        # Results, sin enrutado de la factoría por medio.

        backend = PaddleBackend("weights/LibreYOLO9t_paddle", device="cpu")

        result = backend.predict("parkour.jpg")

        print(result.boxes.xyxy[:3])
    - label: Paddle a secas
      language: python
      code: |
        import numpy as np
        import paddle.inference as paddle_infer
        import yaml

        directory = "weights/LibreYOLO9t_paddle"
        config = paddle_infer.Config(
            f"{directory}/model.pdmodel", f"{directory}/model.pdiparams"
        )
        config.disable_gpu()
        config.disable_mkldnn()
        config.switch_ir_optim(False)

        predictor = paddle_infer.create_predictor(config)
        handle = predictor.get_input_handle(predictor.get_input_names()[0])
        handle.reshape([1, 3, 640, 640])
        handle.copy_from_cpu(np.zeros((1, 3, 640, 640), dtype=np.float32))
        predictor.run()
        for name in predictor.get_output_names():
            print(name, predictor.get_output_handle(name).copy_to_cpu().shape)

        meta = yaml.safe_load(open(f"{directory}/metadata.yaml"))
        print(meta["model_family"], meta["task"], meta["names"])

        # El preprocesado y el postprocesado corren de tu cuenta en esta ruta.
  support:
    - label: Comprobar una familia y una tarea antes de exportar
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: cdd8bf12286e2f53
---

## Instalación

<code-tabs name="install" />

El extra fija exactamente el stack que midió el trabajo de paridad: PaddlePaddle
2.6.2, X2Paddle 1.6.0 y ONNX 1.17 o anterior. Esas versiones fijadas se comprueban
en el momento de exportar, no solo al instalar, y una versión distinta lanza un
`ImportError` que nombra la esperada. Las versiones más nuevas de Paddle rechazan
partes del código estático que genera X2Paddle 1.6.0, así que fallar pronto es
mejor que producir un artefacto que nadie ha validado.

## Exportación

<code-tabs name="export" />

Cuatro argumentos están fijados, no simplemente predeterminados. `dynamic` debe
ser `False`, `batch` debe ser 1, `simplify` debe ser `True` para obtener un grafo
de conversión completamente estático, y `opset` debe ser 15, que es el techo que
acepta X2Paddle 1.6.0. Pasar cualquier otra cosa lanza un error antes del trazado.

Sobre el grafo intermedio corre una única normalización. ONNX define como uno la
dilatación omitida de un MaxPool, PyTorch escribe el atributo explícito de todo
unos y X2Paddle 1.6.0 lo rechaza, así que el exportador elimina ese valor por
defecto redundante y deja la operación especificada sin cambios.

El artefacto es un directorio: `model.pdmodel`, `model.pdiparams` y
`metadata.yaml`. El Python que X2Paddle genera durante la conversión no forma
parte de él.

## Ejecutar el artefacto

<code-tabs name="run" />

`LibreYOLO()` reconoce cualquier directorio que contenga a la vez `model.pdmodel`
y `model.pdiparams`, lee `metadata.yaml` y devuelve el mismo objeto `Results` que
el checkpoint. Un dispositivo que no sea `auto` o `cpu` lanza un error: este
backend es solo para CPU.

Lo que construye la factoría es `PaddleBackend`, exportado desde `libreyolo` e
importable como `libreyolo.backends.paddle.PaddleBackend`. Constrúyelo tú mismo
cuando quieras el backend sin el enrutado por sufijo de la factoría, por ejemplo
para pasar `task=` de forma explícita en un directorio cuyo `metadata.yaml` no has
escrito tú. Su `predict()` acepta las mismas fuentes y devuelve los mismos
resultados.

El snippet del runtime a secas refleja lo que configura el backend, y las tres
opciones desactivadas son deliberadas. El pipeline de fusión en CPU de Paddle 2.6
puede caerse mientras optimiza los grandes grafos de gather y scatter que se
emiten para la atención deformable, así que el grafo estático portable y sin
fusionar es aquel contra el que se midió la paridad. El preprocesado, el
decodificado, el NMS y el reescalado de coordenadas corren de tu cuenta en esa
ruta.

## Restricciones

Sin formas dinámicas, sin FP16, sin INT8, sin NMS embebido, sin runtime en GPU.

Las combinaciones validadas son la detección YOLO9, la detección YOLO9-E2E y
YOLO9-P2, la detección, la pose y la segmentación EC, la detección RT-DETRv4,
D-FINE, DEIM y DEIMv2, y la detección y la pose YOLO-NAS. Cada una está cubierta
por la conversión, una recarga en el runtime de CPU, la paridad de salidas en
crudo y resultados públicos reproducidos.

Bloqueadas, con el motivo registrado por combinación:

| Combinación | Motivo |
|---|---|
| RF-DETR, todas las tareas | Necesita ONNX opset 17 y GridSample; X2Paddle 1.6.0 acepta opset 15 o inferior y no tiene mapper de GridSample |
| Detección RT-DETR y RT-DETRv2 | Los grafos entrenados necesitan GridSample con opset 16 o superior |
| Segmentación D-FINE | Convierte y recarga, pero el error RMS relativo de los logits de máscara es del 3.52% y el IoU mínimo de máscaras emparejadas es 0.582 |
| Segmentación YOLO9 | YOLO9 es solo detección en LibreYOLO |
| Segmentación RTMDet-Ins | El decodificado de máscaras con kernel dinámico no tiene contrato de runtime exportado |

Cualquier cosa que no aparezca como validada o bloqueada se rechaza con la nota de
que no se ha validado a través de la ruta de conversión de ONNX a Paddle.

Para la rejilla completa de familias y tareas, consulta
[la matriz de exportación](/docs/reference/export-matrix). Para una sola
combinación:

<code-tabs name="support" />
