---
title: Core AI
seo_title: "Exportar a Apple Core AI desde LibreYOLO"
description: "Exporta un modelo LibreYOLO a un asset .aimodel de Apple Core AI: solo macOS, lienzo fijo, FP32 y el contrato de orden de salidas con nombre que los consumidores deben respetar."
lead: "Core AI es el stack de inferencia en dispositivo de Apple. LibreYOLO captura el modelo con torch.export, lo baja a través del conversor de Core AI y escribe un asset .aimodel que lleva los metadatos del modelo y los nombres de las salidas exportadas."
keywords:
  - exportar libreyolo core ai
  - aimodel
  - coreai-torch
  - torch.export apple
  - inferencia en dispositivo apple
  - coreai_output_names
last_verified: "1.5.0"
meta:
  - label: Flag
    value: 'export(format="coreai")'
    mono: true
  - label: Escribe
    value: "Un asset .aimodel con los metadatos adjuntos"
  - label: Extra
    value: 'pip install "libreyolo[coreai]"'
    mono: true
  - label: Se recarga
    value: "No con LibreYOLO. Los consumidores usan el runtime de Core AI directamente."
  - label: Formas
    value: "Lienzo fijo. dynamic=True lanza NotImplementedError."
  - label: Precisión
    value: "Solo FP32. half=True e int8=True se rechazan."
  - label: Requiere
    value: "macOS. La toolchain ni convierte ni ejecuta en otro sistema, y coreai-torch fija torch a 2.11.x."
verification: "Leído de libreyolo/export/coreai.py, libreyolo/export/coreai_compat.py, libreyolo/export/exporter.py, libreyolo/export/support.py y pyproject.toml en la rama dev."
snippets:
  install:
    - label: Instalación, en macOS
      language: bash
      code: |
        # Deliberadamente fuera de todos los extras agregados: coreai-torch fija
        # torch a 2.11.x y arrastraría todo el entorno a esa versión.
        pip install "libreyolo[coreai]"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Escribe weights/LibreYOLO9t.aimodel
        path = model.export(format="coreai", imgsz=640)
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format coreai --imgsz 640
    - label: Argumentos
      language: python
      code: |
        model.export(
            format="coreai",
            imgsz=640,        # int, o (alto, ancho); este es el lienzo de ejecución
            batch=1,
            output_path=None, # None escribe weights/<stem>.aimodel
        )

        # dynamic=True lanza NotImplementedError.
        # half=True e int8=True se rechazan durante la validación.
  outputs:
    - label: Lee el orden de las salidas antes de conectar un consumidor
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")
        model.export(format="coreai", imgsz=640)

        # Los metadatos del asset registran los nombres de las salidas exportadas,
        # en orden de grafo, bajo "coreai_output_names". Mapea por nombre el
        # diccionario que devuelve Core AI usando esa lista; nunca lo emparejes
        # por posición con la tupla del modo eager.
  support:
    - label: Comprobar una familia y una tarea antes de exportar
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
---

## Instalación

Este formato es solo para macOS. El requisito `coreai-torch` lleva un marcador
`sys_platform == 'darwin'`, y la toolchain ni convierte ni ejecuta en ningún
otro sistema.

<code-tabs name="install" />

El extra queda fuera de todos los extras agregados, incluido `libreyolo[all]`,
porque `coreai-torch` fija torch a la serie 2.11. Instálalo en un entorno que
estés dispuesto a restringir a ese par.

## Exportación

<code-tabs name="export" />

La captura es `torch.export`, una captura de grafo real con guards, y no un único
trazado grabado. Eso es más estricto que la ruta de Core ML: las lecturas de
escalares en el host y el control de flujo dependiente de datos se rechazan en
lugar de quedar horneados en silencio, y por eso hay unas cuantas familias
bloqueadas aquí con un fallo de captura registrado.

Tres pasos de preparación se ejecutan dentro de un ámbito que restaura el modelo
vivo de quien llama tanto si la exportación tiene éxito como si falla. Las
familias derivadas de Darknet ven su batch normalization de inferencia plegada
exactamente en las convoluciones precedentes, porque Core AI 0.4.1 no preserva la
fórmula de Darknet con el epsilon después de la raíz cuadrada. Las familias de
grid y de anchors ven sus anchors congelados para el lienzo fijo. RF-DETR ve su
position embedding rehorneado para el lienzo solicitado volviendo a ejecutar la
propia ruta de horneado del modelo, porque el conversor no tiene lowering para
`aten._upsample_bicubic2d_aa`.

El lowering incorpora a la tabla de descomposiciones la descomposición de
referencia de PyTorch para `aten.grid_sampler_2d`, ya que el conversor de Core AI
no tiene lowering para el sampler de deformable attention que usan las familias
DETR.

Los assets declaran un SO mínimo de v27, que es el único valor que ofrece la
toolchain. Eso condiciona el despliegue, no la conversión: la conversión y la
ejecución desde Python funcionan en versiones anteriores de macOS gracias al
runtime que viene dentro del wheel, pero los números difieren entre versiones del
SO, así que la paridad registrada se mide en macOS 27.

## Ejecutar el artefacto

No hay ninguna entrada de Core AI en `libreyolo/backends`, así que `LibreYOLO()`
no carga un `.aimodel`. Los consumidores usan el runtime de Core AI directamente,
y el preprocesado, el decodificado, el NMS y el reescalado de coordenadas corren
de su cuenta. Una fila validada en la matriz de soporte afirma que el grafo
exportado calcula los mismos números que la referencia, no que `predict` vaya a
ejecutarlo.

Lo único que un consumidor no puede volver a deducir es el orden de las salidas:

<code-tabs name="outputs" />

Core AI devuelve un diccionario con nombres cuyo orden de claves no coincide ni
con el orden de la tupla del forward en modo eager ni con nada adivinable. Los
nombres exportados se escriben en los metadatos del asset como
`coreai_output_names` justo por este motivo. Mapea por nombre.

## Restricciones

Lienzo fijo, FP32, batch tal y como se exportó. `dynamic=True` lanza
`NotImplementedError`, y `half=True` e `int8=True` se rechazan durante la
validación.

La cobertura es amplia por el lado de la conversión. Las combinaciones validadas
incluyen las familias YOLO9, YOLOX, YOLO7, los cuatro detectores de la era
Darknet, YOLO-NAS, PicoDet, RTMDet, RT-DETR, RT-DETRv2, RT-DETRv4, D-FINE, DEIM,
DEIMv2, EC y la detección RF-DETR; las cuatro familias de clasificación CNN más
CLIP y SigLIP2 con clases congeladas; Depth Anything V2 y ZipDepth; la
restauración con NAFNet y Real-ESRGAN; la segmentación semántica con PIDNet y
LingBotVision; y la detección de puntos FOMO. Cada una lleva su propio contexto
registrado, que imprime `libreyolo formats`.

Bloqueadas, con el motivo registrado por combinación:

| Combinación | Motivo |
|---|---|
| Segmentación semántica EoMT | La captura estricta falla con `GuardOnDataDependentSymNode`: algo en la ruta de máscaras lee un valor de un tensor y ramifica según él |
| Segmentación semántica SegFormer | La ruta de captura no se ha evaluado, y sus pesos publicados son no comerciales sea cual sea el formato |
| Mirada L2CS | El modelo en sí solo soporta ONNX, TorchScript, ExecuTorch, TensorRT y OpenVINO, lo cual es una decisión del lado del modelo |
| Profundidad Depth Anything 3 | La familia rechaza la exportación para todos los formatos |

RF-DETR lleva una advertencia que conviene leer antes de comparar artefactos. Su
paridad se registra contra el grafo que prepara el propio exportador de Core AI, y
no contra ONNX, y con un lienzo de 640 el artefacto ONNX de RF-DETR discrepa de
ese grafo preparado. El rehorneado de Core AI preserva el redimensionado con
antialiasing que hace el modelo en modo eager, mientras que la ruta de ONNX
desactiva el antialiasing. Por tanto ONNX no es una referencia válida para esa
familia con un lienzo que no sea el nativo.

Para el formato anterior de Apple, consulta [Core ML](/docs/export/coreml). Para
la cuadrícula completa de familias y tareas, consulta [la matriz de
exportación](/docs/reference/export-matrix). Para una sola combinación:

<code-tabs name="support" />
