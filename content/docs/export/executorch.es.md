---
title: ExecuTorch
seo_title: "Exportar a ExecuTorch desde LibreYOLO"
description: "Exporta un modelo LibreYOLO a un programa .pte de ExecuTorch con delegación a XNNPACK: forma fija, batch 1, FP32 y el sidecar de metadatos que necesita."
lead: "ExecuTorch ejecuta programas de PyTorch en dispositivos edge. LibreYOLO captura el modelo con torch.export en modo estricto, hace el lowering a XNNPACK y escribe el programa .pte junto con un sidecar de metadatos JSON como una sola unidad."
keywords:
  - exportar yolo executorch
  - programa .pte
  - xnnpack partitioner
  - torch.export strict
  - executorch runtime
  - inferencia pytorch en el edge
last_verified: "1.5.0"
meta:
  - label: Flag
    value: 'export(format="executorch")'
    mono: true
  - label: Escribe
    value: "Un programa .pte más un sidecar de metadatos .pte.json"
  - label: Extra
    value: 'pip install "libreyolo[executorch]"'
    mono: true
  - label: Se recarga con
    value: 'LibreYOLO("weights/LibreYOLO9t.pte")'
    mono: true
  - label: Formas
    value: "Fijas. dynamic=True y batch != 1 se rechazan."
  - label: Precisión
    value: "Solo FP32. half=True e int8=True se rechazan."
  - label: Delegado
    value: "XNNPACK, CPU. delegate='xnnpack' es el único valor aceptado."
verification: "Leído de libreyolo/export/executorch.py, libreyolo/export/exporter.py, libreyolo/export/support.py, libreyolo/backends/executorch.py y pyproject.toml en la rama dev."
snippets:
  install:
    - label: Instalación
      language: bash
      code: |
        # Fuera de libreyolo[all] a propósito: ExecuTorch limita con qué
        # versión de Torch puede emparejarse.
        pip install "libreyolo[executorch]"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Escribe weights/LibreYOLO9t.pte y weights/LibreYOLO9t.pte.json
        path = model.export(format="executorch", imgsz=640)
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format executorch --imgsz 640
    - label: Argumentos
      language: python
      code: |
        model.export(
            format="executorch",
            imgsz=640,             # int, o (alto, ancho)
            batch=1,               # cualquier otro valor lanza ValueError
            dynamic=False,         # True lanza ValueError
            delegate="xnnpack",    # el único valor aceptado
            device="cpu",          # cualquier otro dispositivo lanza ValueError
            output_path=None,      # None escribe weights/<stem>.pte
        )
  run:
    - label: A través de LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t.pte")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: Runtime de ExecuTorch directo
      language: python
      code: |
        import json
        from pathlib import Path

        import torch
        from executorch.runtime import Runtime

        runtime = Runtime.get()
        print(runtime.backend_registry.is_available("XnnpackBackend"))

        program = runtime.load_program(Path("weights/LibreYOLO9t.pte").read_bytes())
        method = program.load_method("forward")

        # En esta vía el preprocesado y el postprocesado son cosa tuya.
        outputs = method.execute((torch.zeros(1, 3, 640, 640),))
        print([tensor.shape for tensor in outputs])

        meta = json.load(open("weights/LibreYOLO9t.pte.json"))
        print(meta["model_family"], meta["task"], meta["executorch_delegate"])
  support:
    - label: Comprobar una familia y una tarea antes de exportar
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
---

## Instalación

<code-tabs name="install" />

Este extra queda deliberadamente fuera de `libreyolo[all]`, porque ExecuTorch fija
con qué versión de Torch funciona e instalarlo arrastraría todo el entorno a esa
pareja. Instálalo en un entorno que estés dispuesto a restringir.

En Windows, el paso de lowering llama al ejecutable `flatc` que viene con
ExecuTorch. Si no está en el `PATH`, la exportación lanza un `RuntimeError` que lo
indica, y la solución es ejecutarla desde una Developer PowerShell de Visual Studio 2022.

## Exportación

<code-tabs name="export" />

La captura es `torch.export.export(..., strict=True)`, que es una captura de grafo
real con guards en lugar de un trazado grabado. Las lecturas de escalares en el host
y el control de flujo dependiente de los datos se rechazan en vez de quedar fijados
en silencio, así que aquí fallan varias familias que sí se trazan bien en otros
formatos; los motivos están registrados por combinación en la matriz de soporte.

El lowering ejecuta `to_edge_transform_and_lower` con el partitioner de XNNPACK. Si
el resultado no contiene ninguna partición delegada, la exportación lanza un error en
lugar de etiquetar como XNNPACK un programa que solo usa kernels portables.

El programa y el sidecar se confirman juntos. Ambos se preparan, ambos se
intercambian y un fallo revierte a lo que hubiera antes, así que nunca llega al disco
una pareja a medias.

## Ejecutar el artefacto

<code-tabs name="run" />

`LibreYOLO()` despacha según el sufijo `.pte` y devuelve el mismo objeto `Results`
que el checkpoint. El sidecar es obligatorio al cargar: sin
`<program>.pte.json` el backend lanza `FileNotFoundError`, porque el programa no
lleva por sí mismo nombres de clase, tarea ni tamaño de entrada. El backend también
comprueba que el runtime instalado ofrezca `XnnpackBackend` antes de cargar, y lee el
programa desde bytes en lugar de mapear el archivo, lo que evita mantener un bloqueo
de archivo de Windows durante toda la vida del backend.

El segundo snippet es la vía del runtime directo. Allí el preprocesado, el
decodificado, el NMS y el reescalado de coordenadas pasan a ser cosa tuya.

## Restricciones

Batch 1, forma fija, FP32, CPU. Tanto `batch != 1` como `dynamic=True` lanzan
`ValueError` antes de que la exportación modifique nada, `half=True` e `int8=True` se
rechazan durante la validación, y se rechaza cualquier dispositivo que no sea la CPU.

`delegate` acepta `"xnnpack"` y nada más en esta versión.

Las exportaciones de clasificación llevan dos claves de metadatos extra, `crop_pct` e
`interpolation`, para que el runtime pueda reproducir la política de redimensionado y
recorte central de la familia.

Las entradas bloqueadas nombran el fallo concreto en lugar de una categoría. La
detección y la segmentación de D-FINE llegan a una lectura de `ContextVar` no
soportada en la atención deformable bajo captura estricta, y forzar la vía manual de
grid-sample serializa pero después falla en tiempo de ejecución por un orden de
dimensiones inválido en un tensor delegado. DEIM y DEIMv2 se capturan, pasan el
lowering y se serializan, y luego fallan durante la ejecución. La segmentación
semántica de EoMT falla en una expresión simbólica dependiente de los datos en la ruta
de las máscaras. El matting de BiRefNet se captura a 1024 por 1024 pero no tiene
variante out para `torchvision::deform_conv2d`. La restauración de SwinIR se recarga y
luego falla en `aten::alias_copy.out` por órdenes de dimensiones que no coinciden.

Para la rejilla completa de familias y tareas, consulta
[la matriz de exportación](/docs/reference/export-matrix). Para una sola combinación:

<code-tabs name="support" />
