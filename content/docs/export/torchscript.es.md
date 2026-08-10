---
title: TorchScript
seo_title: Exportar a TorchScript desde LibreYOLO
description: >-
  Exporta un modelo LibreYOLO a TorchScript: un archivo .torchscript trazado con
  los metadatos de LibreYOLO dentro, cargable desde Python o desde libtorch.
lead: >-
  TorchScript es el formato de grafo serializado propio de PyTorch. LibreYOLO
  traza el modelo con torch.jit.trace y guarda el resultado junto a un archivo
  extra libreyolo_metadata.json, de modo que el archivo lleva dentro la familia,
  la tarea, los nombres de clase y el tamaño de entrada.
keywords:
  - exportar yolo torchscript
  - torch.jit.trace
  - torch.jit.load
  - despliegue libtorch
  - metadatos torchscript
  - extra_files
last_verified: 1.5.0
meta:
  - label: Flag
    value: export(format="torchscript")
    mono: true
  - label: Escribe
    value: Un archivo .torchscript con un archivo extra libreyolo_metadata.json
  - label: Extra
    value: Ninguno. TorchScript viene con PyTorch.
  - label: Se recarga con
    value: LibreYOLO("weights/LibreYOLO9t.torchscript")
    mono: true
  - label: Formas
    value: Fija. El grafo se traza con una sola forma de entrada.
  - label: Precisión
    value: 'FP32, FP16 (half=True). Sin INT8.'
verification: >-
  Leído de libreyolo/export/torchscript.py, libreyolo/export/exporter.py,
  libreyolo/export/support.py y libreyolo/backends/torchscript.py en la rama
  dev.
snippets:
  install:
    - label: Instalación
      language: bash
      code: |
        pip install libreyolo
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Escribe weights/LibreYOLO9t.torchscript
        path = model.export(format="torchscript")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format torchscript
    - label: Argumentos
      language: python
      code: >
        model.export(
            format="torchscript",
            imgsz=640,        # int, o (alto, ancho)
            batch=1,
            half=False,       # pesos y activaciones en FP16
            device=None,      # None traza en CPU para este formato
            output_path=None, # None escribe weights/<stem>.torchscript
        )


        # dynamic se acepta, pero el archivo siempre es un trazado de forma
        fija,

        # y los metadatos incrustados registran dynamic=False en cualquier caso.
  run:
    - label: A través de LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t.torchscript")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: PyTorch a secas
      language: python
      code: |
        import json

        import torch

        extra_files = {"libreyolo_metadata.json": ""}
        module = torch.jit.load(
            "weights/LibreYOLO9t.torchscript",
            map_location="cpu",
            _extra_files=extra_files,
        )
        module.eval()

        metadata = json.loads(extra_files["libreyolo_metadata.json"])
        print(metadata["model_family"], metadata["task"], metadata["imgsz"])

        # En esta vía, el preprocesado y el postprocesado corren de tu cuenta.
        with torch.no_grad():
            out = module(torch.zeros(1, 3, 640, 640))
        print(out.shape if torch.is_tensor(out) else [t.shape for t in out])
  support:
    - label: Comprobar una familia y tarea antes de exportar
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: 286a082969ccd604
---

## Instalación

<code-tabs name="install" />

TorchScript no necesita nada más allá de la instalación base, porque `torch.jit`
viene con PyTorch. Es el único destino de exportación sin dependencia opcional y
sin conversor externo, lo que lo convierte en una primera comprobación útil
cuando falla una cadena de herramientas más larga.

## Exportación

<code-tabs name="export" />

El trazado corre en CPU salvo que se indique un dispositivo, y el archivo se
escribe en `weights/` con el stem del checkpoint cuando se omite `output_path`.

La comprobación de retrazado que `torch.jit.trace` hace normalmente está
desactivada. Varios wrappers de exportación cachean anclas dependientes de la
forma durante su primer forward, así que un segundo trazado observa una ruta de
Python distinta aunque el grafo de forma fija registrado sea correcto. En su
lugar, los tests de paridad validan directamente el módulo guardado.

Los metadatos no viven en un sidecar. `torch.jit.save` guarda
`libreyolo_metadata.json` dentro del archivo, y `torch.jit.load` lo devuelve a
través de `_extra_files`.

## Ejecutar el artefacto

<code-tabs name="run" />

`LibreYOLO()` enruta por el sufijo `.torchscript` y devuelve el mismo objeto
`Results` que el checkpoint del que viene. Con `device="auto"`, el módulo se
mapea a CUDA cuando está disponible, luego a MPS y luego a CPU.

El segundo snippet es la vía para quien no tenga LibreYOLO instalado, y para el
despliegue en C++ con libtorch, donde el mismo archivo se carga con
`torch::jit::load`. Ahí el preprocesado, el decodificado, el NMS y el reescalado
de coordenadas corren de tu cuenta. El archivo extra de metadatos sigue siendo
legible, y es el único sitio donde existen los nombres de clase.

## Restricciones

El grafo es un trazado con una sola forma de entrada. `dynamic=True` se acepta
por simetría de interfaz pero no cambia nada, y los metadatos incrustados
informan `dynamic=False` para que un backend nunca asuma un eje que no puede
usar. Exporta un segundo archivo para una segunda resolución.

`half=True` convierte el modelo y la entrada del trazado a FP16. No hay vía
INT8: `int8=True` lanza `NotImplementedError` durante la validación.

El `imgsz` rectangular funciona para las familias YOLO9, HRNet, NAFNet y
Real-ESRGAN, y se rechaza para las familias con un contrato cuadrado fijo.

Cinco combinaciones se rechazan antes del trazado. Segmentación con YOLO9,
porque en LibreYOLO YOLO9 es solo detección. Segmentación con RTMDet-Ins, cuyo
decodificado de máscaras con kernel dinámico no tiene contrato de runtime
exportado. Detección con SSD, Faster R-CNN y RetinaNet, cuyos grafos de longitud
variable o de anclas dinámicas solo tienen evidencia de paridad a través del
contrato de ONNX Runtime.

Para la tabla completa de familias y tareas, consulta
[la matriz de exportación](/docs/reference/export-matrix). Para una combinación
concreta:

<code-tabs name="support" />
