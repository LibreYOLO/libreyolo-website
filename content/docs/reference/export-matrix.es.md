---
title: Matriz completa de exportación
seo_title: Matriz de soporte de exportación de LibreYOLO y sus reglas
description: >-
  Cómo decide LibreYOLO si una combinación de familia, tarea y formato se puede
  exportar: los doce formatos, los tres niveles, las reglas de respaldo y los
  umbrales de paridad.
lead: >-
  El soporte de exportación es una consulta sobre la terna (familia, tarea,
  formato). Esta página describe la forma de esa matriz, las reglas que rellenan
  las celdas que ninguna entrada explícita cubre, y cómo consultarla para la
  combinación que te interesa.
keywords:
  - soporte exportación libreyolo
  - matriz de exportación
  - onnx tensorrt openvino tflite
  - comando libreyolo formats
  - umbral de paridad exportación
  - NotImplementedError exportar
last_verified: 1.5.0
verification: >-
  Formatos, niveles, orden de respaldo, bloqueos por tarea y por familia y
  bloqueos de NCNN leídos de libreyolo/export/support.py; alias y argumentos
  compartidos de libreyolo/export/exporter.py; definiciones de niveles de
  docs/adr/0011-export-support-tiers.md; umbrales de paridad de
  docs/export_support.md, todo en la v1.5.0. Las celdas de cada combinación no
  se transcriben aquí; consúltalas con el snippet de abajo.
snippets:
  usage:
    - label: 'Consultar la matriz, sin necesidad de modelo'
      language: python
      code: |
        from libreyolo.export.support import (
            EXPORT_FORMATS,
            get_support,
            validated_alternatives,
        )

        print(EXPORT_FORMATS)

        entry = get_support("yolo9", "detect", "onnx")
        print(entry.tier, entry.since)
        print(entry.constraint)

        print(validated_alternatives("yolo9", "detect"))
    - label: CLI
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
        libreyolo formats --family yolo9 --task detect --json
  export:
    - label: 'Exportar, y leer un rechazo'
      language: python
      code: |
        from libreyolo import LibreYOLO
        from libreyolo.export.support import get_support

        model = LibreYOLO("LibreYOLO9t.pt")
        print(model.export(format="onnx"))

        # Compruébalo antes de llamar: una combinación bloqueada falla en el
        # preflight y el mensaje lleva este motivo.
        blocked = get_support("domedetr", "detect", "onnx")
        print(blocked.tier)
        print(blocked.reason)
source_hash: 83de3289634888c6
---

## Forma de la matriz

La matriz se indexa por `(family, task, format)`. Las claves de familia son los
nombres canónicos del registro de modelos, las claves de tarea vienen de
`libreyolo.tasks.TASKS`, y hay doce formatos:

`onnx`, `torchscript`, `executorch`, `tensorrt`, `openvino`, `paddle`, `mnn`,
`rknn`, `ncnn`, `tflite`, `coreml`, `coreai`.

`model.export(format=...)` acepta además dos alias: `engine` para `tensorrt`, y
`litert` para `tflite`, que es el nombre actual de TensorFlow Lite. El formato y
el sufijo `.tflite` no cambian.

<code-tabs name="usage" />

Como una celda es función de tres claves, la rejilla completa es grande y cambia
en cada versión. Se genera en lugar de escribirse a mano, y vive en
`docs/export_support.md` en el repositorio de la biblioteca. Consulta la matriz
desde Python o desde la CLI en vez de leer una copia.

## Los tres niveles

| Nivel | Significado |
|---|---|
| `validated` | La paridad numérica está cubierta en CI o en una ejecución nocturna documentada |
| `available` | La conversión está implementada, pero no se ha registrado evidencia de paridad numérica en runtime |
| `blocked` | El preflight lanza `NotImplementedError` con un motivo antes del trazado |

Las combinaciones validated y available continúan sin necesidad de una
confirmación ni de un aviso genérico. Su evidencia registrada y sus
restricciones siguen visibles en la documentación generada. Una combinación
blocked falla antes de las comprobaciones de dependencias, de la carga de
calibración, del trazado o de la creación de artefactos.

Añadir una entrada validated exige un test de paridad y un campo `since`.

Un `SupportEntry` lleva cuatro campos: `tier`, una cadena `reason`, la versión
`since`, y una cadena `constraint`. La restricción es la parte que importa a la
hora de integrar: una marca de verificación solo aplica bajo las condiciones que
nombra, que normalmente son un lienzo de entrada fijo, batch 1, FP32 y una
versión concreta del runtime.

## Cómo se decide una celda

`get_support(family, task, fmt)` resuelve en este orden. Gana la primera regla
que coincide.

1. Una tarea desconocida, o un formato fuera de los doce, devuelve `blocked`.
2. Una entrada explícita `(family, task, format)` devuelve lo registrado.
3. Un bloqueo a nivel de familia devuelve `blocked` con el motivo de esa familia.
4. Un bloqueo a nivel de tarea devuelve `blocked` con el motivo de esa tarea.
5. Para `ncnn`, una familia en la lista de bloqueo de NCNN devuelve `blocked`.
6. `mnn` devuelve `blocked`: no hay contrato de runtime para esa familia y tarea.
7. `rknn` devuelve `blocked`. RKNN en esta versión se limita a las variantes de detección exactas probadas en el simulador: YOLO9-t, YOLO9-E2E-t, YOLO-NAS-s y PicoDet-s en RK3588.
8. `tensorrt` y `openvino` devuelven `available`: la ruta del conversor existe pero no se ha registrado paridad en runtime para esa familia y tarea.
9. `tflite`, `paddle`, `coreai` y `coreml` devuelven `blocked`, cada uno con su propio motivo.
10. Todo lo demás devuelve `available`: la conversión está implementada, la paridad numérica en runtime no está registrada.

La asimetría de los pasos 8 a 10 es deliberada. TensorRT y OpenVINO convierten
de forma genérica desde ONNX, así que una combinación no listada merece un
intento. TFLite, Paddle, Core AI y CoreML necesitan cada uno una ruta por
familia, así que una combinación no listada es un rechazo y no una invitación.

## Tareas bloqueadas

Estas tareas están bloqueadas para cualquier familia que no tenga una entrada
explícita.

| Tarea | Motivo |
|---|---|
| `ocr` | Dos redes con recorte dinámico por región no encajan en el contrato de exportación de grafo único |
| `point` | La familia no está conectada al contrato compartido de heatmap de puntos y decodificación de picos en el backend |
| `semantic` | La familia no está conectada al contrato compartido de logits densos y argmax en el backend |
| `mesh` | Las salidas del grafo de malla corporal, sus metadatos y su contrato de runtime no están definidos |
| `normal` | La familia no está conectada al contrato de normales unitarias densas con lienzo fijo y renormalización en el backend |
| `panoptic` | La exportación panóptica no tiene contrato de runtime en el backend |
| `gaze` | La familia no está conectada al contrato compartido de logits de dos cabezas y decodificación por esperanza en el backend |

Una entrada explícita sobreescribe estos bloqueos, que es como, por ejemplo, una
familia semántica ya conectada sí se exporta.

## Familias bloqueadas

| Familia | Bloqueada para |
|---|---|
| `depth_anything3` | Todos los formatos; su grafo de profundidad no está en el contrato de runtime exportado |
| `domedetr` | Todos los formatos. PAQI fija el número de queries por imagen, así que un grafo trazado solo es válido para la imagen con la que se trazó. Usa D-FINE para un DETR exportable |
| `eomt` | Exportación de instancias y panóptica, que no tienen parseo en runtime |
| `l2cs` | Cualquier cosa fuera de ONNX, TorchScript, ExecuTorch, TensorRT y OpenVINO |
| `hrnet` | Cualquier cosa fuera de ONNX, TorchScript, OpenVINO y TensorRT |
| `sam`, `sam2`, `sam3`, `edgetam`, `mobilesam` | Todos los formatos; la exportación de modelos promptables queda fuera del alcance del contrato de runtime v1 |
| `grounding_dino`, `owlv2`, `omdet_turbo`, `ov_deim` | Todos los formatos; la exportación en runtime de vocabulario abierto queda fuera del alcance de la v1 |
| `florence2`, `kosmos2`, `lfm2vl`, `internvl3`, `qwen3vl`, `smolvlm2`, `locateanything` | Todos los formatos; la exportación de VLM generativos queda fuera del alcance de la v1 |

PicoSAM3 es la excepción dentro del nivel promptable: exporta a ONNX su red de
ROI en crudo de 96 píxeles.

## Bloqueadas para NCNN

Los decodificadores de estilo DETR necesitan operaciones de sampling que NCNN no
implementa, así que estas familias están bloqueadas para `ncnn` salvo que una
entrada explícita diga lo contrario: Deformable DETR, DETR, DINO-DETR, D-FINE,
LW-DETR, DEIM, DEIMv2, RT-DETR, RT-DETRv2, RT-DETRv4, RF-DETR y EC. El rechazo
nombra ONNX, OpenVINO, TorchScript y TensorRT como alternativas.

## Umbrales de paridad

Una celda validated significa que el artefacto exportado reprodujo el modelo
nativo dentro de estos márgenes:

| Grupo de tareas | Umbral |
|---|---|
| Detección y OBB | IoU de los boxes emparejados por encima de 0,95, MAE de la puntuación por debajo de 0,01 |
| Segmentación y panóptica | IoU de máscara por encima de 0,95 |
| Pose | L2 de keypoints por debajo de 2 píxeles a resolución nativa |
| Clasificación | Coseno de los logits por encima de 0,999 y misma clase top-1 |
| Profundidad y restauración | PSNR por encima de 40 dB frente a la salida nativa |
| Normales de superficie | Error angular medio por debajo de 0,1 grados |
| Point | Ubicaciones de los picos iguales dentro de una misma celda de salida |

Las filas de queries de DETR son un conjunto sin orden, así que la paridad de la
familia DETR alinea las filas de queries como conjunto y no por posición.

## Exportar

<code-tabs name="export" />

Una combinación bloqueada lanza `NotImplementedError` en el preflight y el
mensaje lleva el motivo registrado. `validated_alternatives(family, task)`
devuelve los formatos que están validados para ese par, que es lo útil para
imprimir junto a un rechazo.

Los argumentos que comparten todos los exportadores están listados en la
[página de la API del modelo](/docs/reference/model-api). Los argumentos
específicos de cada formato viven en las páginas de cada formato.

## Leer una restricción

Una celda validated es una afirmación sobre una configuración medida, no sobre
el formato en general. Una cadena de restricción como
`FP32, batch 1, fixed 520x520 input` significa que la paridad se registró con esa
forma y esa precisión. Exportar a otra resolución u otro tamaño de batch sigue
produciendo un artefacto; simplemente no es la configuración de la que salió el
número.
