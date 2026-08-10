---
title: libreyolo export
seo_title: "referencia del comando libreyolo export"
description: "Exporta un checkpoint a un formato de despliegue: cada argumento con su valor por defecto, dónde acaba el artefacto y las combinaciones que el comando rechaza."
lead: "Convierte un checkpoint en un formato de despliegue y escribe el artefacto en weights/. El formato decide cuáles de los argumentos de abajo aplican."
keywords: [libreyolo export cli, exportar yolo a onnx, comando libreyolo export, exportar yolo tensorrt, argumentos libreyolo export]
last_verified: "1.5.0"
meta:
  - label: Comando
    value: libreyolo export
    mono: true
  - label: Requerido
    value: model
    mono: true
  - label: Salida
    value: "weights/<checkpoint-stem>[_fp16|_int8]<format-suffix>"
    mono: true
snippets:
  examples:
    - label: Básico
      language: bash
      code: |
        # Escribe weights/LibreYOLO9s.onnx
        libreyolo export model=LibreYOLO9s.pt format=onnx imgsz=640
    - label: NMS dentro del grafo
      language: bash
      code: |
        libreyolo export model=LibreYOLO9s.pt format=onnx \
          nms=true conf=0.25 iou=0.45 max_det=300
    - label: Ejecutar el artefacto
      language: bash
      code: |
        libreyolo export model=LibreYOLO9s.pt format=onnx imgsz=640

        # La factoría se guía por el sufijo del archivo, así que la exportación se carga como un checkpoint.
        libreyolo predict model=weights/LibreYOLO9s.onnx \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
---

## Sinopsis

```bash
libreyolo export model=<name|path> [format=<format>] [key=value ...]
```

Los argumentos son pares `key=value`, y la forma POSIX también funciona, de modo
que `format=onnx` y `--format onnx` son el mismo argumento.

## Argumentos

| Argumento | Por defecto | Significado |
|---|---|---|
| `model` | | Pesos del modelo `.pt`. Requerido |
| `format` | `onnx` | Formato de exportación: `onnx`, `torchscript`, `executorch`, `tensorrt`, `openvino`, `paddle`, `mnn`, `rknn`, `ncnn`, `tflite`, `coreml`, `coreai` |
| `name` | | Plataforma de destino de RKNN, por ahora solo `rk3588`. Se rechaza con cualquier otro formato |
| `imgsz` | | Tamaño de la imagen de entrada: `640` o `480x640` (alto x ancho). También se acepta `480,640`. El tamaño propio del modelo si no se indica |
| `batch` | `1` | Tamaño de batch de la exportación |
| `half` | `false` | Precisión FP16 |
| `int8` | `false` | Cuantización INT8 |
| `dynamic` | `false` | Formas de entrada dinámicas (ONNX) |
| `simplify` | `true` | Simplificación del grafo ONNX |
| `nms` | `false` | Incrusta el NMS en el modelo. Solo ONNX y CoreML |
| `conf` | `0.25` | Umbral de confianza para el NMS incrustado |
| `iou` | `0.45` | Umbral de IoU para el NMS incrustado |
| `max_det` | `300` | Máximo de detecciones para el NMS incrustado de ONNX |
| `opset` | | Versión del opset de ONNX. Se elige automáticamente si no se indica |
| `data` | | Datos de calibración para INT8 |
| `fraction` | `1.0` | Fracción de los datos de calibración que se usa |
| `device` | `auto` | Dispositivo para el trazado |
| `allow_download_scripts` | `false` | Permite Python incrustado en los bloques de descarga del YAML del dataset |
| `json` | `false` | Salida JSON por stdout |
| `quiet` | `false` | Silencia stderr |
| `verbose` | `false` | Registro detallado de la exportación |
| `verify` | `false` | Ejecuta el simulador de PC de RKNN Toolkit2 y lo compara con ONNX Runtime. Solo RKNN |
| `help_json` | `false` | Vuelca el esquema del comando como JSON y sale |

`engine` es un alias de `tensorrt` y `litert` un alias de `tflite`. Ambos se
resuelven al nombre canónico antes de escribir nada, así que la salida JSON y la
línea de log siempre indican `tensorrt` o `tflite`.

## Ejemplos

<code-tabs name="examples" />

## Notas

### Dónde acaba el archivo

El comando no admite ruta de salida. El artefacto se escribe en `weights/`, con
el nombre de la raíz del checkpoint de origen más el sufijo del formato, y con
`_fp16` o `_int8` intercalado cuando se ha pedido una de esas precisiones.
`LibreYOLO9s.pt` exportado a ONNX en FP16 se convierte en
`weights/LibreYOLO9s_fp16.onnx`. El resultado JSON lleva el `output_path`
resuelto, el tamaño del archivo en MB y la forma de entrada como
`[batch, 3, height, width]`.

### Combinaciones que se rechazan

`nms=true` se acepta para ONNX y CoreML y se rechaza para todos los demás
formatos con `nms_unsupported_format`. En ONNX obliga a desactivar `dynamic`,
ya que el grafo incrustado está fijado a batch 1, y lo indica por stderr. En
CoreML admite `conf` e `iou` pero no `max_det`, así que un `max_det` distinto
del valor por defecto junto a `format=coreml nms=true` sale con
`config_unsupported`.

`half=true` junto con `int8=true` no es un error. Gana INT8, se descarta `half`
y va una advertencia a stderr.

`name` y `verify` son hoy opciones de RKNN. Pasar cualquiera de las dos con otro
formato sale con `config_unsupported` en lugar de ignorarse.

### Qué formatos admite cada familia

El soporte es por familia y por tarea, no global. `libreyolo formats
family=<family> task=<task>` imprime el nivel de cada formato para esa
combinación, con el motivo y cualquier restricción asociada. Consulta
[`libreyolo formats`](/docs/cli/utilities) para ver los argumentos.

Algunos formatos necesitan una instalación opcional y otros necesitan un
toolchain. Una dependencia de Python que falta sale con `export_dep_missing`;
una precisión que el formato no puede producir sale con
`format_precision_unsupported`.

### Ejecutar lo que has exportado

Los artefactos exportados se cargan a través de la misma factoría de modelos que
los checkpoints, guiada por el sufijo del archivo, así que
`libreyolo predict model=weights/LibreYOLO9s.onnx` funciona sin ninguna
conversión adicional. Tres opciones de predicción son la excepción y se rechazan
en los backends de runtime: `tiling`, `overlap_ratio` y `output_file_format`.

Dos destinos de despliegue tienen página propia:
[NVIDIA DeepStream](/docs/export/deepstream) y
[NVIDIA Jetson](/docs/export/jetson).

### Salida y códigos de salida

stdout lleva el resultado; el progreso va a stderr. El código de salida es `0`
en caso de éxito, `2` para un error de uso o de configuración, `4` cuando el
modelo no se puede cargar, `5` para un formato desconocido, una dependencia de
exportación que falta, una precisión no soportada o una petición de NMS
incrustado rechazada, y `1` para otros fallos en tiempo de ejecución.

Relacionado: [`libreyolo quantize`](/docs/cli/quantize), que se queda en PyTorch
y escribe un checkpoint en lugar de un artefacto de despliegue.
