---
title: libreyolo quantize
seo_title: referencia del comando libreyolo quantize
description: >-
  Cuantiza un checkpoint en PyTorch desde la línea de comandos: recetas,
  argumentos de calibración, valores por defecto y las familias que acepta cada
  receta.
lead: >-
  Sustituye los módulos float de un modelo por módulos cuantizados, los calibra
  sobre imágenes sin etiquetar cuando la receta necesita estadísticas y guarda
  el resultado como un checkpoint de PyTorch.
keywords:
  - libreyolo quantize cli
  - cuantizacion int8 linea de comandos
  - cuantizacion fp8
  - cuantizacion post entrenamiento yolo
  - argumentos libreyolo quantize
last_verified: 1.5.0
meta:
  - label: Comando
    value: libreyolo quantize
    mono: true
  - label: Requerido
    value: model
    mono: true
  - label: Salida
    value: >-
      La ruta de origen con -<recipe> antes del sufijo, p. ej.
      LibreYOLO9s-int8.pt
    mono: true
snippets:
  examples:
    - label: Básico
      language: bash
      code: |
        # Calibra sobre coco128 y escribe LibreYOLO9s-int8.pt
        libreyolo quantize model=LibreYOLO9s.pt recipe=int8
    - label: 'Solo cast, sin calibración'
      language: bash
      code: |
        libreyolo quantize model=LibreYOLO9s.pt recipe=fp16 calib=none \
          out=weights/LibreYOLO9s-fp16.pt
    - label: Calibración más amplia y luego recuperación
      language: bash
      code: >
        libreyolo quantize model=LibreYOLO9s.pt recipe=int8 \
          calib=coco128.yaml samples=256 batch=16 algorithm=minmax

        # El entrenamiento consciente de la cuantización sobre el checkpoint
        cuantizado recupera precisión.

        libreyolo train model=LibreYOLO9s-int8.pt data=coco8.yaml epochs=10
        lr0=0.001
source_hash: 7ae663e9f117826e
---

## Sinopsis

```bash
libreyolo quantize model=<name|path> [recipe=<recipe>] [key=value ...]
```

Los argumentos son pares `key=value`, y la forma POSIX también funciona, así que
`recipe=int8` y `--recipe int8` son el mismo argumento.

## Argumentos

| Argumento | Por defecto | Significado |
|---|---|---|
| `model` | | Pesos del modelo `.pt`. Requerido |
| `recipe` | `int8` | Receta de cuantización: `fp16`, `bf16`, `fp8`, `int8`, `w4a16`, `w4a8`, `nvfp4`, `mxfp4`, `int2` |
| `calib` | `coco128.yaml` | Imágenes de calibración: un YAML de datos o el nombre de un dataset integrado. Sin etiquetar, solo forward. `none` omite la calibración |
| `samples` | `128` | Máximo de imágenes de calibración |
| `batch` | `8` | Tamaño de batch de calibración |
| `algorithm` | `auto` | Estimación del rango de activaciones: `auto`, que selecciona minmax, o `minmax`, o `percentile` |
| `out` | | Ruta del checkpoint de salida. Por defecto, la ruta de origen con `-<recipe>` antes del sufijo |
| `device` | `auto` | Dispositivo |
| `allow_download_scripts` | `false` | Permite Python embebido en los bloques de descarga del YAML del dataset |
| `json` | `false` | Salida JSON por stdout |
| `quiet` | `false` | Silencia stderr |
| `help_json` | `false` | Vuelca el esquema del comando como JSON y termina |

## Ejemplos

<code-tabs name="examples" />

## Notas

### Qué familias lo aceptan

La cuantización cubre cuatro familias: `yolo9`, `rfdetr`, `birefnet` y
`feynobg`. Cualquier otra familia sale con `quantize_failed`, que lleva la lista.

### Qué toca cada receta

`fp16` y `bf16` son casts. Cambian solo el dtype, no necesitan calibración y
`calib=none` es el ajuste adecuado para ellas.

`int8` y `fp8` cuantizan los módulos `Conv2d` y `Linear`, y por eso encajan con
las familias convolucionales.

`w4a16`, `w4a8`, `nvfp4`, `mxfp4` e `int2` cuantizan solo `nn.Linear`, así que
apuntan a las familias transformer. Pedir una de ellas sobre `yolo9` se rechaza
con una explicación en lugar de producir en silencio un modelo sin cuantizar,
porque ahí la aceleración por debajo de 8 bits es solo para GEMM y las
convoluciones se quedarían en mayor precisión.

`int8`, `fp8`, `w4a8` e `int2` necesitan estadísticas de calibración para sus
activaciones. `int2` además necesita entrenamiento posterior para recuperarse,
por lo que se rechaza en `birefnet` y `feynobg`, que no tienen entrenador.

Cada familia mantiene un conjunto de módulos en float sea cual sea la receta:
las primeras capas, las cabezas de predicción y, en YOLOv9, la convolución DFL,
que es un operador de esperanza integral fijo que no debe cuantizarse.

### Los datos de calibración no son datos de entrenamiento

`calib` apunta a un pequeño conjunto de imágenes sin etiquetar, usado solo hacia
delante, para derivar los rangos de activación. No se evalúa contra él y sus
etiquetas no se leen nunca. El `coco128.yaml` por defecto se descarga en el
primer uso desde una URL, así que no necesita permiso adicional; un YAML con un
script de descarga en Python embebido necesita `allow_download_scripts=true`.

`algorithm=percentile` está disponible y puede reducir la precisión en las
familias transformer, y por eso `auto` selecciona minmax.

### Recuperar la precisión

La salida es un checkpoint de PyTorch normal, así que
[`libreyolo train`](/docs/cli/train) lo acepta directamente. Entrenar un
checkpoint cuantizado es entrenamiento consciente de la cuantización; añadir
`distill_model=<teacher>` lo convierte en destilación consciente de la
cuantización.

### Salida y códigos de salida

El resultado imprime la ruta guardada, la receta, el modo de ejecución, si se
ejecutó la calibración y el número de módulos intercambiados por tipo. El código
de salida es `0` en caso de éxito, `4` cuando el modelo no se puede cargar, `5`
cuando falla la cuantización o el guardado, y `1` para otros fallos en tiempo de
ejecución.

Relacionado: [`libreyolo export`](/docs/cli/export), que sale de PyTorch y
escribe en su lugar un artefacto de despliegue.
