---
title: utilidades de libreyolo
seo_title: referencia de los comandos de utilidad de la CLI de libreyolo
description: >-
  Los comandos pequeños de LibreYOLO: version, checks, models, formats, cfg,
  info, metadata, enroll y compare, cada uno con sus argumentos y sus valores
  por defecto.
lead: >-
  Nueve comandos que informan o inspeccionan en lugar de calcular. Imprimen
  datos del entorno, el inventario de modelos y formatos, los valores por
  defecto ya resueltos y los detalles de un checkpoint, y construyen y consultan
  una galería de caras.
keywords:
  - libreyolo version
  - libreyolo checks
  - listar modelos libreyolo
  - formatos de exportacion libreyolo
  - ver metadatos de un checkpoint yolo
  - galeria de caras libreyolo enroll
last_verified: 1.5.0
meta:
  - label: Comandos
    value: 'version, checks, models, formats, cfg, info, metadata, enroll, compare'
    mono: true
  - label: Salida
    value: >-
      stdout, en texto o con json=true como un único objeto que lleva
      schema_version
snippets:
  examples:
    - label: Entorno
      language: bash
      code: |
        libreyolo version
        libreyolo checks
    - label: Qué hay disponible
      language: bash
      code: |
        libreyolo models
        libreyolo formats family=yolo9 task=detect
    - label: Inspeccionar un checkpoint
      language: bash
      code: |
        libreyolo info model=LibreYOLO9s.pt
        libreyolo metadata path=weights/LibreYOLO9s.pt
source_hash: 7b5b53c46df00c06
---

## Sinopsis

```bash
libreyolo <command> [key=value ...]
```

Los argumentos son pares `key=value`, y la forma POSIX también funciona, así que
`model=x` y `--model x` son el mismo argumento. Todos los comandos de esta
página escriben sus resultados por stdout y aceptan `json=true` y `quiet=true`.

El comando raíz lleva un flag propio, `libreyolo --version`, que imprime la
cadena de versión y sale. Es una salida más pequeña que la del comando `version`
de abajo.

## version

Imprime la versión de LibreYOLO más las versiones de Python, torch y CUDA
contra las que se está ejecutando.

```bash
libreyolo version
```

| Argumento | Por defecto | Significado |
|---|---|---|
| `json` | `false` | Salida JSON por stdout |
| `quiet` | `false` | Silencia stderr |

## checks

Imprime el entorno con más detalle: Python, torch, CUDA, cuDNN, cada GPU
detectada con su nombre y su memoria, y la versión instalada de cada paquete
opcional que usan las rutas de exportación.

```bash
libreyolo checks
```

| Argumento | Por defecto | Significado |
|---|---|---|
| `json` | `false` | Salida JSON por stdout |
| `quiet` | `false` | Silencia stderr |

La lista de paquetes cubre `onnx`, `onnxruntime`, `tensorrt`, `openvino`,
`paddlepaddle`, `x2paddle`, `mnn`, `ncnn`, `onnx2tf`, `ai-edge-litert`,
`transformers` y `scipy`. Un paquete que no está instalado se informa como tal
en lugar de omitirse, así que una exportación fallida se puede rastrear hasta
una dependencia que falta con este único comando.

## models

Lista cada familia de modelos con sus tareas, sus tamaños, los nombres de CLI
que resuelven a sus checkpoints y la resolución de entrada de cada tamaño.

```bash
libreyolo models
```

| Argumento | Por defecto | Significado |
|---|---|---|
| `json` | `false` | Salida JSON por stdout |
| `quiet` | `false` | Silencia stderr |

Una familia cuya dependencia opcional no está instalada aparece como no
disponible junto con la línea `pip install` que la haría disponible. Los
nombres de CLI son lo que `model=` acepta como forma abreviada: `yolox-s`
resuelve a `LibreYOLOXs.pt`, y las tareas que no son de detección llevan el
sufijo de su tarea.

## formats

Lista los formatos de exportación que el entorno instalado puede producir, con
la extensión de archivo de cada formato y si admite FP16 e INT8.

```bash
libreyolo formats [family=<family>] [task=<task>]
```

| Argumento | Por defecto | Significado |
|---|---|---|
| `family` | | Muestra los niveles de soporte para una familia de modelos. Se acepta `model=` como la misma opción |
| `task` | | Tarea canónica del modelo. La tarea por defecto de la familia si no se indica |
| `json` | `false` | Salida JSON por stdout |
| `quiet` | `false` | Silencia stderr |

Sin `family`, la salida es solo el inventario de formatos. Con ella, cada
formato gana el nivel de soporte para esa familia y esa tarea, el motivo detrás
del nivel y cualquier restricción asociada. Una familia desconocida, o una tarea
que la familia no admite, es un error de uso.

Los alias de formato aparecen junto a su nombre canónico: `engine` para
`tensorrt`, `litert` para `tflite`.

## cfg

Imprime la configuración por defecto ya resuelta: los valores por defecto de
entrenamiento, los de validación, los de predicción y los overrides por familia.

```bash
libreyolo cfg
```

| Argumento | Por defecto | Significado |
|---|---|---|
| `json` | `false` | Salida JSON por stdout |
| `quiet` | `false` | Silencia stderr |

Los valores se leen de las dataclasses de configuración, no de una copia, así
que esto es la autoridad sobre lo que usará un entrenamiento cuando no pasas un
argumento. `family_overrides` es la sección que responde por qué una familia se
entrenó con ajustes que no pediste. Consulta
[`libreyolo train`](/docs/cli/train) para ver cómo se aplican esos overrides.

## info

Carga un modelo en la CPU e informa de su familia, su tamaño, su número de
parámetros, sus clases y el nivel de soporte de exportación de cada formato.

```bash
libreyolo info model=<name|path>
```

| Argumento | Por defecto | Significado |
|---|---|---|
| `model` | | Nombre del modelo o ruta a los pesos. Requerido |
| `detailed` | `false` | Incluye los detalles por parámetro |
| `json` | `false` | Salida JSON por stdout |
| `quiet` | `false` | Silencia stderr |

## metadata

Lee los metadatos de un checkpoint sin construir un modelo, y los valida contra
el esquema de checkpoints de LibreYOLO.

```bash
libreyolo metadata path=<checkpoint.pt>
```

| Argumento | Por defecto | Significado |
|---|---|---|
| `path` | | Ruta a un checkpoint `.pt`. Requerido |
| `json` | `false` | Salida JSON por stdout |
| `quiet` | `false` | Silencia stderr |

Las entradas grandes que llevan tensores se resumen en lugar de imprimirse, así
que la salida sigue siendo legible con un checkpoint completo de entrenamiento.
Un checkpoint que no existe sale con `checkpoint_not_found`, y uno cuyos
metadatos no pasan la validación imprime los errores y sale con `1`.

## enroll

Construye una galería de caras a partir de un árbol de una carpeta por persona,
para que las predicciones posteriores puedan poner nombre a las caras que
encuentran.

```bash
libreyolo enroll model=<embedder> source=<people-dir> gallery=<gallery.npz>
```

| Argumento | Por defecto | Significado |
|---|---|---|
| `model` | | Modelo de embeddings de caras, ruta o nombre. Requerido |
| `source` | | Árbol de una carpeta por persona, `source/<identity>/*.jpg`. Requerido |
| `gallery` | | Archivo `.npz` de galería de salida. Se amplía in situ si ya existe. Requerido |
| `face_detector` | | Detector de caras: un `.onnx` de YuNet o un detector de LibreYOLO. El detector por defecto de la familia si no se indica |
| `device` | `auto` | Dispositivo: `0`, `cpu`, `mps`, `auto` |
| `json` | `false` | Salida JSON por stdout |
| `quiet` | `false` | Silencia stderr |

```bash
# people/ tiene una carpeta por identidad; el nombre de la carpeta pasa a ser la identidad.
libreyolo enroll model=librefacerec-l.onnx source=people/ gallery=people.npz
```

El nombre de la subcarpeta es la identidad. Una imagen de referencia sin ninguna
cara detectable se omite con una línea por stderr y el resto continúan; una
fuente sin subcarpetas de identidad, o una en la que no se encontró ninguna cara
en absoluto, es un error.

Pasa el archivo resultante a
[`libreyolo predict`](/docs/cli/predict) como `gallery=people.npz` para que las
detecciones lleven una identidad y una puntuación de coincidencia.

## compare

Informa de la similitud coseno entre dos imágenes de caras y de si supera el
umbral de misma identidad.

```bash
libreyolo compare model=<embedder> source=<a.jpg> source2=<b.jpg>
```

| Argumento | Por defecto | Significado |
|---|---|---|
| `model` | | Modelo de embeddings de caras, ruta o nombre. Requerido |
| `source` | | Primera imagen. Requerido |
| `source2` | | Segunda imagen con la que comparar. Requerido |
| `face_detector` | | Detector de caras: un `.onnx` de YuNet o un detector de LibreYOLO |
| `threshold` | `0.4` | Umbral de similitud coseno para la decisión de misma identidad |
| `device` | `auto` | Dispositivo: `0`, `cpu`, `mps`, `auto` |
| `json` | `false` | Salida JSON por stdout |
| `quiet` | `false` | Silencia stderr |

```bash
libreyolo compare model=librefacerec-l.onnx source=a.jpg source2=b.jpg
```

`libreyolo verify` está registrado como segundo nombre de este comando y toma
los mismos argumentos.

Tanto `compare` como `enroll` necesitan un modelo cuya tarea sea el embedding de
caras. Cualquier otra cosa sale con `config_unsupported`. Se aceptan como
fuentes tanto rutas locales de imagen como URLs `http` o `https`.

## Ejemplos

<code-tabs name="examples" />

## Notas

stdout lleva el resultado; el progreso y los avisos van a stderr. `json=true`
imprime un único objeto con `schema_version`, que es la forma que hay que leer
desde un script. La salida en texto es la de por defecto y está pensada para que
la lea una persona.

Los códigos de salida siguen el mismo mapa que el resto de la CLI: `0` si todo
va bien, `2` para un error de uso o de configuración, `3` cuando no se encuentra
una fuente, `4` cuando no se puede cargar un modelo o un checkpoint, y `1` para
el resto de fallos en tiempo de ejecución.

Relacionado: [`libreyolo doctor`](/docs/cli/doctor), que es el comando de
inspección del lado del dataset, y [`libreyolo profile`](/docs/cli/profile), el
del lado del rendimiento.
