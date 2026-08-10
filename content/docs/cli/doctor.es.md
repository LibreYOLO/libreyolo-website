---
title: libreyolo doctor
seo_title: referencia del comando libreyolo doctor
description: >-
  Comprueba un dataset de detección antes de entrenar: los argumentos con sus
  valores por defecto, las familias de comprobaciones que puedes omitir o
  seleccionar y los códigos de salida sobre los que CI puede fallar.
lead: >-
  Ejecuta un conjunto de comprobaciones de salud sobre un dataset de detección e
  informa de lo que perjudicaría a un entrenamiento: archivos que faltan,
  etiquetas rotas, imágenes corruptas, fuga entre splits y desbalanceo de
  clases.
keywords:
  - libreyolo doctor cli
  - comprobar salud dataset yolo
  - validar dataset deteccion
  - fuga de datos entre splits
  - libreyolo doctor strict
last_verified: 1.5.0
meta:
  - label: Comando
    value: libreyolo doctor
    mono: true
  - label: Requerido
    value: data
    mono: true
  - label: Salida
    value: >-
      Un informe de hallazgos por stdout. Sale con 1 cuando se encuentran
      errores
snippets:
  examples:
    - label: Básico
      language: bash
      code: >
        # download=true permite que el coco8.yaml incluido descargue sus
        imágenes si faltan.

        libreyolo doctor coco8.yaml download=true
    - label: 'Pasada rápida, sin decodificar imágenes'
      language: bash
      code: |
        libreyolo doctor coco8.yaml download=true fast=true
    - label: Gate de CI sobre comprobaciones seleccionadas
      language: bash
      code: |
        libreyolo doctor coco8.yaml download=true strict=true json=true \
          only=labels,files,config
source_hash: 79e0ef471d567ea3
---

## Sinopsis

```bash
libreyolo doctor <data.yaml> [key=value ...]
```

El dataset es posicional, y se acepta `data=<path>` como alternativa. Dar ambos
con valores distintos termina con `config_conflict`. Todo lo demás son pares
`key=value`, y la forma POSIX también funciona, así que `imgsz=1024` y
`--imgsz 1024` son el mismo argumento.

## Argumentos

| Argumento | Por defecto | Significado |
|---|---|---|
| `data` | | Posicional. YAML del dataset en formato de detección YOLO, p. ej. `coco8.yaml`. Requerido |
| `imgsz` | `640` | Tamaño de imagen de entrenamiento usado en las comprobaciones basadas en píxeles, como la de objetos diminutos |
| `fast` | `false` | Omite la decodificación de imágenes, lo que descarta las comprobaciones de corrupción, duplicados y fuga |
| `skip` | | IDs de comprobación o familias separados por comas que se omiten, p. ej. `images,labels.tiny_object` |
| `only` | | IDs de comprobación o familias separados por comas que se ejecutan en exclusiva |
| `strict` | `false` | Los avisos también hacen fallar el código de salida, para gates de CI |
| `download` | `false` | Permite descargar el dataset por URL si falta. Nunca scripts |
| `json` | `false` | Salida JSON por stdout |
| `quiet` | `false` | Silencia stderr |
| `help_json` | `false` | Vuelca el esquema del comando como JSON y sale |

### Familias de comprobaciones

`skip` y `only` aceptan tanto un id de comprobación completo como un prefijo de
familia, así que `images` selecciona todas las comprobaciones `images.*`.

| Familia | Cubre |
|---|---|
| `config` | El propio YAML del dataset: `names` ausente, `nc` frente a `names`, splits que faltan, `path` que no se puede resolver, nombres de clase duplicados |
| `files` | El emparejamiento de imágenes y etiquetas: etiquetas que faltan, imágenes que faltan, etiquetas huérfanas, extensiones no soportadas, colisiones de mayúsculas y minúsculas |
| `labels` | El contenido de las etiquetas: sintaxis, líneas de polígono, ids de clase fuera de rango, coordenadas fuera de rango, bounding boxes degenerados, objetos diminutos, boxes enormes, relaciones de aspecto extremas, boxes duplicados, imágenes abarrotadas, archivos idénticos |
| `images` | Los datos de píxeles: archivos corruptos, orientación EXIF, modos de color inusuales, dimensiones diminutas o extremas, imágenes uniformes, duplicados exactos y aproximados |
| `splits` | Fuga entre splits, exacta y aproximada |
| `balance` | La distribución de clases: clases con cero o pocas instancias, desbalanceo, cobertura de los splits, proporción de fondo, sesgo entre splits |

## Ejemplos

<code-tabs name="examples" />

## Notas

### Códigos de salida

`0` cuando no se encontraron errores, `1` cuando algún hallazgo es un error. Con
`strict=true`, los avisos también elevan el código de salida a `1`, que es la
configuración que quiere un gate de CI.

Los problemas de uso tienen sus propios códigos: `2` para un id de comprobación
o familia desconocidos en `skip` u `only`, `3` cuando no se encuentra el dataset
y `3` cuando el dataset no tiene forma de detección.

### La selección se resuelve antes del escaneo

`skip` y `only` se resuelven contra el registro de comprobaciones antes de leer
nada del disco, así que una errata falla de inmediato en lugar de hacerlo tras
una larga pasada por las imágenes. Un selector que no coincide con nada es un
error, y el mensaje enumera las familias conocidas.

Si la combinación de `skip`, `only` y `fast` no deja ninguna comprobación por
ejecutar, eso también es un error en lugar de una pasada silenciosa.

### Descargas

El dataset no se descarga salvo que `download=true`, y solo se realizan
descargas por URL. Este comando nunca ejecuta un script de descarga en Python
embebido en el YAML de un dataset, sea cual sea el valor del flag.

### Alcance

Las comprobaciones están escritas para datasets de detección. Un dataset cuyas
etiquetas tienen forma de pose, de segmentación o de oriented box se detecta y
se rechaza con `data_invalid` en lugar de evaluarse con las reglas equivocadas.

### Salida

El informe legible para personas va a stdout, y `json=true` lo sustituye por un
objeto estructurado que lleva los recuentos del resumen, las estadísticas del
dataset, todos los hallazgos y la lista de comprobaciones que se omitieron.

Relacionado: [`libreyolo train`](/docs/cli/train), la ejecución antes de la que
está pensado ejecutar este comando.
