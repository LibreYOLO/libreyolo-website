---
title: libreyolo monitor
seo_title: referencia del comando libreyolo monitor
description: >-
  Sirve un dashboard en vivo para los entrenamientos: argumentos con sus valores
  por defecto, qué lee el servidor del disco y cómo un solo servidor cubre
  muchos runs.
lead: >-
  Sirve un dashboard web para los entrenamientos, leyendo los artefactos que un
  run escribe en disco. Nunca se engancha al proceso de entrenamiento, así que
  se muestran igual los runs en vivo, los terminados y los que han fallado.
keywords:
  - libreyolo monitor cli
  - dashboard de entrenamiento
  - ver entrenamiento en tiempo real
  - libreyolo monitor puerto
  - visor de metricas de entrenamiento
last_verified: 1.5.0
meta:
  - label: Comando
    value: libreyolo monitor
    mono: true
  - label: Salida
    value: >-
      Una URL de servidor por stdout; después el proceso se queda en primer
      plano
snippets:
  examples:
    - label: Básico
      language: bash
      code: |
        # Vigila runs/ y lista todos los runs que hay debajo.
        libreyolo monitor
    - label: Otra raíz de runs
      language: bash
      code: |
        libreyolo monitor experiments/
    - label: 'Un solo run, puerto fijo, sin navegador'
      language: bash
      code: |
        libreyolo monitor runs/train/exp port=9100 no_browser=true
source_hash: 4aa178141d451728
---

## Sinopsis

```bash
libreyolo monitor [<run-dir|runs-root>] [key=value ...]
```

El directorio es posicional. Todo lo demás es un par `key=value`, y la forma
POSIX también funciona, de modo que `port=9100` y `--port 9100` son el mismo
argumento.

## Argumentos

| Argumento | Por defecto | Significado |
|---|---|---|
| `run_dir` | `runs` | Posicional. Una raíz de runs que vigilar, o un único directorio de run que abrir directamente. En ambos casos se listan todos los runs que hay bajo la raíz |
| `host` | `127.0.0.1` | Host o interfaz a la que enlazar |
| `port` | `8420` | Puerto al que enlazar. Salta al siguiente libre si está ocupado |
| `no_browser` | `false` | No abrir el navegador automáticamente |
| `json` | `false` | Salida JSON por stdout |
| `quiet` | `false` | Silencia stderr |
| `verbose` | `false` | Salida detallada por stderr |

## Ejemplos

<code-tabs name="examples" />

## Notas

### Un servidor, muchos runs

El servidor vigila una raíz de runs en lugar de un único run, y direcciona cada
run por URL, así que varios runs en una misma máquina comparten un solo puerto.
Abre la URL raíz para ver el índice, o una pestaña por run; el parámetro `?run=`
de cada URL indica cuál.

Apuntar el comando a un único directorio de run hace que el servidor tome como
raíz el directorio padre, de modo que los runs hermanos siguen apareciendo en el
índice, y enlaza directamente al que se ha nombrado.

### Qué lee

El dashboard se construye a partir de los archivos que escribe
`libreyolo train`: `status.json`, `metrics.jsonl`, `train.log` y las imágenes
del run. No
se lee nada del propio proceso de entrenamiento, así que un run que ha
terminado, o que ha muerto, se muestra exactamente igual que uno en vivo.

### Requisitos previos y puertos

Tiene que existir ya al menos un run. Sin argumento y sin directorio `runs/`, el
comando sale con `source_not_found`; ocurre lo mismo cuando el directorio
indicado no contiene ningún run.

Un puerto ocupado pasa al siguiente, hasta veinte por encima del solicitado. Si
fallan los veinte, sale con `io_error`. La URL impresa por stdout corresponde al
puerto que se ha enlazado realmente.

El comando sirve en primer plano hasta que se pulsa Ctrl+C. `json=true` imprime
la URL, la raíz que se está vigilando y el número de runs encontrados, como un
único objeto con `schema_version`.

Relacionado: [`libreyolo train`](/docs/cli/train), cuyos argumentos `project` y
`name` deciden dónde van a parar estos directorios de run.
