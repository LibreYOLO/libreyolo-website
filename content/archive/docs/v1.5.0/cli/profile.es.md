---
title: libreyolo profile
seo_title: referencia del comando libreyolo profile
description: >-
  Mide la velocidad del entrenamiento y de la inferencia y lee el resultado:
  cada subcomando de profile, sus argumentos y valores por defecto, y qué
  informa cada ángulo de lectura.
lead: >-
  Un grupo de comandos que mide adónde se va el tiempo en un paso de
  entrenamiento o en una llamada de inferencia, escribe un perfil autocontenido
  y vuelve a leer ese perfil desde varios ángulos.
keywords:
  - libreyolo profile cli
  - profiling entrenamiento yolo
  - medir latencia inferencia yolo
  - perfilado kernels gpu pytorch
  - comparar rendimiento libreyolo
last_verified: 1.5.0
meta:
  - label: Comando
    value: libreyolo profile
    mono: true
  - label: Salida
    value: profile.json y profile_trace.json en runs/profile
    mono: true
snippets:
  examples:
    - label: Medir la inferencia
      language: bash
      code: |
        # Sin argumento source se usa la imagen de ejemplo incluida.
        libreyolo profile infer --device cpu --warmup 5 --runs 20
    - label: Leer el veredicto
      language: bash
      code: |
        libreyolo profile summary runs/profile/infer/profile.json
    - label: Comparar dos mediciones
      language: bash
      code: >
        libreyolo profile infer --device cpu --warmup 5 --runs 20 --project
        runs/profile/a

        libreyolo profile infer --device cpu --warmup 5 --runs 20 --batch 4
        --project runs/profile/b


        libreyolo profile compare runs/profile/a/infer/profile.json \
          runs/profile/b/infer/profile.json
source_hash: b967e869fd9ba418
---

## Sinopsis

```bash
libreyolo profile <subcommand> [<positional>] [--flag value ...]
```

Este grupo no acepta argumentos `key=value`. Sus subcomandos usan argumentos
posicionales y flags POSIX, así que es `--weights LibreYOLO9t.pt`, no
`weights=LibreYOLO9t.pt`. Ejecutar `libreyolo profile` sin subcomando imprime
la lista.

Dos subcomandos miden y escriben un perfil; el resto lo leen. Tanto `run` como
`infer` emiten el mismo `profile.json` autocontenido, así que todos los
subcomandos de lectura funcionan con cualquiera de los dos.

## profile run

Ejecuta un entrenamiento corto perfilado y escribe un perfil.

```bash
libreyolo profile run <data> [--flag value ...]
```

| Argumento | Por defecto | Significado |
|---|---|---|
| `data` | | Posicional. YAML o nombre del dataset, p. ej. `coco128`. Requerido |
| `--weights` | `LibreYOLO9t.pt` | Archivo o nombre de los pesos del modelo |
| `--size` | `t` | Variante de tamaño del modelo |
| `--batch` | `16` | Micro-batch. `-1` ajusta automáticamente cerca del 70% de la VRAM |
| `--imgsz` | `640` | Tamaño de la imagen de entrenamiento |
| `--workers` | `8` | Workers del dataloader |
| `--amp` | `true` | Usa la ruta AMP de la familia. `--no-amp` lo desactiva |
| `--steps` | `20` | Pasos perfilados, es decir, medidos |
| `--warmup` | `5` | Pasos de calentamiento antes de medir |
| `--repeat` | `1` | Repite N veces para obtener una media y una desviación estándar |
| `--device` | `0` | Dispositivo |
| `--project` | `runs/profile` | Raíz del directorio de salida |
| `--json` | `false` | Salida JSON por stdout |

La ventana medida son las iteraciones de `--warmup` más las de `--steps`. Un
dataset demasiado pequeño para llenarla no produce ningún perfil y el comando
termina con el código `3`, nombrando las tres salidas: un dataset más grande,
menos pasos o un batch más pequeño.

Un `--repeat` por encima de 1 escribe un `runs/profile/profile_repeat.json`
agregado cuyas métricas escalares se promedian entre pruebas, mientras que las
listas de kernels vienen de la última prueba. Es además el requisito previo
para un veredicto de significancia en `compare`: una sola ejecución no puede
darlo.

## profile infer

Perfila la ruta de inferencia y escribe un perfil.

```bash
libreyolo profile infer [<source>] [--flag value ...]
```

| Argumento | Por defecto | Significado |
|---|---|---|
| `source` | | Posicional. Imagen o directorio. La imagen de ejemplo incluida si se omite |
| `--weights` | `LibreYOLO9t.pt` | Archivo o nombre de los pesos del modelo |
| `--size` | `t` | Variante de tamaño del modelo |
| `--batch` | `1` | Imágenes por pasada forward |
| `--imgsz` | `640` | Tamaño de la imagen de entrada |
| `--half` | `false` | Forward con autocast, solo CUDA. `--no-half` lo desactiva |
| `--amp-dtype` | `float16` | dtype del autocast de CUDA: `float16` o `bfloat16` |
| `--warmup` | `20` | Iteraciones de calentamiento antes de medir |
| `--runs` | `100` | Iteraciones medidas |
| `--repeat` | `1` | Repite N veces para obtener una media y una desviación estándar |
| `--conf` | `0.25` | Umbral de confianza, que cambia cuánto trabajo hace NMS |
| `--iou` | `0.45` | Umbral de IoU para NMS |
| `--max-det` | `300` | Máximo de detecciones por imagen, que cambia cuánto trabajo hace NMS |
| `--device` | `0` | Dispositivo |
| `--trace` | `true` | Emite una traza de Chrome para profundizar en kernels y ops. `--no-trace` la omite |
| `--project` | `runs/profile` | Raíz del directorio de salida |
| `--json` | `false` | Salida JSON por stdout |

Informa de la latencia en p50, p90 y p99, del throughput en imágenes por
segundo y del reparto por etapas entre preprocesado, forward y postprocesado.
Los tres argumentos de umbral están aquí porque mueven la cifra del
postprocesado.

## profile summary

```bash
libreyolo profile summary <trace> [--json]
```

| Argumento | Por defecto | Significado |
|---|---|---|
| `trace` | | Posicional. Ruta a un `profile.json` o `profile_trace.json`. Requerido |
| `--json` | `false` | Salida JSON por stdout |

La lectura de alto nivel: tiempo por paso, throughput, utilización de la GPU,
proporción de Tensor Cores, pico de VRAM, overhead del host, lanzamientos de
kernel por paso, el veredicto sobre el cuello de botella con su motivo, la
mezcla de kernels por categoría y los principales kernels por paso. En un
perfil de inferencia imprime además los percentiles de latencia y el reparto
por etapas.

Un perfil tomado con thrashing de VRAM queda marcado, porque la utilización y
el throughput medidos ahí no son fiables.

## profile get

```bash
libreyolo profile get <trace> [<field>] [--json]
```

| Argumento | Por defecto | Significado |
|---|---|---|
| `trace` | | Posicional. Ruta a un perfil. Requerido |
| `field` | | Posicional. Nombre de la métrica. Omítelo para listar las métricas disponibles |
| `--json` | `false` | Salida JSON por stdout |

Imprime una métrica y nada más, para bucles en scripts. Un campo desconocido
termina con el código `2` y remite a la forma de listado.

## profile phases

```bash
libreyolo profile phases <trace> [--json]
```

| Argumento | Por defecto | Significado |
|---|---|---|
| `trace` | | Posicional. Ruta a un perfil. Requerido |
| `--json` | `false` | Salida JSON por stdout |

Milisegundos de GPU, milisegundos de reloj, número de kernels y número de ops
por fase: forward, backward, dataload, to_device, optimizer.

## profile kernels

```bash
libreyolo profile kernels <trace> [--flag value ...]
```

| Argumento | Por defecto | Significado |
|---|---|---|
| `trace` | | Posicional. Ruta a un perfil. Requerido |
| `--top` | `20` | Muestra los N principales por tiempo de GPU |
| `--category` | | Filtra por subcadena de categoría: `gemm`, `layout`, `norm`, `elementwise` |
| `--grep` | | Filtra por expresión regular sobre el nombre del kernel |
| `--tensorcore` | `false` | Solo kernels de Tensor Core |
| `--sort` | `time` | `time`, `count` o `name` |
| `--phase` | | Limita a una sola fase: `forward`, `backward`, `dataload`, `to_device`, `optimizer` |
| `--json` | `false` | Salida JSON por stdout |

El fondo del análisis: kernels de GPU individuales con su porcentaje del tiempo
de GPU, milisegundos por paso, invocaciones por paso y categoría. Un `--phase`
desconocido termina con el código `2` y lista las fases que tiene el perfil.

## profile ops

```bash
libreyolo profile ops <trace> [--flag value ...]
```

| Argumento | Por defecto | Significado |
|---|---|---|
| `trace` | | Posicional. Ruta a un perfil. Requerido |
| `--top` | `20` | Muestra los N principales por tiempo de CPU |
| `--phase` | | Limita a una sola fase |
| `--json` | `false` | Salida JSON por stdout |

La vista del framework en lugar de la del dispositivo: ops de `aten` y de
autograd ordenadas por tiempo de CPU, que es donde aparece el coste del
lanzamiento desde el host.

## profile compare

```bash
libreyolo profile compare <before> <after> [--json]
```

| Argumento | Por defecto | Significado |
|---|---|---|
| `before` | | Posicional. Perfil de referencia. Requerido |
| `after` | | Posicional. Perfil nuevo. Requerido |
| `--json` | `false` | Salida JSON por stdout |

Compara throughput, milisegundos por imagen, utilización de la GPU, overhead
del host, lanzamientos de kernel por paso y el veredicto sobre el cuello de
botella.

El juicio de significancia necesita que ambos lados se hayan medido con un
`--repeat` de al menos 2. Con eso, una diferencia cuenta como significativa
cuando supera el doble del error estándar combinado, y la salida imprime la
comparación que hizo. Sin ello, la línea dice que una sola ejecución no puede
sostener el juicio.

## profile what-if

```bash
libreyolo profile what-if <trace> [--flag value ...]
```

| Argumento | Por defecto | Significado |
|---|---|---|
| `trace` | | Posicional. Ruta a un perfil. Requerido |
| `--remove-category` | | Proyecta la eliminación de una categoría de kernels: `gemm`, `layout`, `norm`, `elementwise` |
| `--remove-launches` | | Proyecta la eliminación de N lanzamientos de kernel por paso, por ejemplo una ganancia por fusión de ops |
| `--json` | `false` | Salida JSON por stdout |

Estima lo que aportaría un cambio antes de escribir ese cambio. Se requiere una
de las dos opciones; no indicar ninguna termina con el código `2`.

La proyección sigue el veredicto del propio perfil. Por debajo del 80% de
utilización de la GPU modela el ahorro como menos lanzamientos multiplicados
por el coste de host por lanzamiento medido; por encima, como menos trabajo de
GPU. El resultado lleva un campo de advertencia, porque el coste por
lanzamiento es una aproximación y la única prueba es una segunda medición.

## Ejemplos

<code-tabs name="examples" />

## Notas

El profiler mide e informa. No cambia nada: leer el veredicto, editar la
configuración o el código, volver a ejecutar y comparar es el bucle para el que
está construido.

`--device` vale `0` por defecto, que es el dispositivo CUDA 0. Pasar
`--device cpu` mide en la CPU y produce un perfil que los subcomandos de
lectura siguen aceptando, sin el detalle de kernels de GPU.

Todos los subcomandos admiten `--json`, y los de lectura imprimen solo por
stdout, que es lo que hace que el grupo se pueda usar desde un script.

Los códigos de salida aquí son los propios del grupo: `2` para un archivo que
no existe o un argumento que no resuelve, `3` cuando `run` no produjo ningún
perfil, y `1` cuando una traza no se puede analizar.

Relacionado: [`libreyolo train`](/docs/cli/train), cuyos argumentos son lo que
normalmente se busca ajustar cuando se toma un perfil de entrenamiento.
