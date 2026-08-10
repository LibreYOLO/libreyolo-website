---
title: Configuración
seo_title: "Variables de entorno y directorios de LibreYOLO"
description: "Todas las variables de entorno que lee LibreYOLO, los directorios en los que escribe, los tokens que necesita y los interruptores que cambian qué ruta de código se ejecuta."
lead: "LibreYOLO no tiene archivo de configuración. El comportamiento que no es un argumento de función se controla mediante variables de entorno y mediante un pequeño número de directorios convencionales, todos listados aquí."
keywords:
  - LIBREYOLO_DATASETS_DIR
  - LIBREYOLO_KERNELS
  - LIBREYOLO_FASTER_COCO_EVAL
  - HF_TOKEN
  - variables de entorno libreyolo
  - directorio de pesos libreyolo
  - caché de libreyolo
last_verified: "1.5.0"
verification: "Variables localizadas buscando os.environ y os.getenv en libreyolo/**/*.py en la v1.5.0; semántica leída en cada punto de uso. Convenciones de directorios leídas de libreyolo/data/utils.py, libreyolo/utils/download.py, libreyolo/export/exporter.py, libreyolo/models/base/model.py y libreyolo/models/sam3dbody/mhr_body.py."
snippets:
  usage:
    - label: Apuntar la raíz de datasets a otro sitio
      language: bash
      code: |
        export LIBREYOLO_DATASETS_DIR=/data/datasets
        python -c "from libreyolo.data import DATASETS_DIR; print(DATASETS_DIR)"
    - label: Leer el valor resuelto desde Python
      language: python
      code: |
        from libreyolo.data import DATASETS_DIR

        # Por defecto es ~/datasets; LIBREYOLO_DATASETS_DIR lo sobrescribe al importar.
        print(DATASETS_DIR)
---

## Variables de entorno

| Variable | Valor por defecto | Efecto |
|---|---|---|
| `LIBREYOLO_DATASETS_DIR` | `~/datasets` | Raíz de datasets. Se lee una sola vez al importar, en `libreyolo.data.DATASETS_DIR` |
| `LIBREYOLO_FASTER_COCO_EVAL` | sin definir | Sobrescribe el flag de validación `faster_coco_eval`. `1`, `true`, `yes` u `on` fuerza el backend rápido; cualquier otro valor lo desactiva; sin definir, se delega en el flag de la configuración |
| `LIBREYOLO_KERNELS` | sin definir | Selección de kernels. `off` o `reference` fuerza las implementaciones de referencia; cualquier otro valor selecciona solo las implementaciones registradas bajo ese nombre |
| `LIBREYOLO_QUANT_KERNELS` | sin definir | Alias heredado de `LIBREYOLO_KERNELS`, se lee solo cuando aquella está sin definir |
| `LIBREYOLO_HUB_KERNELS` | sin definir | `0`, `false`, `off` o `no` desactiva la carga de kernels desde Hugging Face Hub. Cualquier otro valor, incluido sin definir, la deja activada |
| `LIBREYOLO_MHR_PATH` | `~/.cache/libreyolo/mhr/mhr_model.pt` | Ubicación del modelo corporal MHR que usa la tarea `mesh` |
| `LIBRELABEL_ENABLE_LOCATE` | sin definir | Tiene que ser exactamente `1`, `true`, `yes` u `on` para exponer el asistente LocateAnything en la herramienta de etiquetado. Cualquier otro valor lo mantiene desactivado |
| `SAM_3D_BODY_PATH` | sin definir | Ruta al paquete SAM 3D Body para la familia mesh, cuando no se pasa al constructor |
| `HF_TOKEN` | sin definir | Token de acceso de Hugging Face, usado para repositorios restringidos |

<code-tabs name="usage" />

`LIBREYOLO_DATASETS_DIR` se lee al importar, así que definirla después de
importar `libreyolo.data` no tiene efecto sobre `DATASETS_DIR`.

Los kernels del Hub se activan en dos partes. La descarga en tiempo de ejecución
solo ocurre cuando el paquete opcional `kernels` está instalado, así que
instalar `libreyolo[hub-kernels]` es la activación y `LIBREYOLO_HUB_KERNELS=0`
la desactivación. Una instalación sin ese extra no se ve afectada en ningún caso.

La selección de kernels también cortocircuita los imports: cuando
`LIBREYOLO_KERNELS` fuerza `off` o `reference`, los proveedores acelerados
incluidos en el árbol no se importan en absoluto. El registro que controlan
estas tres variables está documentado en
[kernels](/docs/reference/kernels).

## Variables que define la biblioteca

Estas se escriben en lugar de leerse, así que definirlas a mano no es la vía
soportada.

| Variable | Definida por |
|---|---|
| `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT` | El ayudante de spawn de DDP, un valor por cada proceso worker |
| `CUDA_VISIBLE_DEVICES` | Se restringe temporalmente durante la configuración distribuida y luego se restaura |
| `PYTORCH_ENABLE_MPS_FALLBACK` | Los entrenadores EC la ponen a `1`, con `setdefault`, de modo que un valor ya existente gana |
| `MOMENTUM_ENABLED` | La define con `setdefault` el cargador de la familia mesh |

`LOCAL_RANK` hace también de señal de modo distribuido: su presencia en el
entorno es la forma en que el código de entrenamiento detecta que se está
ejecutando bajo DDP.

## Variables de los loggers

Los loggers opcionales de entrenamiento recurren a valores por defecto del
entorno para el nombre del proyecto.

| Variable | Valor por defecto | Usada por |
|---|---|---|
| `WANDB_PROJECT` | `libreyolo` | El logger de Weights and Biases, cuando no se pasa ningún proyecto |
| `COMET_PROJECT_NAME` | `libreyolo` | El logger de Comet, cuando no se pasa ningún proyecto |

La autenticación en esos servicios sigue sus propias herramientas, no las de
LibreYOLO.

## Tokens

`HF_TOKEN` es el token de acceso de Hugging Face. Cuando está sin definir, el
token se lee de `~/.cache/huggingface/token`, que es donde lo escribe un login
con la CLI de Hugging Face. Ambas vías funcionan.

Solo hace falta un token para repositorios restringidos. SAM 3 es el ejemplo
que viene de serie: sus pesos se descargan de un repositorio restringido bajo
una licencia propia, así que hay que aceptar los términos en la página del
repositorio y la sesión tiene que estar autenticada.

## Directorios

| Ruta | Contenido |
|---|---|
| `weights/` | Checkpoints descargados, snapshots de Hugging Face descargados y artefactos exportados |
| `~/datasets` | Raíz de datasets, salvo que `LIBREYOLO_DATASETS_DIR` indique otra cosa |
| `~/.cache/huggingface/token` | Token de Hugging Face, cuando no está en `HF_TOKEN` |
| `~/.cache/libreyolo/mhr/mhr_model.pt` | Modelo corporal MHR, salvo que `LIBREYOLO_MHR_PATH` indique otra cosa |
| `runs/track/` | Salida por defecto de `model.track(save=True)` |

`weights/` es relativo al directorio de trabajo. Un nombre de archivo a secas se
resuelve a través de él, así que `LibreYOLO("LibreYOLO9t.pt")` busca
`weights/LibreYOLO9t.pt` y descarga ahí cuando no está. `model.export()` escribe
en ese mismo directorio cuando no se indica `output_path`. Los niveles hermanos
descargan snapshots de varios archivos en `weights/<Prefix><size>/`.

## Comportamiento de las descargas

Las descargas de pesos se reintentan tres veces con backoff, se reanudan desde
un archivo parcial y están protegidas por un archivo de bloqueo para que dos
procesos no descarguen el mismo checkpoint a la vez. Una familia que descarga
desde un host de terceros puede fijar un checksum y abortar ante una
discrepancia.

Algunas descargas imprimen un aviso de licencia antes de empezar. Esos avisos
forman parte de la ruta de descarga y no se pueden silenciar mediante
configuración.

## Backend de validación

`model.val()` acepta `faster_coco_eval=True` por defecto y recurre a
pycocotools cuando el paquete no está instalado, avisando una sola vez. Definir
`LIBREYOLO_FASTER_COCO_EVAL` sobrescribe el flag de cada llamada, que es lo que
debería usar un banco de benchmarks que no puede tocar la configuración de cada
ejecución. El backend que realmente se usó se indica en
`model.last_eval_backend`.

## Scripts de descarga de datasets

Un YAML de dataset puede llevar un campo `download` con código Python. No se
ejecuta salvo que se pase `allow_download_scripts=True` a la llamada que lo lee,
que es un argumento de función en `val()` y `export()` en lugar de una variable
de entorno.
