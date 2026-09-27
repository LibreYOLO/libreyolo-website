---
title: Conceptos básicos
seo_title: Conceptos básicos de LibreYOLO
description: >-
  Cómo encajan las tareas, las familias de modelos, los tamaños y los nombres de
  archivo de los checkpoints en LibreYOLO, y qué promete cada nivel de soporte.
lead: >-
  Cuatro ideas describen todos los modelos de LibreYOLO: la tarea que realiza,
  la familia a la que pertenece, el tamaño dentro de esa familia y el nivel de
  soporte en el que está la familia. El nombre de archivo del checkpoint
  codifica las tres primeras.
keywords:
  - conceptos libreyolo
  - tareas libreyolo
  - familias de modelos libreyolo
  - nombres de checkpoints libreyolo
  - niveles de soporte libreyolo
  - tipos de tareas vision por computador
last_verified: 1.5.0
meta:
  - label: Esquema del nombre de archivo
    value: 'Libre<FAMILY><size>[-<task>].pt'
    mono: true
  - label: Tareas canónicas
    value: 17
  - label: Niveles de soporte
    value: 'Flagship, Core, Supported, Inference only, Museum, Sibling tier'
snippets:
  inspect:
    - label: Listar familias
      language: bash
      code: >
        # Tareas, tamaños y resoluciones de entrada de todas las familias
        registradas.

        libreyolo models
    - label: Un modelo
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        print(model.family, model.size, model.task)
        print(model.input_size)
        print(model.nb_classes, model.names[0])
    - label: Elegir una tarea
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Los alias se normalizan en la frontera de la API: "keypoints" se
        resuelve

        # como "pose", "det" como "detect", "semantic-segmentation" como
        "semantic".

        model = LibreYOLO("LibreYOLO9t.pt", task="det")

        print(model.task)
source_hash: 23d045463a6a8411
---

## Tareas

Una tarea es lo que devuelve un modelo. LibreYOLO tiene diecisiete nombres
canónicos de tarea, y cada uno da nombre al campo del objeto `Results` que
transporta su salida.

| Tarea | Devuelve |
|---|---|
| `detect` | Bounding boxes alineados con los ejes, con una clase y una confianza |
| `segment` | Máscaras por instancia, una máscara por objeto detectado |
| `semantic` | Una etiqueta de clase por píxel, sin separación por instancias |
| `panoptic` | Una etiqueta no solapada por píxel, que fusiona los elementos contables con el material amorfo |
| `pose` | Keypoints por instancia, con las filas alineadas con los boxes |
| `classify` | Una probabilidad sobre un conjunto de etiquetas para la imagen entera |
| `obb` | Boxes orientados, con un ángulo de rotación |
| `point` | Una coordenada de imagen por detección, en lugar de un box |
| `depth` | Un mapa denso de profundidad inversa relativa |
| `normal` | Un campo denso de normales de superficie en vectores unitarios |
| `edge` | Un mapa denso de probabilidad de bordes |
| `restore` | Una imagen RGB restaurada, para eliminar desenfoque, quitar ruido o superresolución |
| `matte` | Un mapa suave de primer plano de 0 a 1, para eliminar el fondo |
| `ocr` | Cuadriláteros de texto con sus transcripciones, en orden de lectura |
| `embed` | Un vector normalizado con L2 cuyo producto escalar mide la coincidencia |
| `gaze` | Una dirección de mirada por cara detectada |
| `mesh` | Un cuerpo 3D con pose por persona detectada |

Esos son los nombres que aparecen en los metadatos de los checkpoints y en los
nombres de archivo. Se aceptan los alias habituales allá donde se pasa una tarea,
y se normalizan antes de cualquier otra cosa: `detection` y `det` se convierten
en `detect`, `keypoints` se convierte en `pose`, `cls` se convierte en
`classify`, y `deblur`, `denoise` y `super-resolution` se convierten todos en
`restore`; `face-recognition` y `reid` se convierten en `embed`. Un nombre no
reconocido lanza un error en lugar de recurrir en silencio a un valor por
defecto.

`segment`, `semantic` y `panoptic` son tres tareas distintas, no tres palabras
para una sola. Las máscaras de instancia, las etiquetas por píxel y el mapa
fusionado de elementos y material tienen distinto ground truth, distintas
métricas y distintos campos de resultado.

## Familias de modelos

Una familia es un linaje de arquitectura con su propio código de carga,
preprocesado y postprocesado. Cada familia declara un identificador `FAMILY`
como `yolo9`, `rfdetr` o `dfine`, las tareas que soporta y la resolución de
entrada de cada tamaño que publica.

`LibreYOLO()` es una fábrica, no una clase. Dada una ruta, carga el archivo,
identifica la familia a partir de los metadatos del checkpoint o, si no los hay,
a partir de las propias claves de los tensores, y devuelve una instancia del
modelo de esa familia. Por eso cambiar de detector es una modificación de una
línea: el objeto que se obtiene expone la misma superficie de `predict`, `train`,
`val` y `export`, y devuelve el mismo tipo `Results`.

<code-tabs name="inspect" />

Una familia que sirve más de una tarea suele publicar un checkpoint distinto por
tarea, a menudo con un conjunto de tamaños diferente para cada una; unas pocas,
en cambio, comparten un mismo artefacto entre dos tareas de ejecución. En
cualquier caso, las tareas soportadas son una lista fija, y pedir una que esté
fuera de ella lanza un error con la lista soportada en el mensaje, en lugar de
cargar algo aproximado.

La lista completa, con benchmarks por familia y los pesos publicados, está en
[todos los modelos](/docs/models).

## Tamaños

Un tamaño es una variante dentro de una familia, escrita como un código en
minúsculas pegado directamente al prefijo de la familia. Las letras habituales
son `n` de nano, `t` de tiny, `s` de small, `m` de medium, `l` de large y `x` de
xlarge, pero los códigos son específicos de cada familia y varias familias usan
algo completamente distinto: códigos con el nombre del backbone como `r50` o
`r101`, donde el tamaño es una profundidad de ResNet; códigos de escalado
compuesto como de `b0` a `b3`; o un nombre que identifica el único checkpoint
publicado. YOLOv9 usa `c` de compact donde otras familias usan `l`.

El tamaño también fija la resolución de entrada, y en las familias con varias
tareas la resolución puede variar según la tarea. Ambas se leen de la familia,
nunca se dan por supuestas; `libreyolo models` las imprime.

## Nombres de archivo de los checkpoints

Todos los archivos de pesos publicados siguen un único esquema:

```text
Libre<FAMILY><size>[-<task>].pt
```

El prefijo de familia es una cadena fija por familia, el tamaño va en minúsculas
y pegado sin separador, y el sufijo de tarea lleva un guion delante. La detección
no lleva sufijo, siguiendo la convención que los checkpoints YOLO han usado
siempre, así que `LibreYOLO9t.pt` es un detector y `LibreRFDETRn-seg.pt` es un
modelo de segmentación de la misma familia.

| Tarea | Sufijo |
|---|---|
| `detect` | |
| `segment` | `-seg` |
| `semantic` | `-sem` |
| `panoptic` | `-panoptic` |
| `pose` | `-pose` |
| `classify` | `-cls` |
| `gaze` | `-gaze` |
| `obb` | `-obb` |
| `point` | `-point` |
| `depth` | `-depth` |
| `edge` | `-edge` |
| `normal` | `-normal` |
| `restore` | `-restore` |
| `matte` | `-matte` |
| `ocr` | `-ocr` |
| `embed` | `-embed` |
| `mesh` | `-mesh` |

Una familia que no tenga ninguna tarea sin sufijo puede exigir el sufijo, de modo
que un nombre sin él no se acepta como checkpoint válido para ella. Una familia
que publica pesos entrenados con un dataset distinto del suyo por defecto añade
el nombre del dataset como un sufijo adicional, y esa variante sigue formando
parte del nombre del repositorio desde el que se descarga el archivo.

Tres niveles quedan fuera de este esquema. Las familias de segmentación
promptable, las familias de visión y lenguaje y los detectores de vocabulario
abierto no están registrados en la fábrica de checkpoints y no emiten ningún
archivo `Libre<FAMILY><size>.pt`. Su prefijo nombra en su lugar un snapshot
descargado de Hugging Face o un checkpoint promptable, y ahí se conserva a
propósito el uso de mayúsculas de la marca original.

## Cómo se decide la tarea

Cuando varias señales podrían dar nombre a una tarea, se consultan en un orden
fijo y gana la primera que esté presente: el argumento `task` que hayas pasado,
luego la tarea registrada en los metadatos del checkpoint, luego el sufijo de
tarea del nombre de archivo, y luego la tarea por defecto de la familia. El
resultado se comprueba contra las tareas soportadas por la familia antes de
construir el modelo, así que un desajuste falla en el momento de la carga en
lugar de producir una salida incorrecta más adelante.

## Niveles de soporte

Las familias están inscritas en exactamente un nivel. Un nivel es una afirmación
sobre la atención de ingeniería, no sobre la precisión: te dice dónde aterriza
primero una función nueva y qué se mantiene en verde.

| Nivel | Qué significa |
|---|---|
| Flagship | Las funciones se diseñan y se validan por completo en GPU aquí primero |
| Core | Detectores entrenables principales. Las funciones siguen a los flagships en la misma oleada de releases |
| Supported | Familias entrenables de apoyo. Se mantienen en verde en CI, las funciones llegan de forma oportunista |
| Inference only | Predicción, validación y exportación. Las funciones de entrenamiento no aplican |
| Museum | Una pieza de museo congelada. Solo correcciones de bugs |
| Sibling tier | Una superficie de producto aparte, con su propia fábrica y su propio contrato |

Cada página de modelo lleva el nivel de su familia en la cabecera. Las dos
familias flagship son [YOLOv9](/docs/models/yolov9) para los detectores CNN y
[RF-DETR](/docs/models/rf-detr) para los detectores transformer; empieza por ahí
salvo que tengas un motivo para no hacerlo.

Inference only dice lo que falta, que es un bucle de entrenamiento en LibreYOLO.
La predicción, la validación y, donde la familia lo soporte, la exportación
funcionan todas. Llamar a `train()` en una familia así lanza un
`NotImplementedError` que nombra el motivo.
