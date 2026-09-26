---
title: Ensembles de detectores
seo_title: Ensembles de detectores en LibreYOLO
description: >-
  Ejecuta varios detectores sobre una misma imagen y fusiona sus boxes con
  weighted boxes fusion o NMS, incluidos modelos con listas de clases distintas.
lead: >-
  LibreEnsemble ejecuta dos o más detectores sobre la misma imagen decodificada
  y fusiona sus boxes en un único objeto Results. Cada miembro conserva sus
  propios pesos, umbrales, dispositivos y lista de clases.
keywords:
  - ensemble de modelos detección de objetos
  - weighted boxes fusion
  - wbf python
  - combinar dos detectores
  - fusionar bounding boxes
  - LibreEnsemble
  - ensemble detección python
  - min_votes
last_verified: 1.5.0
verification: >-
  Firmas del constructor y de la llamada, valores por defecto, errores de
  validación, unificación del espacio de clases, recuento de votos y el Results
  devuelto leídos de libreyolo/ensemble/model.py. Algoritmos de fusión y sus
  argumentos de libreyolo/ops/fusion.py. Intención de diseño de
  docs/adr/0004-model-ensembling.md. Patrones de uso contrastados con
  tests/unit/test_ensemble.py y tests/unit/test_ops_fusion.py.
snippets:
  basic:
    - label: 'Dos detectores, fusionados'
      language: python
      code: |
        from libreyolo import LibreEnsemble, SAMPLE_IMAGE

        # Los miembros pueden ser rutas a checkpoints o modelos ya cargados.
        ensemble = LibreEnsemble(["LibreYOLO9s.pt", "LibreRFDETRs.pt"])

        result = ensemble(SAMPLE_IMAGE)
        for xyxy, conf, cls in zip(
            result.boxes.xyxy.tolist(),
            result.boxes.conf.tolist(),
            result.boxes.cls.tolist(),
        ):
            print(result.names[int(cls)], round(float(conf), 3), xyxy)
    - label: Pesos y un requisito de votos
      language: python
      code: |
        from libreyolo import LibreEnsemble, SAMPLE_IMAGE

        ensemble = LibreEnsemble(
            ["LibreYOLO9s.pt", "LibreRFDETRs.pt"],
            weights=[1.0, 1.3],   # por convención, proporcionales al mAP de validación
            fusion="wbf",
            fusion_iou=0.55,
            min_votes=2,          # conserva solo los boxes que encontraron ambos miembros
        )

        result = ensemble(SAMPLE_IMAGE)
        print(len(result.boxes), "agreed detections")
    - label: Umbrales por miembro
      language: python
      code: >
        from libreyolo import LibreEnsemble, SAMPLE_IMAGE


        ensemble = LibreEnsemble(["LibreYOLO9s.pt", "LibreRFDETRs.pt"])


        # Un escalar se aplica a todos los miembros; una lista se lee por
        miembro.

        result = ensemble(SAMPLE_IMAGE, conf=[0.3, 0.5], iou=0.5)

        print(len(result.boxes))
  external:
    - label: Incorporar un detector que LibreYOLO no ha cargado
      language: python
      code: |
        from libreyolo import ExternalDetector, LibreEnsemble, SAMPLE_IMAGE

        def my_detector(pil_image):
            # Devuelve (boxes, scores, labels): xyxy en píxeles de la imagen original.
            return ([[100.0, 100.0, 200.0, 300.0]], [0.9], [0])

        external = ExternalDetector(my_detector, names={0: "person"})

        ensemble = LibreEnsemble(["LibreYOLO9s.pt", external])
        result = ensemble(SAMPLE_IMAGE)
        print(len(result.boxes))
  sources:
    - label: Las mismas fuentes que acepta un modelo individual
      language: python
      code: |
        from libreyolo import LibreEnsemble

        ensemble = LibreEnsemble(["LibreYOLO9s.pt", "LibreRFDETRs.pt"])

        # Sustituye clip.mp4 por un archivo de vídeo en disco.
        for result in ensemble("clip.mp4", stream=True, vid_stride=2):
            print(result.frame_idx, len(result.boxes))
source_hash: 6dcd2f84ec6f3f65
---

## Qué es un ensemble

`LibreEnsemble` toma dos o más detectores, ejecuta cada uno sobre la misma
imagen y fusiona sus boxes en un único `Results`. Es una construcción de tiempo
de predicción: no hay nada que entrenar, y los miembros siguen siendo modelos
independientes que se pueden validar y exportar por separado.

La detección es la única tarea que admite. Un miembro cuya tarea sea otra lanza
`ValueError` en la construcción, indicando el índice del miembro y su tarea.

Ambos nombres se importan de forma perezosa, así que no cuestan nada hasta que
se usan:

```python
from libreyolo import LibreEnsemble, ExternalDetector
```

## Construir uno

<code-tabs name="basic" />

```python
LibreEnsemble(
    members,
    *,
    weights=None,
    fusion="wbf",
    fusion_iou=0.55,
    min_votes=1,
)
```

`members` es una secuencia de dos o más. Una entrada `str` o `Path` se carga a
través de `LibreYOLO()`; cualquier otra cosa tiene que ser invocable y exponer
un dict `names`. Menos de dos lanza `ValueError`, y pasar una cadena suelta
lanza `TypeError` en lugar de iterar sus caracteres.

`weights` vale `None` por defecto, lo que equivale a una ponderación uniforme.
Los pesos que se pasen deben ser uno por miembro y estrictamente positivos, así
que un peso de cero lanza un error en vez de descartar un miembro en silencio.
La convención documentada es fijarlos proporcionales al mAP de validación de
cada miembro.

`fusion_iou` vale `0.55` por defecto y es el IoU al que se agrupan en un mismo
clúster los boxes de distintos miembros. Es un umbral distinto del `iou` de cada
llamada, que es el ajuste de NMS propio de cada miembro.

`min_votes` vale `1` por defecto, lo que significa que un solo miembro basta
para sostener un box. Subirlo conserva únicamente los clústeres confirmados por
ese número de miembros distintos. Tiene que ser un entero positivo no mayor que
el número de miembros, y se limita por clase al número de miembros que realmente
conocen esa clase, de modo que una clase con la que solo se entrenó un miembro
no se borra en silencio.

## Métodos de fusión

Se aceptan tres por nombre, y también se acepta un invocable.

| `fusion` | Comportamiento |
|---|---|
| `"wbf"` | Weighted boxes fusion, secuencial y fiel al paper [1]. El valor por defecto |
| `"wbf_seeded"` | Weighted boxes fusion en una sola pasada; una NMS consciente de la clase elige las semillas de los clústeres |
| `"nms"` | Concatena los boxes de todos los miembros y luego aplica una NMS consciente de la clase |

[1] Roman Solovyev, Weimin Wang, Tatiana Gabruseva, ["Weighted boxes fusion:
Ensembling boxes from different object detection models"](https://arxiv.org/abs/1910.13302),
arXiv:1910.13302.

Weighted boxes fusion promedia las coordenadas de un clúster ponderadas por la
confianza, y produce un box que ningún miembro propuso por sí solo. Las dos
variantes ponderadas coinciden siempre que los clústeres sean inequívocos y
pueden diferir ligeramente en cadenas de clústeres solapados. `"nms"` elige un
superviviente en lugar de promediar, así que los supervivientes conservan sus
puntuaciones originales y los pesos solo influyen en qué box gana. Como
selecciona en vez de agrupar, no puede contar votos: combinar `fusion="nms"` con
un `min_votes` mayor que `1` lanza `ValueError`.

Weighted boxes fusion reescala la puntuación de un clúster según la fracción del
peso de los miembros que lo respaldó. Con dos miembros ponderados por igual, un
box que solo encontró uno de ellos conserva la mitad de su puntuación: `0.9`
pasa a ser `0.45`. Por tanto, una confianza fusionada puede quedar por debajo
del `conf` con el que se ejecutó cada miembro, así que filtra por la puntuación
fusionada en lugar de dar por hecho que el umbral del miembro sigue vigente.

## Miembros con listas de clases distintas

Los miembros no tienen por qué compartir lista de clases. Sus espacios de
etiquetas se unen por nombre, y cada miembro recibe una tabla de consulta que
remapea sus propios ids de clase a la unión. `ensemble.names` es esa unión, y es
la que lleva el `Results` devuelto.

Los boxes solo se fusionan dentro del mismo nombre de clase. Una clase que solo
conoce un miembro pasa sin fusionar, y no se la penaliza por ello: el reescalado
de la puntuación usa un denominador por clase, así que una clase conocida por
uno solo conserva su puntuación.

Un solapamiento parcial registra un aviso que nombra las clases que no comparten
todos los miembros. Ese aviso es el que conviene leer con atención, porque un
checkpoint cuyos nombres de clase son marcadores de posición como `class_0`
construye una unión disjunta de la de cualquier otro miembro, y no se produce
ninguna fusión entre miembros.

Un miembro que devuelva un id de clase fuera de sus propios `names` lanza
`RuntimeError`.

## Detectores externos

<code-tabs name="external" />

`ExternalDetector(fn, names)` envuelve cualquier invocable que reciba una imagen
PIL y devuelva `(boxes, scores, labels)`, con los boxes en formato xyxy en
píxeles de la imagen original. Valida la aridad, la forma de los boxes, la
coincidencia de longitudes y que todos los ids de clase aparezcan en `names`, y
aplica él mismo el umbral `conf`.

Así es como un detector que LibreYOLO no ha cargado participa en una fusión.

## Llamarlo

<code-tabs name="sources" />

La firma de llamada refleja la de un modelo individual, y acepta las mismas
fuentes: imágenes, carpetas, listas, vídeo, captura de pantalla, webcams y
streams de red. Las fuentes en directo requieren `stream=True` por la misma
razón que en el resto de la biblioteca.

| Argumento | Por defecto | Notas |
|---|---|---|
| `conf` | `0.25` | Por miembro; un escalar se propaga, o uno por miembro |
| `iou` | `0.45` | El umbral de NMS propio de cada miembro, no el umbral de fusión |
| `imgsz` | `None` | Una `list` se lee por miembro; un `int` o una tupla se propagan |
| `device` | `None` | Escalar o uno por miembro, de modo que los miembros pueden estar en dispositivos distintos |
| `classes` | `None` | Filtra el resultado fusionado, sobre los ids de clase de la unión |
| `max_det` | `300` | Se aplica al resultado fusionado |

Como para `imgsz` una `list` significa por miembro, `imgsz=[480, 640]` es 480
para el primer miembro y 640 para el segundo, mientras que `imgsz=(480, 640)` es
un único tamaño rectangular para todos. Es una distinción fácil de pasar por
alto.

A los miembros se les llama con un `max_det` de al menos 300 independientemente
de lo que pidas, de modo que cada uno se ejecuta con holgura y el ensemble
recorta una sola vez al final.

La imagen se decodifica una sola vez y se entrega el mismo objeto a todos los
miembros. `batch` se acepta por paridad y se ignora; las imágenes se procesan de
forma secuencial.

## Qué devuelve

Un `Results` corriente, el mismo tipo que devuelve un modelo individual, con
`names` fijado al espacio de clases de la unión. Todo lo de
[Trabajar con resultados](/docs/predict/results) se aplica sin cambios.

La única diferencia es `result.speed`, que un ensemble sí rellena. Sus claves
son `member_0`, `member_1` y así sucesivamente, más `fusion`, en milisegundos.
Es el único punto de la biblioteca donde `speed` se rellena.

Las filas con boxes o puntuaciones no finitos se descartan antes de la fusión.
Cuando los miembros están en dispositivos distintos, la fusión se ejecuta en el
dispositivo del primer miembro que haya devuelto algo.

## Lo que un ensemble no puede hacer

`val()` y `export()` lanzan ambos `NotImplementedError` y te remiten a los
miembros: valida y exporta cada uno por separado. No existe ningún método
`train`, así que llamarlo lanza `AttributeError`.

La precisión media no se gestiona a nivel del ensemble. `half=True` acaba en la
misma ruta no operativa con aviso que en el resto de la biblioteca; configura la
precisión en cada miembro.

No hay interfaz de línea de comandos para los ensembles. Es una API de Python.
