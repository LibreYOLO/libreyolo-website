---
title: API de ensemble
seo_title: "API de LibreEnsemble y operaciones de fusión"
description: "LibreEnsemble, ExternalDetector y las tres operaciones de fusión de libreyolo.ops: weighted boxes fusion, su variante con semillas y la fusión por NMS por clase."
lead: "LibreEnsemble ejecuta varios detectores sobre la misma imagen y fusiona sus detecciones en un único Results. La fusión ocurre después del postprocesado propio de cada miembro, así que cada uno conserva su tamaño de entrada, su normalización y su supresión."
keywords:
  - LibreEnsemble
  - ensemble de detectores de objetos
  - weighted boxes fusion python
  - ExternalDetector
  - libreyolo.ops.fusion
  - consenso min_votes
last_verified: "1.5.0"
verification: "Firmas y valores por defecto leídos de libreyolo/ensemble/model.py y libreyolo/ops/fusion.py en la v1.5.0. Intención de diseño según docs/adr/0004-model-ensembling.md."
snippets:
  usage:
    - label: Dos miembros, fusión por defecto
      language: python
      code: |
        from libreyolo import LibreEnsemble, SAMPLE_IMAGE

        ens = LibreEnsemble(["LibreYOLO9t.pt", "LibreYOLO9s.pt"])

        # Una fuente de una sola imagen devuelve un Results, no una lista.
        result = ens(SAMPLE_IMAGE, conf=0.25)

        print(result.boxes.xyxy)
        print(result.speed)
    - label: Consenso y umbrales por miembro
      language: python
      code: |
        from libreyolo import LibreEnsemble, SAMPLE_IMAGE

        ens = LibreEnsemble(
            ["LibreYOLO9t.pt", "LibreYOLO9s.pt"],
            weights=[1.0, 2.0],
            fusion="wbf",
            fusion_iou=0.55,
            min_votes=2,
        )
        result = ens(SAMPLE_IMAGE, conf=[0.25, 0.4])
        print(len(result))
  ops:
    - label: Operación de fusión, sin modelo de por medio
      language: python
      code: |
        import torch
        from libreyolo.ops import weighted_boxes_fusion

        boxes = torch.tensor([[10.0, 10.0, 50.0, 50.0], [12.0, 11.0, 51.0, 49.0]])
        scores = torch.tensor([0.9, 0.8])
        labels = torch.tensor([0, 0])
        model_ids = torch.tensor([0, 1])

        fused = weighted_boxes_fusion(
            boxes, scores, labels, model_ids, num_models=2, iou_thr=0.55
        )
        print(fused)
---

## LibreEnsemble

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

| Argumento | Por defecto | Significado |
|---|---|---|
| `members` | | Dos o más detectores |
| `weights` | `None` | Factores de fiabilidad por miembro; todos `1.0` si se omite |
| `fusion` | `"wbf"` | `"wbf"`, `"wbf_seeded"`, `"nms"` o un callable |
| `fusion_iou` | `0.55` | Umbral de IoU para el clustering de la fusión |
| `min_votes` | `1` | Conserva solo los boxes confirmados por al menos este número de miembros |

Un miembro es una ruta de pesos que se resuelve a través de la factoría
`LibreYOLO()`, un modelo ya construido, un backend exportado o un
`ExternalDetector`. Todos los miembros deben ser modelos de la tarea de detección.

<code-tabs name="usage" />

La construcción rechaza menos de dos miembros, una lista `weights` de longitud
incorrecta, un peso no positivo, un `min_votes` que no sea un entero positivo y
un `min_votes` mayor que el número de miembros. `fusion="nms"` con
`min_votes > 1` también lanza un error, porque NMS descarta la pertenencia a los
clústeres y no puede contar votos.

`weights` escala la fiabilidad que se atribuye a cada miembro. Un peso más alto
acerca las coordenadas y las puntuaciones fusionadas a ese miembro. La convención
es hacerlos proporcionales al mAP de validación.

## Espacios de clases

Los miembros con `names` idénticos pasan directamente. En caso contrario, los
espacios de clases se unen por nombre, los IDs de clase de cada miembro se
remapean mediante tablas de correspondencia y el `Results.names` fusionado es la
unión. La fusión combina boxes solo dentro de una misma clase unificada, así que
una clase que solo conoce un miembro pasa sin fusionar. Una discrepancia registra
un aviso en la construcción.

`min_votes` se limita por clase según cuántos espacios de etiquetas de los
miembros contienen esa clase, de modo que el consenso siga teniendo sentido con
vocabularios parcialmente compartidos.

## Llamar al ensemble

```python
ens(
    source=None,
    *,
    conf=0.25,
    iou=0.45,
    imgsz=None,
    device=None,
    classes=None,
    max_det=300,
    augment=False,
    save=False,
    output_path=None,
    color_format="auto",
    batch=1,
    stream=False,
    stream_buffer=False,
    vid_stride=1,
    show=False,
    **kwargs,
)
```

`predict` es un alias de `__call__`. El retorno es el `Results` habitual, cuyo
`speed` desglosa el coste por miembro y añade una entrada `fusion`. Una fuente de
una sola imagen devuelve uno solo, una lista o un directorio devuelven una lista
y `stream=True` devuelve un generador.

`conf`, `iou` y `device` se propagan a todos los miembros y también aceptan un
valor por miembro, así que `conf=[0.25, 0.4]` le da al miembro 0 un umbral de
0.25 y al miembro 1 un umbral de 0.4. `imgsz` se propaga cuando es un int o una
tupla y es por miembro solo cuando es una lista, así que `imgsz=(480, 640)` es un
único tamaño rectangular para todos, mientras que `imgsz=[480, 640]` es 480 para
el miembro 0 y 640 para el miembro 1. Cada entrada debe ser válida para la
familia de ese miembro.

`augment` se propaga a los miembros que admiten aumento en tiempo de test
(test-time augmentation), y los backends exportados lo ignoran. `classes` toma
IDs de clase de la unión y `max_det` se aplica al resultado fusionado, así que
los miembros trabajan con holgura y el ensemble recorta una sola vez. `batch` se
acepta por paridad de API; las imágenes se procesan de forma secuencial.

`val()` y `export()` lanzan `NotImplementedError`. Valida y exporta los miembros
por separado.

## ExternalDetector

```python
ExternalDetector(fn: Callable, names: dict[int, str])
```

Adapta cualquier callable de detección para convertirlo en un miembro. `fn`
recibe una imagen PIL y devuelve `(boxes, scores, labels)`, donde los boxes son
xyxy en píxeles de la imagen original y las etiquetas son IDs de clase válidos en
`names`. Funcionan tanto tensores como arrays y listas anidadas. LibreYOLO no
importa nada del código externo.

El adaptador valida el retorno: debe ser una tupla de tres elementos, los boxes
deben tener forma `(N, 4)`, los tres arrays deben tener la misma longitud y todos
los IDs de clase deben aparecer en `names`. Las detecciones iguales o inferiores
a `conf` se descartan antes de la fusión.

## Operaciones de fusión

Las primitivas de fusión son operaciones de torch independientes que viven en
`libreyolo.ops`. No dependen de ningún modelo y se pueden importar por su cuenta,
y por eso se exportan aparte del ensemble.

<code-tabs name="ops" />

Las tres reciben los mismos argumentos posicionales, `boxes, scores, labels,
model_ids`, y devuelven `(boxes, scores, labels)`.

| Operación | Clave del registro | Comportamiento |
|---|---|---|
| `weighted_boxes_fusion` | `wbf` | Weighted boxes fusion secuencial, fiel al paper |
| `wbf_seeded` | `wbf_seeded` | Variante paralela en una sola pasada de la misma reducción |
| `nms_fusion` | `nms` | Concatena todo y aplica NMS por clase |

`FUSIONS` asocia las tres claves del registro con los callables, y
`LibreEnsemble` resuelve ahí el valor de `fusion=`.

```python
weighted_boxes_fusion(
    boxes, scores, labels, model_ids,
    *,
    weights=None,
    num_models=None,
    iou_thr=0.55,
    skip_box_thr=0.0,
    conf_type="avg",
    min_votes=1,
    models_per_label=None,
    label_weights=None,
)
```

`wbf_seeded` tiene una firma idéntica. `nms_fusion` recibe los mismos argumentos
salvo `conf_type`, y lanza `ValueError` cuando `min_votes > 1`.

En `weighted_boxes_fusion`, las detecciones se recorren en orden decreciente de
confianza escalada por el peso. Cada una se une al clúster existente con cuyo box
fusionado provisional solapa mejor, con un IoU por encima de `iou_thr` y la misma
etiqueta, o bien abre un clúster nuevo. El box fusionado de un clúster es la
media de las coordenadas de sus miembros ponderada por la confianza, y su
puntuación es la media ponderada o el máximo de sus confianzas, reescalada para
que los boxes confirmados por menos modelos puntúen más bajo.

`wbf_seeded` elige las semillas de los clústeres con NMS por clase a `iou_thr`,
asigna cada detección a la semilla de su misma etiqueta con la que tiene mejor
IoU y después reduce cada clúster de la misma forma. Las formas de los clústeres
nunca cambian a mitad de pasada, así que toda la operación es aritmética de
tensores de forma fija. Las dos variantes coinciden siempre que los clústeres son
inequívocos y pueden diferir ligeramente en cadenas de clústeres solapados.

`nms_fusion` conserva sin cambios el box de mayor confianza de cada grupo
solapado. Los `weights` por modelo escalan las confianzas solo para el ranking de
supresión, y los boxes que sobreviven conservan sus puntuaciones originales.

## Fusión personalizada

`fusion=` también acepta un callable con la misma firma que las operaciones
anteriores. Su nombre queda registrado en `ens.fusion`, o `"custom"` si no tiene
ninguno. El retorno se valida: debe ser una tripleta `(boxes, scores, labels)`
con formas coherentes.
