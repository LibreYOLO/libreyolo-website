---
title: Umbrales y filtrado
seo_title: "conf, iou y max_det en LibreYOLO"
description: "Qué hacen realmente conf, iou, max_det y classes en la predicción, qué familias ignoran iou porque no ejecutan NMS y por qué agnostic_nms es un no-op."
lead: "Cuatro argumentos deciden qué predicciones sobreviven: conf, iou, max_det y classes. Solo dos de ellos se aplican a todas las familias, porque un predictor de conjuntos decodifica un conjunto fijo de queries y nunca ejecuta NMS."
keywords:
  - umbral de confianza yolo
  - conf yolo python
  - umbral iou nms
  - max_det yolo
  - filtrar clases deteccion python
  - agnostic nms
  - detr sin nms
  - filtrado por clases inferencia
last_verified: "1.5.0"
verification: "Valores por defecto tomados de InferenceRunner.__call__ en libreyolo/models/base/inference.py. Comportamiento de NMS por familia leído de todos los módulos de libreyolo/postprocess/ y contrastado con _is_nms_free_family en libreyolo/backends/base.py. Filtrado por clases de InferenceRunner._apply_classes_filter y _wrap_results. Estado de agnostic_nms de NOOP_PREDICT_KWARGS en libreyolo/utils/predict_args.py. Gestión de vocabulario abierto de NMS_THRESHOLD en libreyolo/models/openvocab/base.py. Valores por defecto de la validación de BaseModel.val."
snippets:
  basic:
    - label: Los cuatro argumentos
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")

        result = model(
            SAMPLE_IMAGE,
            conf=0.25,      # conserva las predicciones con esta puntuación o más
            iou=0.45,       # umbral de solape de NMS, donde se ejecuta NMS
            max_det=300,    # tope por imagen
            classes=None,   # o una lista de ids de clase
        )
        print(len(result.boxes))
    - label: Barrido de conf
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")

        for conf in (0.1, 0.25, 0.5, 0.75):
            result = model(SAMPLE_IMAGE, conf=conf)
            print(conf, len(result.boxes))
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt conf=0.4 iou=0.5 max_det=100 \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
  classes:
    - label: Filtrar a clases concretas
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")

        # Los ids de clase indexan model.names. En COCO, 0 es person.
        result = model(SAMPLE_IMAGE, classes=[0])

        print({result.names[int(c)] for c in result.boxes.cls.tolist()})
    - label: Encontrar el id de un nombre
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")
        result = model(SAMPLE_IMAGE)

        wanted = {"person", "backpack"}
        ids = [i for i, name in result.names.items() if name in wanted]
        print(ids)

        filtered = model(SAMPLE_IMAGE, classes=ids)
        print(len(filtered.boxes))
  nmsfree:
    - label: iou en una familia que no ejecuta NMS
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # RF-DETR decodifica un conjunto fijo de queries, así que iou no cambia nada aquí.
        model = LibreYOLO("LibreRFDETRs.pt")

        loose = model(SAMPLE_IMAGE, iou=0.9)
        tight = model(SAMPLE_IMAGE, iou=0.1)

        # El mismo recuento en ambos casos. conf y max_det son los controles que funcionan.
        print(len(loose.boxes), len(tight.boxes))
---

## Los cuatro argumentos

| Argumento | Por defecto | Se aplica a |
|---|---|---|
| `conf` | `0.25` | Todas las familias |
| `iou` | `0.45` | Familias que ejecutan non-maximum suppression |
| `max_det` | `300` | Todas las familias |
| `classes` | `None` | Todas las familias |

<code-tabs name="basic" />

Dos de ellos son universales y dos no, que es lo más útil que puedes saber antes
de ajustar nada.

La validación usa valores por defecto distintos a propósito: `val()` se ejecuta
con `conf=0.001` e `iou=0.6`, porque la precisión media se calcula sobre una
curva precision-recall completa y un corte en 0.25 la truncaría.

## conf

`conf` es la puntuación por debajo de la cual se descarta una predicción. Se
aplica a todas las familias, incluidas las que nunca ejecutan NMS, y es el primer
control al que recurrir cuando hay demasiadas o demasiado pocas detecciones.

El valor por defecto de `0.25` sirve para mirar imágenes. Alimentar a un sistema
posterior suele pedirlo más alto; medir la precisión lo pide mucho más bajo.

## iou

`iou` es el solape por encima del cual la non-maximum suppression elimina el de
menor puntuación de dos boxes de la misma clase. Solo significa algo si la
familia llega a ejecutar supresión.

Un predictor de conjuntos decodifica un número fijo de queries y se queda con las
de mayor puntuación. Los duplicados se suprimen dentro de la arquitectura durante
el entrenamiento, no en un paso de postprocesado, así que no hay ningún umbral
que tocar. Estas familias aceptan `iou` por paridad de API y lo ignoran:

CenterNet, DEIM, DETR, Deformable DETR, D-FINE, DINO-DETR, EdgeCrafter,
Faster R-CNN, LW-DETR, Mask R-CNN, RF-DETR, RT-DETR y la cabeza end-to-end de
YOLOv9. Las variantes construidas sobre esos decodificadores heredan el
comportamiento.

<code-tabs name="nmsfree" />

La mayoría lo indica en los docstrings de su postprocesado, pero en tiempo de
ejecución no se lanza ningún aviso, así que un barrido de `iou` sobre RF-DETR
produce una línea plana en lugar de un error. Faster R-CNN y Mask R-CNN son un
caso algo distinto: ambos ya han ejecutado NMS dentro del modelo, con un umbral
fijo aguas arriba que `iou` no tiene forma soportada de cambiar.

Estas familias sí lo usan: de YOLOv1 a YOLOv4, YOLOv7, YOLOv9, YOLOX,
YOLO-NAS, RTMDet, PicoDet, EfficientDet, FCOS, RetinaNet y SSD.

Dos opciones del momento de la predicción hacen que `iou` importe incluso para un
predictor de conjuntos, porque ambas fusionan boxes después de que el modelo haya
terminado:

- `tiling=True` reconcilia los tiles solapados con NMS por clase en `iou`
- `augment=True` fusiona las vistas volteadas con NMS por clase en `iou`

Ambas se explican en [Rendimiento de la inferencia](/docs/predict/performance).

Los detectores de vocabulario abierto tienen su propia regla. Una familia cuyo
procesador ejecuta NMS declara su propio umbral por defecto y respeta `iou`, que
es el caso de OMDet-Turbo. Las familias que no suprimen nada, Grounding DINO,
OWLv2 y OV-DEIM, emiten un aviso cuando se les pasa `iou`. Ese aviso es el único
de su tipo en la biblioteca.

## max_det

`max_det` limita cuántas predicciones se devuelven para una imagen. Se aplica en
todas partes, pero por mecanismos distintos: una familia con NMS trunca después
de la supresión, y un predictor de conjuntos lo usa como el tamaño de su
selección top-k.

Algunas familias recortan por debajo de lo que pidas, porque su configuración de
referencia original lo hace. SSD se queda en 200, la segmentación de instancias
de RTMDet en 100 y FCOS en su propio límite de detecciones por imagen. Subir
`max_det` por encima de esos valores no tiene efecto.

El único sitio donde `max_det` se aplica de forma centralizada en lugar de por
familia es la inferencia por tiles, donde la lista fusionada se trunca después de
reconciliar los tiles.

## Filtrado por clases

<code-tabs name="classes" />

`classes` recibe una lista de ids de clase y conserva solo las predicciones cuya
clase esté en ella. Los ids indexan `result.names`, y la forma más segura de
obtener uno es leer `names` de un resultado en lugar de dar por supuesto el orden
de un dataset.

El filtrado ocurre de forma centralizada, después del postprocesado de cada
familia, en el único embudo por el que pasan todos los caminos de predicción. Eso
tiene dos consecuencias que conviene conocer. Funciona en todas las familias,
incluidas las que no tienen NMS. Y además filtra los payloads alineados con los
boxes, de modo que las máscaras, los keypoints y los boxes orientados se recortan
junto a ellos en lugar de quedar descuadrados.

En la línea de comandos, `classes` acepta un entero suelto, una lista o una
cadena separada por comas:

```bash
libreyolo predict model=LibreYOLO9s.pt classes=0 source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
libreyolo predict model=LibreYOLO9s.pt classes="[0,2,5]" source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
```

Filtrar no es precisión gratis. El modelo sigue gastando su presupuesto
prediciendo clases que luego descartas, y la familia aplica `max_det` antes del
filtro, así que una imagen abarrotada de clases no deseadas puede llegar al tope
antes de alcanzar tu clase. Baja `conf` o sube `max_det` si eso pasa.

## agnostic_nms

`agnostic_nms` se acepta y no hace nada. Pasarlo lanza un aviso que dice que es un
no-op por compatibilidad con la línea de comandos, y el argumento se descarta.

No existe un modo de supresión agnóstico a la clase. Todas las llamadas a NMS de
la biblioteca tienen en cuenta la clase, así que dos boxes solapados de clases
distintas sobreviven ambos, sea cual sea el `iou`. Cuando eso sea un problema,
filtra antes con `classes`, o haz tú mismo la supresión entre clases sobre
`result.boxes`.

## Qué rechaza predict

Dos argumentos lanzan un error en vez de un aviso: `visualize` y `embed` lanzan
ambos `NotImplementedError`. Para los embeddings, carga el modelo con
`task="embed"` y llama a `predict` o a `embed` con normalidad.

Cualquier cosa no reconocida lanza un `TypeError` que nombra las opciones
soportadas, de modo que una errata falla al momento en lugar de ignorarse en
silencio.

Estos se aceptan, avisan y se descartan: `agnostic_nms`, `boxes`, `dnn`,
`half`, `line_width`, `retina_masks`, `show_conf`, `show_labels` y `verbose`.
