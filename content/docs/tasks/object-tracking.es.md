---
title: Seguimiento de objetos
seo_title: "Seguimiento de objetos en LibreYOLO"
description: "Sigue objetos a lo largo de los fotogramas de un vídeo en LibreYOLO con ByteTrack, BoT-SORT, OC-SORT o Deep OC-SORT, sobre cualquier modelo de detección, segmentación o pose."
lead: "El seguimiento asigna una identidad estable a cada detección a lo largo de los fotogramas de un vídeo. LibreYOLO no lo modela como una tarea con pesos propios: es un modo de predicción, model.track(), que ejecuta el tracker elegido sobre la salida por fotograma de un modelo de detección, segmentación o pose."
keywords: [seguimiento de objetos python, tracking multiobjeto video, bytetrack, botsort, ocsort, deep ocsort, track id yolo, seguimiento con reid]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # track() es un generador: un Results por fotograma procesado.
        for result in model.track("video.mp4"):
            print(result.track_id)        # tensor int (N,), alineado con boxes
            print(result.boxes.xyxy)
    - label: Elegir un tracker
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # "bytetrack" (por defecto), "botsort", "ocsort" o "deepocsort".
        for result in model.track("video.mp4", tracker="botsort"):
            print(result.track_id)
    - label: Guardar un vídeo anotado
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # Sin output_path, el fichero acaba en runs/track/<video_stem>.mp4.
        for result in model.track("video.mp4", save=True, vid_stride=2):
            pass
    - label: Ajustar un tracker
      language: python
      code: |
        from libreyolo import BoTSortConfig, LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # El tipo de config selecciona el tracker, así que tracker= sobra aquí.
        config = BoTSortConfig(track_buffer=60, frame_rate=25, enable_cmc=False)
        for result in model.track("video.mp4", tracker_config=config):
            print(result.track_id)

        # O pasa los mismos campos como argumentos y deja que track() la construya.
        for result in model.track("video.mp4", tracker="botsort", track_buffer=60):
            print(result.track_id)
---

## Definición

El seguimiento no es una de las claves de tarea de LibreYOLO, y no hay ningún
checkpoint de seguimiento que descargar. Es un método del modelo,
`model.track(source)`, que ejecuta la detección en cada fotograma y asocia los
resultados a lo largo del tiempo. El método es un generador: entrega un
`Results` por fotograma procesado, con `result.track_id` fijado a un tensor de
enteros `(N,)` alineado con `result.boxes`. Los mismos IDs están también en
`result.boxes.id`.

Solo se entregan los objetos confirmados y actualmente seguidos. Un track que la
asociación pierde sigue vivo durante un número configurado de fotogramas antes de
descartarse, `track_buffer` para ByteTrack y BoT-SORT y `max_age` para las dos
variantes de OC-SORT, de modo que un objeto recuperado dentro de esa ventana
conserva su ID original.

Como la asociación ocurre después de la detección, el resto de contenidos del
fotograma sobreviven: el `Results` con seguimiento es el `Results` de detección
recortado a las filas emparejadas, así que las máscaras y los keypoints llegan
junto con los boxes.

## Modelos

En una ejecución de seguimiento entran dos decisiones independientes: el modelo
que produce los boxes en cada fotograma, y el tracker que los enlaza.

Cualquier modelo nativo de LibreYOLO cuya tarea sea detección, segmentación o
pose expone `track()`, así que la elección del detector es la de siempre. Consulta
[el índice de modelos](/docs/models) para ver la lista completa, o empieza por
[YOLO9](/docs/models/yolov9), [RF-DETR](/docs/models/rf-detr),
[D-FINE](/docs/models/d-fine) o [RTMDet](/docs/models/rtmdet). Las tareas cuyos
resultados no tienen ningún box que asociar rechazan la llamada en vez de
devolver IDs sin sentido: clasificación, boxes orientados, puntos, profundidad,
normales de superficie, bordes, segmentación semántica y panóptica, restauración,
OCR y malla corporal lanzan todas una excepción desde `track()`.

Dos de los niveles de modelo de LibreYOLO también lo rechazan. Los modelos
cargados con `LibreSAM` son segmentadores de imagen, y los modelos cargados con
`LibreOpenVocab` son detectores por fotograma; ambos lanzan una excepción desde
`track()` y se usan con `predict()` fotograma a fotograma.

El seguimiento se ejecuta sobre modelos nativos de PyTorch. Un artefacto
exportado que se carga con `LibreYOLO("model.onnx")` devuelve un objeto de
backend de runtime, que trae `predict()` pero no `track()`.

La biblioteca incluye cuatro trackers, seleccionables con el argumento `tracker`:

`"bytetrack"` es el que viene por defecto. Es solo de movimiento, con un filtro
de Kalman y una asociación en tres etapas: primero las detecciones de confianza
alta, luego una segunda pasada que da a las detecciones de confianza baja la
oportunidad de emparejarse con un track existente antes de descartarlas, y por
último los tracks sin confirmar. Se configura con `TrackConfig`.

`"botsort"` mantiene el ciclo de vida en tres etapas de ByteTrack, pero usa un
estado de Kalman de centro-anchura-altura y compensa el movimiento de la cámara
en los tracks predichos antes de emparejar. Esta es la variante de BoT-SORT solo
de movimiento; no ejecuta ningún modelo de apariencia. Se configura con
`BoTSortConfig`, que añade `enable_cmc`, `cmc_method` y `cmc_downscale`.

`"ocsort"` también es solo de movimiento, y añade un término de dirección de la
velocidad al coste de asociación, una segunda pasada de asociación contra la
última observación real de cada track, y un suavizado del estado de Kalman a lo
largo de una trayectoria virtual cuando se vuelve a encontrar un track. Se
configura con `OCSortConfig`.

`"deepocsort"` extiende OC-SORT con apariencia. Cada track mantiene una media
móvil ponderada por confianza de embeddings de reidentificación, y un término de
similitud coseno se suma al coste de asociación, de modo que las identidades
sobreviven a oclusiones largas y a objetivos que se cruzan. Cuesta un forward por
fotograma de una pequeña red de embeddings, y sus pesos de OSNet se descargan la
primera vez que se usa. Se configura con `DeepOCSortConfig`.

## Predicción

<code-tabs name="predict" />

`track_conf` fija el umbral de la primera etapa de asociación:
`track_high_thresh` para ByteTrack y BoT-SORT, `det_thresh` para OC-SORT y
Deep OC-SORT. No es el `conf` de `predict()`, y para ByteTrack, BoT-SORT y
OC-SORT el detector se ejecuta internamente con un umbral más bajo para que las
detecciones débiles sigan disponibles para la pasada de recuperación. Deep
OC-SORT ejecuta el propio detector con `det_thresh`. Para ByteTrack y BoT-SORT,
`track_conf` debe ser igual o mayor que `track_low_thresh`, cuyo valor por
defecto es 0.1.

Los ajustes del tracker llegan de una de dos maneras. Pasa una instancia de
config a `tracker_config=`, y su tipo selecciona el tracker, con lo que `tracker=`
sobra. O pasa los campos como argumentos con nombre y deja que `track()`
construya la config del tracker que hayas nombrado; las claves desconocidas
avisan en vez de aplicarse en silencio. En cualquiera de los dos casos,
`track_conf` se ignora en cuanto la clave correspondiente se fija explícitamente.

El resto de argumentos son los mismos que en predicción: `iou`, `imgsz`,
`classes`, `max_det`, `vid_stride`, `show`, y `save` con `output_path`. La fuente
es la ruta a un fichero de vídeo. Consulta [predicción](/docs/predict) para el
manejo de resultados.

## Entrenamiento

Los trackers no se entrenan. Tres de los cuatro son modelos de movimiento puros,
sin ningún parámetro aprendido, y la red de apariencia de Deep OC-SORT es un
checkpoint publicado de reidentificación que se descarga la primera vez que se
usa. Mejorar la calidad del seguimiento significa mejorar el detector, o ajustar
los umbrales de asociación de arriba.
