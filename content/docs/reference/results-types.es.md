---
title: Tipos de Results
seo_title: "Referencia del objeto Results de LibreYOLO"
description: "Todos los payloads que puede llevar un objeto Results de LibreYOLO, un slot por forma de tarea: boxes, masks, keypoints, probs, obb, depth, ocr, embeddings y diez más."
lead: "Results es el único tipo de retorno por imagen de todos los modelos de LibreYOLO. Lleva dieciocho slots de payload opcionales, uno por forma de tarea, y solo rellena los que el modelo ha producido."
keywords:
  - objeto Results de libreyolo
  - Results.boxes
  - Results.masks
  - Results.probs
  - Results.depth_map
  - resultados de deteccion a json python
  - obtener coordenadas de bounding box python
last_verified: "1.5.0"
verification: "Nombres de slot, formas, propiedades y valores por defecto leídos de libreyolo/utils/results.py en la v1.5.0. Semántica citada de los docstrings de las clases de payload."
snippets:
  usage:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9t.pt")
        result = model(SAMPLE_IMAGE)

        print(result.orig_shape, result.path)
        print(result.boxes.xyxy)
        print(result.boxes.conf)
        print(result.names[int(result.boxes.cls[0])])
  convert:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9t.pt")
        result = model(SAMPLE_IMAGE)

        # Todos los payloads se mueven juntos.
        result = result.cpu().numpy()

        # Las filas, como dicts planos, y luego como JSON.
        print(result.summary()[:1])
        print(result.to_json())
---

## El objeto Results

Un `Results` describe una imagen. Una fuente de una sola imagen devuelve uno,
una fuente de tipo lista o un directorio devuelve una lista, y `stream=True`
devuelve un generador que los va produciendo.

| Atributo | Tipo | Significado |
|---|---|---|
| `orig_shape` | `(int, int)` | Alto y ancho de la imagen original |
| `path` | `str` | Ruta de origen cuando la entrada vino de disco |
| `names` | `dict[int, str]` | Índice de clase a nombre de clase |
| `speed` | `dict[str, float]` | Milisegundos por etapa |
| `track_id` | tensor | IDs de track cuando el resultado vino de `track()` |
| `frame_idx` | `int` | Índice de frame para fuentes de vídeo y de stream |
| `restore_scale` | `int` | Factor de escalado de salida respecto a la entrada en un resultado de restore; `1` en el resto de casos |

<code-tabs name="usage" />

## Slots de payload

Cada slot es `None` salvo que el modelo lo haya producido. El slot que rellena
cada familia lo decide su tarea.

| Slot | Clase | Tarea |
|---|---|---|
| `boxes` | `Boxes` | detect |
| `masks` | `Masks` | segment |
| `keypoints` | `Keypoints` | pose |
| `probs` | `Probs` | classify |
| `obb` | `OBB` | obb |
| `gaze` | `Gaze` | gaze |
| `points` | `Points` | point |
| `semantic_mask` | `SemanticMask` | semantic |
| `panoptic` | `PanopticSegmentation` | panoptic |
| `depth_map` | `DepthMap` | depth |
| `normal_map` | `NormalMap` | normal |
| `edges` | `EdgeMap` | edge |
| `restored` | `RestoredImage` | restore |
| `matte` | `Matte` | matte |
| `ocr` | `OCRRegions` | ocr |
| `embeddings` | `Embeddings` | embed |
| `identities` | `Identities` | embed, con una galería |
| `meshes` | `Meshes` | mesh |

`result.normals` es un alias de lectura y escritura de `result.normal_map`.

Puede haber más de un slot relleno a la vez. Un modelo de segmentación rellena
tanto `boxes` como `masks`; un modelo de gaze rellena `boxes` con los bounding
boxes de las caras y `gaze` con los ángulos; un modelo de mesh rellena `boxes`
con los bounding boxes de las personas y `meshes` alineado por filas con ellos.

## Boxes

Bounding boxes de detección de una imagen.

| Miembro | Devuelve |
|---|---|
| `xyxy` | Coordenadas de las esquinas en píxeles de la imagen original |
| `xywh` | Centro y tamaño en píxeles |
| `xyxyn` | Esquinas normalizadas a `[0, 1]` |
| `xywhn` | Centro y tamaño normalizados a `[0, 1]` |
| `conf` | Confianza por box |
| `cls` | Índice de clase por box |
| `id` | ID de track por box, o `None` |
| `is_track` | `True` cuando hay IDs de track |
| `data` | El tensor empaquetado |

`with_id(id)` y `with_orig_shape(orig_shape)` devuelven un `Boxes` nuevo con ese
campo sustituido.

## Masks

Máscaras de instancia de una imagen. `data` es el tensor de máscaras; `xy`
devuelve los contornos por instancia en píxeles y `xyn` los devuelve normalizados.

## Keypoints

Keypoints de pose, alineados por filas con `boxes`. `xy` es el par de
coordenadas por keypoint y `xyn` el par normalizado. `conf` es el tercer canal
cuando los datos lo llevan, y `None` en caso contrario. `has_visible` es un
array de booleanos, verdadero donde `conf > 0`, y todo verdadero cuando no hay
canal de confianza.

## Points

Localización de puntos de una imagen. `data` tiene forma `(N, 4)` con filas
`x, y, class, confidence`. Las coordenadas son píxeles absolutos; `xy`, `cls` y
`conf` reparten las columnas, y `xyn` normaliza las coordenadas.

## Probs

Puntuaciones de clasificación. `top1` es el índice ganador, `top5` los cinco
mejores índices, y `top1conf` y `top5conf` sus puntuaciones.

## OBB

Bounding boxes orientados. `data` contiene 7 u 8 valores por fila: `xywhr`, un
ID de track opcional, y después la confianza y la clase.

| Miembro | Devuelve |
|---|---|
| `xywhr` | Centro, tamaño y rotación en radianes |
| `xyxyxyxy` | Las cuatro esquinas en píxeles |
| `xyxyxyxyn` | Las cuatro esquinas normalizadas |
| `xyxy` | Envolvente alineada con los ejes, en píxeles |
| `conf`, `cls`, `id`, `is_track` | Igual que en `Boxes` |

## Gaze

Ángulos de mirada por cara en radianes, forma `(N, 2)`, alineados por filas con
los bounding boxes de las caras de `boxes`. La columna 0 es el pitch y la
columna 1 el yaw, según la convención L2CS: un yaw positivo gira la mirada hacia
la izquierda del sujeto y un pitch positivo la gira hacia abajo. `pitch_deg` y
`yaw_deg` convierten a grados, y `direction_3d` devuelve el vector de dirección
unitario.

## SemanticMask

Mapa semántico denso, forma `(H, W)` de IDs de clase enteros sobre el lienzo de
la imagen original. `255` es el valor de ignorar y nunca cuenta como clase
(`SemanticMask.IGNORE_INDEX`). `classes` lista los IDs de clase presentes, y
`class_mask(class_id)` devuelve la máscara booleana de una clase.

## PanopticSegmentation

Cada píxel recibe exactamente un segmento sin solapamientos, unificando las
regiones de stuff y las instancias de thing. `data` es un mapa `(H, W)` entero
de IDs de segmento; el ID de segmento `0` es sin etiquetar
(`PanopticSegmentation.IGNORE_INDEX`). `segments_info` es una lista de dicts,
uno por segmento, cada uno con al menos `{"id": int,
"category_id": int}`, donde `id` coincide con un valor del mapa y `category_id`
indexa `names`. `segment_ids` lista los IDs presentes y
`segment_mask(segment_id)` devuelve la máscara booleana de un segmento.

La distinción thing/stuff es una propiedad de la categoría, no del segmento. Un
payload puede desnormalizarla en cada segmento como `"isthing": bool`, y cuando
lo hace, el valor debe coincidir con el mapa a nivel de categoría.

## DepthMap

Mapa denso de profundidad inversa relativa, forma `(H, W)` de floats sobre el
lienzo de la imagen original. Los valores más altos significan más cerca de la
cámara. Los valores son relativos, no metros métricos. `min`, `max` y `mean` se
calculan sobre los valores finitos, y `normalized()` reescala el mapa a `[0, 1]`.

## NormalMap

Campo denso de normales de superficie, float32 `(H, W, 3)` sobre el lienzo de la
imagen original, en el sistema de cámara de OpenCV: `+x` a la derecha, `+y`
hacia abajo, `+z` hacia el interior de la escena. Las normales apuntan a la
cámara, así que una superficie fronto-paralela es `(0, 0, -1)`. Cada píxel es un
vector unitario. `assert_normalized(atol=1e-4)` comprueba esa invariante.

## EdgeMap

Mapa denso de probabilidad de borde, float32 `(H, W)` sobre el lienzo de la
imagen original, donde `0` es no-borde y `1` es borde. El mapa continuo se
conserva para que el umbral siga siendo decisión de quien llama:
`binary(threshold=0.5)` aplica uno, y `array` devuelve la vista numpy.

## RestoredImage

La imagen RGB restaurada, `(H, W, 3)` uint8. En superresolución el lienzo es
`Results.restore_scale` veces la entrada. `array` devuelve la vista numpy y
`save(path)` escribe la imagen.

## Matte

Matte de opacidad suave, float32 `(H, W)` en `[0, 1]` sobre el lienzo de la
imagen original. `1` es primer plano completo y `0` fondo completo. Un matte
suave engloba una máscara dura de eliminación de fondo, umbralizada en 0.5, y
conserva los bordes con antialiasing que una máscara binaria descarta. `array`
devuelve la vista numpy.

En un resultado de matte, `Results.cutout(image=None)` devuelve un array RGBA
`(H, W, 4)` uint8 cuyo cuarto canal es el matte, y `Results.save(path, image=None)`
escribe ese recorte como un PNG con fondo transparente. Ambos toman el RGB de
`image` cuando se les pasa, y si no lo recargan desde `Results.path`.

## OCRRegions

Texto localizado con sus transcripciones. `data` son polígonos float
`(N, 4, 2)` en píxeles de la imagen original, ordenados arriba-izquierda,
arriba-derecha, abajo-derecha, abajo-izquierda, y las regiones vienen en orden
de lectura, de arriba abajo y luego de izquierda a derecha. `texts` es la lista
de N transcripciones. `conf` es la puntuación de reconocimiento por región y
`det_conf` la de detección, ambas `(N,)`.

Los cuadriláteros de detección son polígonos de verdad, así que no rellenan
`Results.boxes`. `xyxy` da las envolventes alineadas con los ejes.

## Embeddings

Vectores normalizados con L2 de la tarea `embed`, siempre de forma `(N, D)`. Un
resultado de imagen completa lleva una sola fila y ningún box; los embeddings de
región están alineados por filas con `boxes`. Como cada fila está normalizada,
la similitud coseno es un producto escalar.

| Miembro | Devuelve |
|---|---|
| `dim` | `D` |
| `normalized` | Las filas, renormalizadas |
| `similarity(other)` | Similitud coseno por pares contra otro `Embeddings` o tensor |
| `verify(i, j, threshold=0.4)` | `True` cuando las filas `i` y `j` coinciden |

## Identities

Coincidencias con nombre de la galería, alineadas por filas con `embeddings`. Se
producen cuando se pasa una `Gallery` a una predicción `embed`. `name` es una
lista en la que una entrada es `None` por debajo del umbral de coincidencia, y
nunca se adivina el nombre más cercano que queda por debajo del umbral. `score`
es el array de puntuaciones de coincidencia y `data` los empareja.

## Meshes

Mallas paramétricas de cuerpo humano, alineadas por filas con los bounding boxes
de personas de `boxes`. Todo está en el sistema de cámara de la imagen original.
`transl` es métrico en metros con `+z` apuntando en dirección contraria a la
cámara; `vertices` y `joints3d` son métricos y ya incluyen `transl`; `joints2d`
está en píxeles sobre el lienzo de la imagen original, no sobre el recorte que
vio la red. Ningún campo lleva un sistema de mundo ni de gravedad.

La disposición de los parámetros cambia según el modelo de cuerpo, así que nada
de las formas está hard-codeado. `body_model` nombra la parametrización y los
recuentos se leen de los propios tensores: `num_vertices`, `num_joints`,
`num_betas` y `has_vertices`. `params` devuelve el dict de parámetros, y
`save_obj(path, index=0)` escribe una malla. Los campos son `global_orient`,
`body_pose`, `betas`, `transl`, `vertices`, `faces`, `joints3d`, `joints2d`,
`conf`, `focal_length` y `extras`.

Con `body_model="mhr"` las rotaciones son ángulos de Euler en radianes en lugar
de axis-angle, `body_pose` es un vector plano de parámetros por articulación en
lugar de un triplete por articulación, y `betas` son coeficientes de blendshape
de identidad. La escala del esqueleto, la pose de las manos y la expresión
facial viven en `extras`.

## Conversión y selección

Todos los payloads llevan `to(*args, **kwargs)`, `cpu()`, `cuda()` y `numpy()`,
y llamar a uno de ellos sobre el `Results` lo aplica de golpe a todos los slots
rellenos.

<code-tabs name="convert" />

`result[idx]` selecciona filas en todos los payloads alineados por filas.
`len(result)` es el número de detecciones, o de puntos cuando no hay boxes.
`result.update(...)` devuelve una copia con los slots indicados sustituidos;
acepta todos los slots más `track_id` y `restore_scale`.

## summary y to_json

`summary(normalize=False, decimals=5, embeddings=False)` devuelve una lista de
dicts planos, una fila por detección, segmento, punto o región según qué slots
estén rellenos. `to_json(**kwargs)` pasa sus argumentos a `summary` y devuelve
la cadena JSON.

`plot()` renderiza un resultado denso de normales o de bordes en su
visualización canónica; lanza una excepción para los demás tipos de resultado.
Las imágenes anotadas de las otras tareas salen de `predict(save=True)`.
