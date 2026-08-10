---
title: Trabajar con resultados
seo_title: "El objeto Results de LibreYOLO"
description: "Un objeto Results por imagen, con un slot por tipo de payload: boxes, máscaras, keypoints, probs, profundidad, panóptico, OCR y más. Dibujado, guardado y JSON."
lead: "Cada predicción devuelve un objeto Results por imagen. Tiene un slot con nombre por cada tipo de payload, todos vacíos salvo los que el modelo produce, más los mismos slots sobre un artefacto exportado."
keywords:
  - objeto results yolo python
  - results.boxes xyxy
  - convertir results a json yolo
  - guardar imagen anotada yolo
  - máscaras de segmentación python
  - keypoints results python
  - mapa de profundidad results
  - results summary yolo
  - onnx mismos resultados yolo
last_verified: "1.5.0"
verification: "Payload classes, slots, move semantics, summary(), to_json(), plot(), save() and cutout() read from libreyolo/utils/results.py. Annotation and disk-writing behavior from InferenceRunner._save_annotated_image in libreyolo/models/base/inference.py and resolve_save_path in libreyolo/utils/general.py. Suffix dispatch from LibreYOLO() in libreyolo/models/__init__.py."
snippets:
  basic:
    - label: Boxes
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")
        result = model(SAMPLE_IMAGE)

        print(result.orig_shape)   # (alto, ancho) de la imagen original
        print(result.path)         # ruta de origen, None para entrada en memoria

        for xyxy, conf, cls in zip(
            result.boxes.xyxy.tolist(),
            result.boxes.conf.tolist(),
            result.boxes.cls.tolist(),
        ):
            print(result.names[int(cls)], round(float(conf), 3), xyxy)
    - label: Coordenadas normalizadas
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy[:1])    # píxeles, x1 y1 x2 y2
        print(result.boxes.xywh[:1])    # píxeles, centro x, centro y, w, h
        print(result.boxes.xyxyn[:1])   # el mismo box dividido por el ancho y el alto
        print(result.boxes.xywhn[:1])
    - label: NumPy y dispositivos
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")
        result = model(SAMPLE_IMAGE)

        # Cada uno devuelve un nuevo Results; el original no cambia.
        as_numpy = result.numpy()
        on_cpu = result.cpu()

        print(type(as_numpy.boxes.xyxy).__name__)
        print(type(on_cpu.boxes.xyxy).__name__)
  json:
    - label: summary and to_json
      language: python
      code: |
        import json

        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")
        result = model(SAMPLE_IMAGE)

        rows = result.summary()
        print(json.dumps(rows[:2], indent=2))

        # El mismo contenido como cadena, con los mismos argumentos con nombre.
        print(result.to_json(normalize=True, decimals=3)[:200])
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt --json \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
  saving:
    - label: Imágenes anotadas
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")

        # save=True dibuja el payload y lo escribe en runs/detect/predict*.
        result = model(SAMPLE_IMAGE, save=True)
        print(result.saved_path)
  exported:
    - label: Instalar el extra de exportación
      language: bash
      code: |
        pip install "libreyolo[onnx]"
    - label: Los mismos Results desde un artefacto exportado
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")
        path = model.export(format="onnx")   # devuelve la ruta escrita

        # LibreYOLO() despacha según la extensión del archivo.
        exported = LibreYOLO(path)
        result = exported(SAMPLE_IMAGE)

        print(type(result).__name__, len(result.boxes))
---

## Un objeto, un slot por payload

Una predicción sobre una imagen devuelve un `Results`. Lleva dieciocho slots de
payload, y un modelo rellena solo los que su tarea produce. Todos los demás
quedan a `None`, así que leer `result.masks` en un detector da `None` en lugar
de un error.

| Slot | Clase | Forma | Lo produce |
|---|---|---|---|
| `boxes` | `Boxes` | `(N, 4)` más puntuaciones y clases | Detección, y cualquier tarea que localiza primero |
| `masks` | `Masks` | `(N, H, W)` | Segmentación de instancias |
| `keypoints` | `Keypoints` | `(N, K, 2)` o `(N, K, 3)` | Pose |
| `probs` | `Probs` | `(C,)` | Clasificación |
| `obb` | `OBB` | `(N, 7)` o `(N, 8)` | Bounding boxes orientados |
| `gaze` | `Gaze` | `(N, 2)` pitch y yaw en radianes | Estimación de mirada |
| `points` | `Points` | `(N, 4)` como x, y, clase, confianza | Localización de puntos |
| `semantic_mask` | `SemanticMask` | `(H, W)` ids de clase | Segmentación semántica |
| `panoptic` | `PanopticSegmentation` | `(H, W)` ids de segmento más `segments_info` | Segmentación panóptica |
| `depth_map` | `DepthMap` | `(H, W)` floats | Estimación de profundidad |
| `normal_map` | `NormalMap` | `(H, W, 3)` vectores unitarios | Normales de superficie |
| `edges` | `EdgeMap` | `(H, W)` floats en `[0, 1]` | Detección de bordes |
| `restored` | `RestoredImage` | `(H, W, 3)` uint8 RGB | Restauración y superresolución |
| `matte` | `Matte` | `(H, W)` floats en `[0, 1]` | Alpha matting y eliminación de fondo |
| `ocr` | `OCRRegions` | `(N, 4, 2)` polígonos más transcripciones | Detección y reconocimiento de texto |
| `embeddings` | `Embeddings` | `(N, D)` filas normalizadas L2 | La tarea `embed` |
| `identities` | `Identities` | N nombres y puntuaciones | La tarea `embed` con una galería |
| `meshes` | `Meshes` | Parámetros del cuerpo y vértices opcionales | Recuperación de malla corporal |

Junto a ellos están los campos que todo resultado tiene: `orig_shape` como
`(alto, ancho)`, `path` (la ruta de origen, o `None` para entrada en memoria),
`names` que mapea id de clase a nombre de clase, `frame_idx` para vídeo y
frames en vivo, `track_id` durante el tracking, y `restore_scale`, el factor
entero de escalado de un resultado de restauración.

`result.normals` es un alias de `result.normal_map`.

`result.speed` existe en todo resultado pero solo lo rellenan los
[ensembles](/docs/predict/ensembling), donde sus claves son `member_0`,
`member_1` y `fusion` en milisegundos. Para un modelo individual queda como un
dict vacío.

## Boxes

<code-tabs name="basic" />

`Boxes` guarda coordenadas y puntuaciones como arrays separados en lugar de un
único tensor empaquetado.

| Atributo | Contenido |
|---|---|
| `xyxy` | `(N, 4)` píxeles absolutos, x1 y1 x2 y2 |
| `xywh` | `(N, 4)` píxeles absolutos, centro x, centro y, ancho, alto |
| `xyxyn`, `xywhn` | Lo mismo dividido por el ancho y el alto de la imagen |
| `conf` | `(N,)` confianza |
| `cls` | `(N,)` id de clase, como float |
| `id` | `(N,)` id de track, o `None` |
| `is_track` | Si `id` está asignado |
| `data` | Todo concatenado: boxes, id opcional, conf, cls |

`cls` es un array de floats, así que úsalo como `result.names[int(cls)]`.

`xyxyn` y `xywhn` necesitan `orig_shape`, que `Results` rellena por ti.

## Payloads densos

Los payloads que cubren la imagen entera se comportan distinto de los que van
por instancia, y eso importa al hacer slicing.

`SemanticMask` guarda `(H, W)` ids de clase sobre el lienzo original, con `255`
reservado como valor de ignorado que nunca cuenta como clase. `classes` lista
los ids presentes y lo excluye; `class_mask(id)` devuelve un `(H, W)` booleano.

`PanopticSegmentation` guarda `(H, W)` ids de segmento, con `0` como id de
vacío, y una lista `segments_info` de dicts que llevan al menos `id` y
`category_id`. `segment_ids` lista los ids presentes, `segment_mask(id)`
selecciona uno.

`DepthMap` guarda `(H, W)` de profundidad inversa relativa: un valor mayor
significa más cerca, y los valores no son metros métricos. Expone `min`, `max`
y `mean` sobre los valores finitos, y `normalized()` que reescala a `[0, 1]`.

`NormalMap` guarda `(H, W, 3)` vectores unitarios en el sistema de cámara de
OpenCV, con `+x` a la derecha, `+y` hacia abajo y `+z` hacia dentro de la
escena, de modo que una superficie que mira a la cámara es `(0, 0, -1)`.
`assert_normalized()` comprueba que cada píxel es finito y de longitud
unitaria.

`EdgeMap` guarda `(H, W)` float32 en `[0, 1]`. Se conserva el mapa continuo en
lugar de aplicarle un umbral, así que `binary(threshold=0.5)` es donde eliges
el punto de corte.

`Matte` guarda `(H, W)` float32 en `[0, 1]`, donde `1` es totalmente primer
plano. `array` lo devuelve recortado como float32.

`RestoredImage` guarda `(H, W, 3)` uint8 RGB, con `array` para el ndarray en
crudo y `save(path)` para escribirlo.

`Probs` guarda un vector de probabilidad para la imagen. `top1` y `top5` son
índices de clase, `top1conf` y `top5conf` las puntuaciones correspondientes.

`Embeddings` guarda `(N, D)` filas ya normalizadas L2, así que la similitud
coseno es un producto escalar. `similarity(other)` devuelve `(N, M)` contra una
galería o `(N,)` contra un vector individual, y `verify(i, j, threshold=0.4)`
compara dos filas.

`OCRRegions` guarda `(N, 4, 2)` polígonos en orden de lectura, con las esquinas
ordenadas superior-izquierda, superior-derecha, inferior-derecha,
inferior-izquierda. Las transcripciones están en `texts`, las puntuaciones de
reconocimiento en `conf`, las de detección en `det_conf`. Como son polígonos
genuinamente rotados no rellenan `boxes`; `ocr.xyxy` da las envolventes
alineadas a los ejes cuando necesitas rectángulos.

## Slicing y movimiento

`result[i]` devuelve un nuevo `Results` con una sola instancia. Los payloads
por instancia se recortan; los payloads de imagen completa pasan sin cambios,
así que hacer slicing de un resultado de clasificación no puede truncar su
vector de probabilidad a una sola clase, y hacer slicing de un resultado de
profundidad no puede corromper la disposición `(H, W)`.

`len(result)` cuenta instancias: boxes, puntos, embeddings, regiones OCR o
mallas. Cualquier payload denso de imagen completa cuenta como `1`. Un
resultado sin nada dentro es `0`.

`to()`, `cpu()`, `cuda()` y `numpy()` devuelven cada uno un nuevo `Results` con
todos los slots rellenos convertidos. No modifican el original.

`update()` es el único método que muta in situ, reemplazando los slots
indicados y devolviendo el mismo objeto.

## JSON

<code-tabs name="json" />

`summary()` devuelve una lista de dicts planos, y `to_json()` es esa lista
pasada por `json.dumps`. Ambos aceptan los mismos tres argumentos:
`normalize=False` cambia las coordenadas a `[0, 1]`, `decimals=5` fija el
redondeo, y `embeddings=False` controla si se incluyen los vectores de
embedding.

La forma de cada fila sigue al payload. Las filas de detección llevan `name`,
`class`, `confidence` y un dict `box`, y ganan `segments` cuando hay máscaras,
`obb` y `corners` para boxes orientados, ángulos `gaze` en radianes y grados,
`track_id` durante el tracking, y parámetros `mesh` cuando hay mallas.

Donde no hay boxes, un solo payload decide las filas: OCR emite una fila por
región con su `text`, puntos una fila por punto, panóptico una fila por
segmento con `pixel_count` y `pixel_fraction`, semántico una fila por clase
presente, clasificación las cinco clases principales. Profundidad, normales,
bordes, restauración y matting emiten cada uno una única fila resumen que
describe el mapa en lugar de sus píxeles.

Dos payloads se abrevian deliberadamente. Un vector de embedding se reporta
solo como `embedding_dim`, porque una fila de 512 floats son unos 2 KB por
cara; pasa `embeddings=True` para incluir los valores. Los vértices de malla no
se incluyen nunca, ya que son decenas de miles de coordenadas por persona. Lee
`result.meshes.vertices` o llama a `result.meshes.save_obj(path)` para la
geometría.

## Dibujar y guardar

<code-tabs name="saving" />

`predict(save=True)` es la vía que anota y escribe. Elige la rutina de dibujo
según el slot que esté relleno, así que un resultado semántico se escribe como
máscara coloreada, un resultado de profundidad como visualización de
profundidad, un resultado panóptico con sus segmentos, un matte como PNG RGBA
con fondo transparente, y un detector como boxes con las máscaras debajo. La
ruta escrita se adjunta al resultado como `result.saved_path`.

`Results.plot()` es más estrecho de lo que su nombre sugiere. Está definido
solo para mapas de normales y mapas de bordes, y lanza `NotImplementedError`
para cualquier otra cosa. Usa `save=True` para el resto de tareas.

`Results.save(path)` es igual de estrecho: escribe un resultado de matte como
recorte PNG RGBA con fondo transparente y lanza `NotImplementedError` en los
demás casos. `Results.cutout()` devuelve ese mismo array RGBA sin escribirlo.
Ambos necesitan la imagen de origen, tomada de `result.path` o pasada como
`image=`.

Dos payloads llevan sus propios escritores: `result.restored.save(path)` para
una imagen restaurada, y `result.meshes.save_obj(path, index=0)` para una
malla.

Para saber dónde acaban los archivos y cómo se comportan `output_path` y
`output_file_format`, mira [Fuentes de predicción](/docs/predict/sources).

## Los artefactos exportados devuelven el mismo objeto

<code-tabs name="exported" />

`LibreYOLO()` despacha según la extensión del archivo, así que un artefacto
exportado se carga con la misma llamada que un checkpoint `.pt` y devuelve el
mismo `Results`. Los archivos `.onnx`, `.engine`, `.pte` y `.mnn` se reconocen
por la extensión, igual que los directorios de OpenVINO, Paddle y ncnn y una
URL de modelo de Triton. El código que lee `result.boxes.xyxy` no cambia cuando
un modelo se sustituye por su build exportada. Consulta
[Exportación](/docs/export) para el conjunto completo de formatos.

Recurrir en su lugar a la API propia del runtime significa encargarte tú mismo
del preprocesado, el postprocesado y los nombres de clase.
