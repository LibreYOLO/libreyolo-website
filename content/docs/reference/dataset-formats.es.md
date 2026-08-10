---
title: Formatos de dataset
seo_title: "Formatos de dataset de LibreYOLO para cada tarea"
description: "El contrato de archivos de dataset por tarea canónica: claves YAML, estructura de carpetas, filas de etiquetas, convenciones de máscaras y mapas, y el loader que lee cada uno."
lead: "Esta página refleja el contrato de archivos de dataset del propio docs/dataset_schema.md de la biblioteca. Cubre las claves YAML y la estructura en disco que espera cada tarea canónica."
keywords:
  - formato dataset libreyolo
  - formato etiquetas yolo
  - data.yaml
  - dataset máscaras segmentación
  - formato coco panoptic
  - dataset profundidad
  - pose kpt_shape
last_verified: "1.5.0"
verification: "Refleja docs/dataset_schema.md del repositorio libreyolo en la v1.5.0, con los nombres de los loaders verificados contra libreyolo/data/."
snippets:
  usage:
    - label: Parsear una fila de etiqueta de detección
      language: python
      code: |
        from libreyolo.data import parse_yolo_label_line

        # class_id cx cy w h, normalizado a [0, 1]
        row = parse_yolo_label_line("0 0.5 0.5 0.25 0.5", 640, 480, num_classes=80)

        # (class_id, x1, y1, x2, y2, area) en píxeles
        print(row)
---

## YAML común

Se aplica a `detect`, `segment`, `pose` y `obb`.

| Clave | Obligatoria | Significado |
|---|---|---|
| `path` | | Raíz del dataset |
| `train` | Para entrenar | Imágenes de entrenamiento |
| `val` | Para validar | Imágenes de validación |
| `test` | | Imágenes de test |
| `names` | Sí | Lista de clases, o un mapeo con claves enteras |
| `nc` | | Número de clases; debe coincidir con `names` cuando está presente |
| `download` | | Instrucciones de descarga; los scripts de Python requieren activación explícita |
| `annotations` | | Split a archivo COCO JSON nativo, para detect, segment y obb |

`train`, `val` y `test` pueden ser directorios de imágenes, archivos `.txt` con
listas de imágenes, o listas de esos elementos. Las rutas de etiquetas siguen
una única sustitución:

```text
images/.../image.jpg -> labels/.../image.txt
```

Para un dataset COCO JSON nativo, `annotations` asocia cada split a su archivo
JSON y la ruta del split da la raíz de imágenes:

```yaml
path: dataset
train: images/train
val: images/val
annotations:
  train: annotations/train.json
  val: annotations/val.json
```

Cuando `names` está presente, los nombres de categoría del COCO JSON nativo
deben coincidir con los nombres de clase del YAML, y esos nombres definen los
IDs de etiqueta del modelo. Sin `names`, los IDs de categoría de COCO se ordenan
y se mapean de forma densa a `0..N-1`.

Un YAML de dataset no lleva una clave `task`. La selección explícita de modelo y
tarea tiene prioridad.

Reglas comunes a todos los archivos de etiquetas de texto:

- un archivo de etiquetas `.txt` por imagen;
- un archivo de etiquetas ausente o vacío significa que no hay objetos;
- `class_id` es un entero en `0..nc-1`;
- las coordenadas son floats normalizados y finitos en `[0, 1]`;
- las coordenadas son relativas al ancho y al alto de la imagen original;
- las filas no llevan confianza ni ID de seguimiento.

<code-tabs name="usage" />

## detect

Exactamente cinco campos por fila:

```text
<class_id> <cx> <cy> <w> <h>
```

`cx cy w h` es un box normalizado alineado a los ejes, y `w` y `h` deben ser
positivos.

## segment

Una fila de polígono:

```text
<class_id> <x1> <y1> ... <xN> <yN>
```

`N` es como mínimo 3, el número de coordenadas después de `class_id` debe ser
par, y el polígono no puede ser degenerado. También se acepta una fila de
detección de cinco campos, que representa un segmento rectangular.

## pose

El YAML añade `kpt_shape`, que es obligatoria y vale `[K, 2]` o `[K, 3]`, y la
opcional `flip_idx`, una permutación entera de `0..K-1`.

```text
<class_id> <cx> <cy> <w> <h> <k1x> <k1y> [<k1v>] ... <kKx> <kKy> [<kKv>]
```

El número de campos es exactamente `5 + K * D`, donde `D` es el segundo valor
de `kpt_shape`. Las coordenadas de los keypoints están normalizadas. La
visibilidad `v`, cuando está presente, es `0`, `1` o `2`.

## obb

Exactamente nueve campos:

```text
<class_id> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4>
```

Los cuatro puntos son coordenadas de imagen normalizadas en `[0, 1]` y forman un
rectángulo orientado no degenerado. En el archivo de etiquetas no se guarda
ningún ángulo.

El parser canónico es estricto por defecto y rechaza las coordenadas fuera de
rango. La ingesta de dataset y de validación puede recortar las coordenadas a
`[0, 1]` en etiquetas por lo demás válidas que quedan en el borde del recorte, y
aun así rechaza los boxes degenerados. El parseo depende de la tarea: nueve
campos significan `obb` solo en modo `obb`, mientras que en modo `segment` pueden
ser un polígono de cuatro puntos.

Internamente, las esquinas normalizadas se convierten al `xywhr` canónico, con
el ángulo en radianes representando la rotación del lado del ancho alrededor del
centro del box. Los resultados públicos exponen las detecciones OBB como filas
`xywhr, conf, cls`.

La carga de OBB desde COCO JSON nativo acepta anotaciones en este orden de
prioridad: `obb` como ocho esquinas en espacio de píxeles; `obb` como
`[cx, cy, w, h, angle]` con el ángulo en radianes; un polígono o RLE de
`segmentation` de COCO, reajustado a un rectángulo de área mínima; y un `bbox`
de COCO, leído como alineado a los ejes y canonicalizado.

Mosaic y mixup están desactivados en el entrenamiento OBB hasta que exista un
aumento de datos para OBB que tenga en cuenta las esquinas.

El parser canónico de filas es `libreyolo.data.parse_yolo_obb_label_line`.

## semantic

Cada imagen se empareja con una máscara densa de un solo canal en un formato sin
pérdidas, normalmente PNG, en lugar de un archivo `.txt`:

```text
images/.../image.jpg -> <masks_dir>/.../image.png
```

La máscara es de un solo canal, y los PNG en modo paleta se leen como índices de
paleta. Cada valor de píxel es un ID de clase en `0..nc-1`, el valor de píxel
`255` significa ignorar y queda excluido de la loss (la función de pérdida) y de
las métricas, y la resolución de la máscara debe ser igual a la de la imagen.

Dos claves YAML opcionales se apoyan sobre el contrato común. `masks_dir` es el
nombre del directorio de máscaras que sustituye a `images` en cada ruta de
imagen, y por defecto es `masks`. `label_mapping` es un remapeo
`{source_id: train_id}` aplicado a los valores de píxel de la máscara en el
momento de la carga, donde los valores de origen sin mapear pasan a ignorarse y
los train IDs deben caer en `0..nc-1`.

Cuando se omite `masks_dir`, las máscaras se rasterizan en el momento de la
carga a partir de las etiquetas de polígono de `segment`, resueltas mediante la
convención de `images` a `labels`, y se añade una clase `background` después de
las clases de objeto, de modo que `nc` crece en uno.

Loader canónico: `libreyolo.data.SemanticDataset`.

## panoptic

LibreYOLO adopta el formato COCO-panoptic tal cual (Kirillov et al., CVPR
2019). No existe un formato panóptico específico de LibreYOLO.

Un PNG RGB por imagen, a la resolución de la imagen, codifica en su color el ID
de segmento de cada píxel:

```text
segment_id = R + 256 * G + 256 * 256 * B
```

Cada píxel pertenece exactamente a un segmento y los segmentos nunca se solapan.
El ID de segmento `0`, negro RGB, es void: píxeles sin etiquetar que se excluyen
de la métrica.

```json
{
  "images":      [{"id": 139, "file_name": "000000000139.jpg"}],
  "annotations": [{"image_id": 139, "file_name": "000000000139.png",
                   "segments_info": [
                     {"id": 3226956, "category_id": 1, "area": 2840,
                      "bbox": [413, 158, 53, 138], "iscrowd": 0}]}],
  "categories":  [{"id": 1, "name": "person", "isthing": 1, "supercategory": "person"}]
}
```

`annotations[].file_name` nombra el PNG de IDs de segmento dentro de
`panoptic_dir`, y `segments_info[].id` coincide con un valor de ese PNG.
`iscrowd` marca regiones de grupo: nunca son falsos negativos, y una predicción
que cubre una de ellas en su mayor parte no es un falso positivo.

Thing frente a stuff es una propiedad por categoría. `isthing` vive en
`categories`, nunca en `segments_info`.

Los valores de `category_id` de COCO-panoptic son los IDs originales del dataset
y normalmente no son contiguos. Los modelos predicen `0..nc-1` de forma
contigua, así que los IDs originales se remapean a través de los `names` del
YAML por nombre de categoría, la misma regla que sigue el loader de detect de
COCO JSON nativo. Una categoría del JSON ausente de `names` es un error, y no un
descarte silencioso, porque de lo contrario puntuaría como un falso negativo
permanente.

```yaml
path: coco
val: images/val2017
annotations:
  val: annotations/panoptic_val2017.json
panoptic_dir:
  val: annotations/panoptic_val2017
names: {0: person, 1: bicycle, 132: rug-merged}
```

`annotations` y `panoptic_dir` aceptan o bien una única ruta o bien un mapeo por
split.

La validación reporta Panoptic Quality, calculada a la resolución del ground
truth y promediada sobre las categorías que aparecen, y después separada en
`PQ_things` y `PQ_stuff`. El emparejamiento es único: un segmento predicho y uno
del ground truth de la misma categoría se emparejan cuando el IoU supera 0.5.

Loader canónico: `libreyolo.data.PanopticDataset`.

## depth

Cada imagen se empareja con un mapa de profundidad denso de un solo canal:

```text
images/.../image.jpg -> <depths_dir>/.../image.png
```

El mapa es un PNG o TIF de un solo canal, o un archivo `.npy`, a la resolución
de la imagen. Los valores son profundidad directa en una unidad consistente
dentro del dataset. Los valores cero, negativos, NaN e infinitos marcan píxeles
inválidos y se excluyen de la loss y de las métricas.

| Clave | Por defecto | Significado |
|---|---|---|
| `depths_dir` | `depths` | Directorio de profundidad que sustituye a `images` |
| `depth_stem_suffix` | | Sufijo añadido al stem de la imagen; cuando se omite se prueban tanto el mismo stem como un sufijo `_depth` |
| `depth_mask_suffix` | `_mask` | Sufijo para una máscara de validez; los valores de máscara iguales o menores que cero, NaN e infinitos invalidan el píxel de profundidad |
| `depth_scale` | `256.0` | Divisor para los mapas de profundidad de tipo entero, la convención habitual de PNG de 16 bits |

Los mapas `.npy` en coma flotante se usan tal cual y no aplican `depth_scale`.

Loader canónico: `libreyolo.data.DepthDataset`.

## edge

Cada imagen RGB se empareja con un mapa sin pérdidas de un solo canal con el
mismo stem y una máscara de validez opcional:

```text
images/val/scene.jpg -> edges/val/scene.png
                     -> masks/val/scene.png
```

El mapa es un PNG o TIF de un solo canal, no una visualización RGB, a la
resolución de la imagen. Los mapas de tipo entero se dividen por el máximo de su
dtype; los mapas en coma flotante deben ser ya finitos y estar en `[0, 1]`. `0`
significa no-borde y `1` significa borde. Los píxeles de la máscara opcional son
válidos cuando son distintos de cero. El redimensionado usa interpolación de
vecino más cercano para los targets y las máscaras, y los píxeles de padding son
inválidos y no contribuyen a la validación.

| Clave | Por defecto | Significado |
|---|---|---|
| `edges_dir` | `edges` | Directorio de mapas de bordes que sustituye a `images` |
| `edge_stem_suffix` | | Sufijo añadido a los stems de las imágenes |
| `edge_extension` | `.png` | Extensión del target sin pérdidas |
| `edge_invert` | | Ponlo a true cuando los mapas de origen guardan bordes negros sobre blanco |
| `masks_dir` | `masks` | Directorio opcional de máscaras de validez |

```yaml
path: edge-dataset
train: images/train
val: images/val
edges_dir: edges
masks_dir: masks
nc: 1
names: {0: edge}
```

La validación adelgaza las predicciones continuas con non-maximum suppression de
gradiente en cuatro direcciones y reporta las F-measure ODS y OIS sobre un
barrido de umbrales configurable. Los píxeles predichos y los del ground truth
se emparejan uno a uno dentro de `edge_max_dist * image_diagonal`, con una
tolerancia normalizada por defecto de `0.0075`.

Loader canónico: `libreyolo.data.EdgeDataset`. El loader se limita al formato:
no descarga ni redistribuye datos de benchmark.

## normal

Cada imagen se empareja con un PNG de tres canales y 16 bits con el mismo stem,
más una máscara de validez opcional con el mismo stem:

```text
images/val/room.jpg -> normals/val/room.png
                    -> masks/val/room.png
```

El PNG es exactamente `uint16` de tres canales, con los canales guardados como
RGB, a la resolución de la imagen. Decodifícalo con `n = png / 65535 * 2 - 1` y
después renormaliza cada vector. Los vectores decodificados usan el sistema de
cámara de OpenCV, `+x` a la derecha, `+y` hacia abajo, `+z` hacia el interior de
la escena, y apuntan hacia la cámara. La máscara opcional es un PNG de un solo
canal donde un valor distinto de cero significa válido; sin máscara, todo vector
decodificado finito y distinto de cero es válido. Los píxeles de target
inválidos y los de padding se representan internamente con `(0, 0, 0)`. El
redimensionado interpola bilinealmente las tres componentes y después
renormaliza, las máscaras de validez usan interpolación de vecino más cercano, y
un volteo horizontal además niega la componente x.

| Clave | Por defecto | Significado |
|---|---|---|
| `normals_dir` | `normals` | Directorio de mapas de normales que sustituye a `images` |
| `masks_dir` | `masks` | Directorio opcional de máscaras de validez |

La validación reporta el error angular medio y mediano en grados y el porcentaje
de píxeles válidos dentro de 11.25, 22.5 y 30 grados.

Loader canónico: `libreyolo.data.NormalDataset`.

## restore

Cada imagen de entrada degradada se empareja con un target RGB limpio:

```text
inputs/.../image.jpg -> targets/.../image.jpg
```

La entrada y el target son archivos de imagen compatibles con RGB y sus
resoluciones deben coincidir exactamente. La validación mantiene la resolución
nativa y solo aplica el padding justo para apilar un batch, y las métricas se
calculan sobre el lienzo de la imagen original. El entrenamiento aplica un
recorte y un volteo horizontal acoplados al par de entrada y target.

| Clave | Por defecto | Significado |
|---|---|---|
| `input_dir` | `inputs` | Directorio de entradas degradadas usado en las rutas de los splits |
| `target_dir` | `targets` | Directorio de targets limpios que sustituye a `input_dir` |
| `target_stem_suffix` | | Sufijo añadido al stem de la entrada antes de buscar el target |
| `target_stem_suffixes` | | Forma de lista de `target_stem_suffix` |
| `degradation` | | Etiqueta de metadatos como `deblur` o `denoise` |
| `dataset` | | Etiqueta de dataset o de procedencia |

Los campos YAML de tipo clase son marcadores de posición del esquema: usa
`nc: 1` y `names: {0: image}`. Los modelos de restore exponen
`Results.restored`, no detecciones.

Loader canónico: `libreyolo.data.RestoreDataset`.

## matte

Cada imagen RGB se empareja con un matte de ground truth de un solo canal que
comparte el mismo stem, donde 0 es fondo y 255 es primer plano:

```text
images/subject.jpg -> mattes/subject.png
```

Se aceptan dos estructuras. Una raíz de directorio que contenga `images/` y un
directorio de mattes, detectado automáticamente entre `mattes/`, `matte/`,
`gt/`, `masks/`, `mask/` y `alpha/`, pasada como `data=`. O un YAML con `path`
más `val_images` y `val_mattes` por split, y opcionalmente `train_images` y
`train_mattes`, cada uno relativo a `path` o absoluto.

El matte está en escala de grises y se lee como opacidad en `[0, 1]`, y se
redimensiona al lienzo de la predicción con interpolación bilineal cuando las
formas difieren. Las métricas son MAE y S-measure (Fan et al., ICCV 2017) sobre
el lienzo de la imagen original, con S-measure como fitness del mejor
checkpoint.

Los campos YAML de tipo clase son marcadores de posición del esquema: usa
`nc: 1` y `names: {0: matte}`. Los modelos de matte exponen `Results.matte`.

En esta versión la validación es solo de inferencia. Resolutor canónico de
pares: `libreyolo.data.matte_dataset.resolve_matte_pairs`.

## ocr

Las etiquetas son un archivo JSONL por split, con un objeto JSON por imagen:

```text
images/val/receipt.jpg -> labels/val.jsonl
```

```json
{"image": "receipt.jpg", "regions": [{"polygon": [[10, 12], [118, 14], [117, 40], [9, 38]], "text": "TOTAL 12.50"}]}
```

`polygon` es un cuadrilátero de cuatro puntos en coordenadas de píxel absolutas,
ordenados arriba-izquierda, arriba-derecha, abajo-derecha, abajo-izquierda. Las
regiones con texto ilegible usan `"text": "###"`, la convención do-not-care de
ICDAR: quedan excluidas de la puntuación de reconocimiento, y las predicciones
que se solapan con ellas se ignoran en lugar de penalizarse en el emparejamiento
de detección.

Las métricas son el hmean de detección con emparejamiento uno a uno de polígonos
por encima de IoU 0.5, el F1 end-to-end que exige tanto un IoU por encima de 0.5
como una transcripción exacta tras la normalización NFKC y la eliminación de
espacios, distinguiendo mayúsculas y minúsculas, y 1-NED sobre los pares
emparejados. El fitness del mejor checkpoint es el F1 end-to-end.

Se aceptan dos estructuras: una raíz de directorio que contenga
`images/<split>/` y `labels/<split>.jsonl`, pasada como `data=`, o un YAML con
`path` más los nombres opcionales de directorio `images` y `labels`.

Los campos YAML de tipo clase son marcadores de posición del esquema: usa
`nc: 1` y `names: {0: text}`. Los modelos de OCR exponen `Results.ocr`.

En esta versión la validación es solo de inferencia. Resolutor canónico de
muestras: `libreyolo.data.ocr_dataset.resolve_ocr_samples`.

## classify

Un árbol de directorios estilo ImageFolder, no archivos de etiquetas:

```text
dataset_root/
  train/
    class_a/*.jpg
    class_b/*.jpg
  val/
    class_a/*.jpg
    class_b/*.jpg
```

`train/` es obligatorio para entrenar y define el mapeo de clase a índice por
nombre de carpeta ordenado. `val/` es obligatorio para validar. `test/` puede
estar presente, pero los comandos train y val por defecto no lo usan. Los splits
que no son de entrenamiento deben contener los mismos nombres de carpeta de
clase que el conjunto de clases esperado de train o del checkpoint. Las
extensiones de imagen admitidas están definidas en
`libreyolo.data.classify_dataset.IMAGE_EXTENSIONS`.

## gaze y point

No hay implementado ningún contrato de archivos de dataset de entrenamiento ni
de validación para `gaze`.

`point` es una tarea de salida del modelo más que un esquema de etiquetas de
dataset. Las familias point pueden adaptar internamente etiquetas existentes,
por ejemplo derivando los centros de los objetos a partir de filas de box, pero
no está definido un formato de etiquetas de texto exclusivo de point.
