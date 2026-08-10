---
title: Datasets
seo_title: "Datasets de entrenamiento en LibreYOLO"
description: "El YAML de dataset que lee LibreYOLO, la estructura de carpetas que espera, cómo funciona la descarga automática y el comando doctor que comprueba un dataset antes de entrenar."
lead: "Un dataset de LibreYOLO es un archivo YAML que nombra una raíz, sus splits y sus nombres de clases. Todo lo demás, incluido dónde viven los archivos de etiquetas, se deriva de ese archivo por convención."
keywords:
  - formato dataset yolo
  - data.yaml
  - entrenar yolo dataset propio
  - formato etiquetas yolo
  - dataset coco json
  - descarga automatica dataset
  - libreyolo doctor
  - desbalanceo de clases dataset
  - fuga de datos train val
last_verified: "1.5.0"
snippets:
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # Un nombre incluido, una ruta relativa o una ruta absoluta funcionan igual.
        model.train(data="coco8.yaml", epochs=10)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=coco8.yaml epochs=10
  doctor:
    - label: Comprobar un dataset
      language: bash
      code: |
        libreyolo doctor my-dataset.yaml
    - label: Hacer fallar un job de CI también con warnings
      language: bash
      code: |
        libreyolo doctor my-dataset.yaml strict=true json=true
    - label: Omitir la pasada de decodificación de imágenes
      language: bash
      code: |
        # Lee solo las etiquetas y el YAML. Las comprobaciones de corrupción,
        # duplicados y fuga entre splits necesitan los píxeles, así que se omiten.
        libreyolo doctor my-dataset.yaml fast=true
    - label: Python
      language: python
      code: |
        from libreyolo import doctor

        report = doctor.diagnose("my-dataset.yaml", imgsz=640)

        for finding in report.findings:
            print(finding.severity.value, finding.check_id, finding.message)

        raise SystemExit(report.exit_code(strict=False))
---

## Apunta el entrenamiento a un dataset

`data=` acepta una ruta a un YAML o el nombre de una config que se distribuye con
el paquete.

<code-tabs name="train" />

El nombre se resuelve en un orden fijo: una ruta absoluta que exista, luego el
nombre tal cual relativo al directorio de trabajo, luego el mismo nombre con
`.yaml` añadido y por último el directorio de configs incluidas. Cuando nada
coincide, el error nombra cada directorio en el que se buscó y lista las configs
incluidas.

## Configs incluidas

Trece configs de datasets se distribuyen dentro del paquete, bajo
`libreyolo/config/datasets/`.

| Config | Tarea | Notas |
|---|---|---|
| `coco8.yaml` | detect | 8 imágenes, se descarga desde una URL simple |
| `coco128.yaml` | detect | 128 imágenes |
| `coco1000.yaml` | detect | 800 de train, 200 de val |
| `coco5000.yaml` | detect | 4000 de train, 1000 de val |
| `coco.yaml` | detect | COCO 2017 completo |
| `coco-val-only.yaml` | detect | solo val2017 |
| `coco8-pose.yaml` | pose | 8 imágenes, keypoints COCO-17 |
| `coco-pose.yaml` | pose | keypoints de COCO 2017 |
| `ade20k.yaml` | semantic | 150 clases |
| `cityscapes.yaml` | semantic | 19 clases, descarga manual |
| `cocostuff.yaml` | semantic | 182 clases, descarga manual |
| `gopro.yaml` | restore | pares de deblurring |
| `sr8.yaml` | restore | pares de superresolución |

Solo `coco8.yaml` y `coco128.yaml` llevan una URL de descarga simple. El resto o
bien lleva un bloque de descarga en Python, que necesita la activación explícita
descrita más abajo, o bien espera que los datos ya estén en disco.

## Dónde vive un dataset en disco

La clave `path` del YAML nombra la raíz del dataset. Un `path` absoluto se usa
tal cual está escrito. Uno relativo se busca primero bajo el directorio de
datasets, después junto al propio archivo YAML, y un dataset que está a punto de
descargarse va al directorio de datasets.

Ese directorio es `~/datasets`, sobrescrito por la variable de entorno
`LIBREYOLO_DATASETS_DIR`. No hay archivo de configuración para ello.

## Las claves del YAML

```yaml
path: my-dataset        # raíz del dataset
train: images/train     # obligatorio para entrenar
val: images/val         # obligatorio para validar
test: images/test       # opcional
nc: 3                   # opcional; debe coincidir con names
names:
  0: person
  1: helmet
  2: vest
download: https://example.com/my-dataset.zip   # opcional
```

`train`, `val` y `test` aceptan cada uno un directorio de imágenes, un archivo
`.txt` con una ruta de imagen por línea, o una lista que mezcle ambos. Las líneas
de una lista `.txt` pueden ser relativas, en cuyo caso se resuelven contra el
directorio del propio archivo de lista, y las líneas que empiezan por `#` se
omiten.

`names` puede ser una lista o un mapping con claves enteras. `nc` es opcional;
cuando ambos están presentes y no coinciden, el doctor lo reporta como error.

## Estructura de directorios y archivos de etiquetas

La detección, la segmentación, la pose y los boxes orientados comparten una misma
estructura. La ruta de la etiqueta se deriva de la ruta de la imagen
reescribiendo un componente de directorio `images` a `labels` y cambiando la
extensión a `.txt`:

```text
my-dataset/
  images/train/0001.jpg   ->   labels/train/0001.txt
  images/val/0002.jpg     ->   labels/val/0002.txt
```

Solo se reescribe un componente de ruta `images` completo, así que un directorio
llamado `images_old` se deja intacto.

Una fila de detección son cinco campos, todos normalizados a `[0, 1]` respecto al
ancho y alto originales de la imagen:

```text
<class_id> <cx> <cy> <w> <h>
```

Un archivo de etiquetas ausente o vacío significa que la imagen no tiene objetos,
y se entrena como fondo en lugar de lanzar un error. Una fila con más de cinco
campos se lee como un polígono y su box pasa a ser la extensión del polígono, de
modo que una exportación de segmentación usada para entrenamiento de detección
carga sin quejarse. El doctor reporta cuántas filas tomaron ese camino.

## Otras tareas

La segmentación mantiene la misma estructura con filas de polígonos,
`<class_id> <x1> <y1> ... <xN> <yN>`, con al menos tres puntos. Una fila de
detección de cinco campos se acepta y significa una instancia rectangular.

La pose añade al YAML `kpt_shape: [K, D]` y una permutación opcional `flip_idx`.
Cada fila tiene exactamente `5 + K * D` campos: el box y luego `K` keypoints de
`x y` o `x y v`, con visibilidad `0`, `1` o `2`.

Los boxes orientados usan exactamente nueve campos, la clase seguida de cuatro
puntos de esquina en coordenadas normalizadas. No se guarda ningún ángulo en el
archivo.

La segmentación semántica empareja cada imagen con una máscara de un solo canal
de la misma resolución, resuelta sustituyendo `masks_dir` (por defecto `masks`)
en lugar de `images`. El valor de píxel `255` significa ignorar. `label_mapping`
remapea los ids de origen a ids de entrenamiento en el momento de la carga.

La clasificación usa un árbol tipo ImageFolder en lugar de archivos de etiquetas,
con `train/` y `val/` conteniendo cada uno un directorio por clase. El mapeo de
clase a índice es el orden de los nombres de carpeta ordenados alfabéticamente.

La restauración empareja una entrada degradada con un objetivo limpio de
resolución idéntica mediante `input_dir` y `target_dir`. La profundidad, las
normales de superficie y los bordes emparejan cada uno una imagen con un mapa
denso mediante su propia clave de directorio.

El contrato completo por tarea, incluidas las convenciones de escala de
profundidad y la codificación PNG de ids de segmento panópticos, está en
`docs/dataset_schema.md` en el repositorio de la librería.

## COCO JSON nativo

Un archivo de anotaciones COCO JSON puede usarse directamente. Añade un mapping
`annotations` y la ruta del split pasa a ser la raíz de imágenes:

```yaml
path: my-dataset
train: images/train
val: images/val
annotations:
  train: annotations/train.json
  val: annotations/val.json
```

Cuando `names` está presente, los nombres de categoría del JSON deben coincidir
con él, y `names` define los ids de etiqueta que el modelo predice. Sin `names`,
los ids de categoría de COCO se ordenan y se mapean densamente a `0..N-1`.

Este camino espera un directorio de imágenes por split. Una lista de rutas o una
lista de imágenes `.txt` lanza un error en lugar de cargar en silencio un
conjunto distinto.

## Descarga automática

Un dataset cuenta como presente cuando su ruta `train` o `val` resuelve a un
directorio no vacío o a un archivo existente. Cuando no es así, y el YAML tiene
una clave `download`, el valor decide qué pasa a continuación.

Una URL `http` o `https` se descarga y, si es un zip, se extrae en la raíz del
dataset. Cualquier otra cosa se trata como un script de Python embebido y solo se
ejecuta con `allow_download_scripts=True`. Sin eso, el script se omite con un
warning y el entrenamiento continúa contra lo que haya en disco.

```bash
libreyolo train model=LibreYOLO9s.pt data=coco.yaml allow_download_scripts=true
```

El flag es una puerta de ejecución de código, no una puerta de red. Las descargas
por URL ocurren igualmente; son los bloques `download: |` los que lo necesitan.
La CLI imprime un warning cuando el flag está activado, y el doctor nunca lo
habilita.

## Comprueba el dataset antes de entrenar

`libreyolo doctor` lee un dataset de detección y reporta lo que saldría mal antes
de que haya una GPU involucrada. Sale con código 1 cuando encuentra errores, así
que funciona como puerta de CI.

<code-tabs name="doctor" />

Las comprobaciones se agrupan en seis familias:

| Familia | Busca |
|---|---|
| `config` | `names` ausente, `nc` que no coincide con `names`, splits ausentes o vacíos, nombres de clase duplicados |
| `files` | imágenes sin archivo de etiquetas, etiquetas sin imagen, imágenes ausentes listadas en un split, colisiones de nombres base |
| `labels` | filas malformadas, ids de clase fuera de `[0, nc)`, coordenadas fuera de `[0, 1]`, boxes de área cero, boxes diminutos o enormes, boxes duplicados, archivos de etiquetas idénticos byte a byte |
| `balance` | clases con cero o pocas instancias, ratio de desbalanceo de clases, clases presentes en un solo split, proporción de imágenes de fondo |
| `images` | archivos que no se pueden decodificar, rotación EXIF, disposiciones de canales extrañas, imágenes uniformes, duplicados exactos y casi exactos |
| `splits` | la misma imagen apareciendo en dos splits, de forma exacta o casi idéntica |

`--only` y `--skip` aceptan un id de comprobación o un prefijo de familia, así
que `skip=images,labels.tiny_object` es válido. `--fast` descarta toda
comprobación que necesite decodificar píxeles, que son las familias `images` y
`splits`.

Vale la pena conocer dos comportamientos. `--strict` hace que los warnings
también hagan fallar el código de salida, además de los errores. Y el doctor
cubre solo datasets de detección: un dataset de pose, segmentación o boxes
orientados se rechaza con un mensaje que nombra lo que detectó, en lugar de
comprobarse contra el contrato equivocado.

## Relacionado

- [Hiperparámetros](/docs/train/hyperparameters) para los argumentos que
  `train()` acepta una vez que los datos están en su sitio.
- [Validación y métricas](/docs/train/validation) para evaluar sobre el split
  `val` o `test`.
