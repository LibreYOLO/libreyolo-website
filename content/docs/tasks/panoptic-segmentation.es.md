---
title: Segmentación panóptica
seo_title: "Segmentación panóptica en LibreYOLO"
description: "Asigna a cada píxel un único segmento en LibreYOLO: las familias que sirven la tarea, el formato de dataset COCO-panoptic y las llamadas de predicción y validación."
lead: "La segmentación panóptica asigna cada píxel a exactamente un segmento sin solapamiento, unificando las instancias de objetos contables con las regiones amorfas del fondo. La clave de la tarea es panoptic."
keywords: [segmentación panóptica python, panoptic quality PQ, things y stuff segmentación, formato COCO panoptic, mapa de ids de segmento, métrica PQ]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # El sufijo -panoptic del nombre de archivo selecciona la tarea, así que
        # no hace falta el argumento task.
        model = LibreYOLO("LibreEoMTl-panoptic.pt")
        result = model(SAMPLE_IMAGE, save=True)

        pan = result.panoptic
        print(pan.data.shape)       # (H, W) ids de segmento
        print(pan.segments_info)    # [{"id": ..., "category_id": ...}, ...]
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreEoMTl-panoptic.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Un segmento cada vez
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        result = LibreYOLO("LibreEoMTl-panoptic.pt")(SAMPLE_IMAGE)
        pan = result.panoptic

        for segment in pan.segments_info:
            pixels = pan.segment_mask(segment["id"])   # booleano (H, W)
            print(result.names[segment["category_id"]], int(pixels.sum()))
    - label: Un checkpoint más pequeño
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreEoMTs-panoptic.pt")
        result = model(SAMPLE_IMAGE)

        print(len(result.panoptic.segment_ids))
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEoMTl-panoptic.pt")

        # val() devuelve un dict simple, no un objeto.
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/PQ"])
        print(metrics["metrics/SQ"], metrics["metrics/RQ"])
        print(metrics["metrics/PQ_things"], metrics["metrics/PQ_stuff"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreEoMTl-panoptic.pt data=my-dataset.yaml
---

## Definición

La segmentación panóptica es la unión de las otras dos tareas de segmentación.
Cada píxel recibe exactamente un segmento, los segmentos nunca se solapan y un
segmento es o bien un *thing*, una instancia de objeto contable, o bien *stuff*,
una región amorfa como el cielo o la carretera. Eso la hace más estricta que la
[segmentación de instancias](/docs/tasks/instance-segmentation), que deja sin
asignar los píxeles de fondo y permite que las máscaras se solapen, y más
estricta que la [segmentación semántica](/docs/tasks/semantic-segmentation), que
etiqueta cada píxel pero fusiona las instancias contiguas de una misma clase.

`panoptic` es la clave canónica de la tarea, y el sufijo `-panoptic` del nombre
de archivo de un checkpoint la selecciona, así que no hace falta `task=` al
cargar pesos publicados.

`predict()` rellena `result.panoptic`. `.data` es un mapa entero de ids de
segmento de tamaño `(H, W)` sobre el lienzo de la imagen original.
`.segments_info` es una lista de diccionarios, uno por segmento, cada uno con al
menos `{"id", "category_id"}`, donde `id` coincide con un valor del mapa y
`category_id` indexa `result.names`. `.segment_ids` lista los ids presentes en
orden y `.segment_mask(id)` devuelve la selección booleana `(H, W)` de un
segmento. El id de segmento `0` es el valor void: píxeles sin etiquetar,
excluidos de la métrica y fuera de `.segment_ids`.

Que algo sea *thing* o *stuff* es una propiedad de la categoría, no del segmento
individual. Va en los metadatos de categoría del conjunto de etiquetas, y una
carga útil de predicción puede copiarlo en cada segmento como `"isthing"` por
comodidad, pero los metadatos de categoría siguen siendo los autoritativos.

## Modelos

[EoMT](/docs/models/eomt) es la familia que sirve esta tarea a través de
`LibreYOLO()`. Funciona con el paquete base y ofrece checkpoints panópticos en
tres tamaños, s, b y l, entrenados con COCO.

[SenseNova-Vision](/docs/models/sensenova-vision) también emite mapas
panópticos. Es un modelo generativo guiado por prompts, con su propia factoría,
`LibreVLM`, y su propio extra; si no se define ningún vocabulario, recurre a las
categorías panópticas de COCO con las que fue ajustado. Sus pesos son no
comerciales. La latencia por imagen es mucho mayor que la de un segmentador
específico, porque cada predicción es una decodificación por difusión.

## Predicción

Los pesos se descargan de Hugging Face en el primer uso y quedan cacheados
localmente.

<code-tabs name="predict" />

`conf` filtra la selección de queries. Consulta [predicción](/docs/predict) para
fuentes, streaming y manejo de resultados.

## Formato del dataset

LibreYOLO adopta el formato COCO-panoptic tal cual, de Kirillov et al., CVPR
2019. No hay un diseño panóptico específico de LibreYOLO.

```text
dataset/
  data.yaml
  images/
    val/000000000139.jpg
  annotations/
    panoptic_val.json
    panoptic_val/000000000139.png
```

Cada imagen se empareja con un PNG RGB de la misma resolución, donde el color de
cada píxel codifica el id del segmento al que pertenece:

```text
segment_id = R + 256 * G + 256 * 256 * B
```

El id de segmento `0`, el negro RGB, es void: píxeles sin etiquetar que ni
premian ni penalizan una predicción. Todos los demás píxeles pertenecen a
exactamente un segmento.

El JSON lista, por imagen, el PNG de ids de segmento y los segmentos que
contiene:

```json
{
  "images":      [{"id": 139, "file_name": "000000000139.jpg"}],
  "annotations": [{"image_id": 139, "file_name": "000000000139.png",
                   "segments_info": [
                     {"id": 3226956, "category_id": 1, "area": 2840,
                      "bbox": [413, 158, 53, 138], "iscrowd": 0}]}],
  "categories":  [{"id": 1, "name": "person", "isthing": 1}]
}
```

`annotations[].file_name` nombra el PNG dentro del directorio panóptico, y
`segments_info[].id` coincide con un valor de ese PNG. `iscrowd` marca regiones
de grupo: nunca cuentan como falsos negativos, y una predicción que cubra en su
mayor parte una de ellas no es un falso positivo. `isthing` vive en
`categories` y nunca en un segmento individual.

El YAML apunta a ambos:

```yaml
path: dataset
val: images/val
annotations:
  val: annotations/panoptic_val.json
panoptic_dir:
  val: annotations/panoptic_val
names:
  0: person
  1: bicycle
```

`annotations` y `panoptic_dir` aceptan cada uno una única ruta o un mapeo por
split. Los ids de categoría de COCO en bruto suelen ser no contiguos, mientras
que los modelos predicen un rango contiguo `0..nc-1`, así que los ids se
remapean a través de `names` por nombre de categoría. Que una categoría del JSON
falte en `names` es un error, en vez de un descarte silencioso, porque
descartarla puntuaría como un falso negativo permanente.

El loader canónico es `libreyolo.data.PanopticDataset`.

## Entrenamiento

Hoy ninguna familia entrena segmentación panóptica en LibreYOLO: el `train()` de
EoMT lanza `NotImplementedError`, así que los checkpoints panópticos se usan tal
como se publican.

## Validación

`val()` devuelve un diccionario simple de claves `metrics/`, calculadas a la
resolución del ground truth sobre el split indicado por `val` en el YAML del
dataset. Un segmento predicho y uno real de la misma categoría casan cuando su
IoU supera 0.5, y esa correspondencia es única.

<code-tabs name="val" />

`metrics/PQ` es la Panoptic Quality, la cifra principal. Dentro de una categoría
es el producto de dos factores. La calidad de segmentación es el IoU medio sobre
los segmentos emparejados y dice cómo de bien encajan las formas emparejadas. La
calidad de reconocimiento es `TP / (TP + 0.5 FP + 0.5 FN)`, el F1 del propio
emparejamiento, y dice cuántos segmentos se encontraron siquiera. Las tres
cifras se promedian después sobre las categorías que aparecieron, y se reportan
como `metrics/PQ`, `metrics/SQ` y `metrics/RQ`, de modo que la PQ reportada es la
media de los productos por categoría y no el producto de las dos medias
reportadas.

`metrics/PQ_things` y `metrics/PQ_stuff` promedian esa misma PQ por categoría
sobre las categorías thing y las categorías stuff por separado, y
`metrics/categories` cuenta las categorías que aparecieron y sobre las que, por
tanto, se promedió. El diccionario también lleva `fitness`, una copia del valor
de PQ.

## Exportación

Los checkpoints panópticos no se exportan. `export()` lanza
`NotImplementedError` para esta tarea, porque la salida de máscaras por query
todavía no tiene un contrato de exportación en runtime. La tarea semántica de
EoMT sí exporta; consulta
[segmentación semántica](/docs/tasks/semantic-segmentation) y
[exportación y despliegue](/docs/export).
