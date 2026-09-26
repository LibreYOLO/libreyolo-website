---
title: OCR
seo_title: 'OCR: detección y reconocimiento de texto en LibreYOLO'
description: >-
  Encuentra y lee texto en imágenes con LibreYOLO. Predice cuadriláteros y
  transcripciones, etiqueta un dataset JSONL y valida con hmean, F1 end-to-end y
  1-NED.
lead: >-
  El OCR localiza el texto de una imagen y lo lee. LibreYOLO lo expone como la
  tarea ocr, que devuelve un polígono de cuatro puntos más una transcripción por
  cada región de texto, en orden de lectura.
keywords:
  - ocr python
  - extraer texto de una imagen python
  - reconocimiento de texto en escenas
  - detección de texto en imágenes
  - PP-OCRv5 python
last_verified: 1.5.0
snippets:
  predict:
    - label: Leer el texto de una imagen
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # El nivel t es el más ligero de los dos, pensado para CPU. SAMPLE_IMAGE
        # mantiene esto ejecutable; apúntalo a una imagen con texto tuya.
        model = LibreYOLO("LibrePPOCRt-ocr.pt")
        result = model(SAMPLE_IMAGE)

        regions = result.ocr
        print(len(regions), "regions")
        for text, score in zip(regions.texts, regions.conf):
            print(repr(text), float(score))
    - label: Leer los cuadriláteros
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibrePPOCRt-ocr.pt")

        result = model(SAMPLE_IMAGE)


        regions = result.ocr

        print(regions.data.shape)   # polígonos (N, 4, 2), TL TR BR BL

        print(regions.xyxy)         # envolventes de esos polígonos alineadas a
        los ejes

        print(regions.det_conf)     # puntuación de detección, distinta de .conf
    - label: Filtrar por confianza de reconocimiento
      language: python
      code: >
        import numpy as np

        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibrePPOCRt-ocr.pt")

        result = model(SAMPLE_IMAGE)


        # Indexa con posiciones, no con una máscara booleana: el slicing
        arrastra

        # las transcripciones y ambos arrays de puntuaciones junto con la
        geometría.

        regions = result.ocr.numpy()

        keep = regions[np.flatnonzero(regions.conf >= 0.9)]

        print(keep.texts)
  val:
    - label: Validar y leer las claves de las métricas
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePPOCRt-ocr.pt")
        metrics = model.val(data="my-ocr-dataset")

        print(metrics["metrics/det_precision"], metrics["metrics/det_recall"])
        print(metrics["metrics/det_hmean"])
        print(metrics["metrics/e2e_f1"])       # fitness
        print(metrics["metrics/rec_1-NED"])
source_hash: 58ad5305c9dd458c
---

## Definición

La tarea `ocr` hace dos cosas en una sola llamada: localiza cada región de texto
de una imagen y la transcribe. Las regiones vuelven como polígonos de cuatro
puntos en lugar de cajas alineadas a los ejes, porque el texto en escenas suele
estar rotado, y en orden de lectura, de arriba abajo y luego de izquierda a
derecha.

Una predicción rellena `result.ocr`, un payload `OCRRegions`. `.data` es un
array de floats `(N, 4, 2)` con los polígonos en píxeles de la imagen original,
ordenados como superior izquierda, superior derecha, inferior derecha e inferior
izquierda; `.texts` es la lista de las N transcripciones; `.conf` es la
puntuación de reconocimiento de cada región y `.det_conf` la de detección;
`.xyxy` da la envolvente alineada a los ejes de cada polígono. Como los
cuadriláteros son polígonos de verdad, no rellenan `result.boxes`. Hacer slicing
de un `OCRRegions` arrastra las transcripciones y ambos arrays de puntuaciones
junto con la geometría.

## Modelos

Dos familias sirven `ocr`.

[PP-OCRv5](/docs/models/pp-ocrv5) es el pipeline dedicado: un detector de
binarización diferenciable encuentra los cuadriláteros de texto y un reconocedor
SVTR/CTC los lee, con ambas etapas agrupadas en un único archivo `.pt` junto con
el charset de reconocimiento. Se publica en dos niveles, uno más ligero para CPU
y uno de servidor para mayor precisión, y un único diccionario cubre el chino
simplificado y tradicional, el inglés, el japonés y el pinyin.

[SenseNova-Vision](/docs/models/sensenova-vision) llega al OCR generando las
palabras como texto etiquetado desde el mismo checkpoint de 7B que sirve sus
otras seis tareas, cargado con `LibreVLM("sensenova-vision", task="ocr")`.
Necesita el extra `sensenova`, y sus pesos están restringidos a uso no
comercial; la licencia está en su página.

## Predicción

Los pesos se descargan de Hugging Face en el primer uso y quedan cacheados
localmente.

<code-tabs name="predict" />

PP-OCRv5 ejecuta la detección con un límite fijo del lado largo y luego reconoce
las regiones recortadas en batches; `rec_batch` controla cuántos recortes pasan
por el reconocedor en cada forward pass. Las fuentes con varias imágenes se
procesan de forma secuencial, porque un pipeline de dos etapas no agrupa en
batch entre imágenes. Consulta [predicción](/docs/predict) para fuentes,
streaming y manejo de resultados.

## Formato del dataset

Las etiquetas de OCR son un archivo JSONL por split, un objeto JSON por imagen,
junto a las propias imágenes.

```text
my-ocr-dataset/
  images/
    val/receipt.jpg
  labels/
    val.jsonl
```

Cada línea nombra una imagen y enumera sus regiones:

```json
{"image": "receipt.jpg", "regions": [{"polygon": [[10, 12], [118, 14], [117, 40], [9, 38]], "text": "TOTAL 12.50"}]}
```

`polygon` es un cuadrilátero de cuatro puntos en coordenadas de píxel
absolutas, ordenado como superior izquierda, superior derecha, inferior derecha
e inferior izquierda. Una región cuyo texto no puede leerse se etiqueta con
`"text": "###"`, la convención don't-care de ICDAR: queda excluida de la
puntuación de reconocimiento, y una predicción que se solape con ella se ignora
en lugar de contarse como falso positivo.

Pasar el directorio raíz como `data=` es suficiente. La alternativa es un YAML
de dataset, con `path` más los nombres opcionales de los directorios `images` y
`labels`, y `nc: 1` con `names: {0: text}` como marcadores de posición del
esquema, ya que un modelo de OCR devuelve `Results.ocr` en lugar de detecciones.
Consulta [formatos de dataset](/docs/reference/dataset-formats) para el contrato
completo.

## Entrenamiento

Ninguna de las dos familias de OCR tiene implementación de entrenamiento:
`train()` lanza `NotImplementedError` en ambas, y el soporte de OCR cubre solo
la predicción y la validación. La página de PP-OCRv5 nombra el código de
entrenamiento upstream con licencia Apache-2.0 y el script de conversión que
devuelve a LibreYOLO un checkpoint al que se le ha hecho fine-tuning.

## Validación

`val()` puntúa el pipeline completo, detección y reconocimiento juntos,
emparejando uno a uno los polígonos predichos con los polígonos del ground truth
con un IoU por encima de 0.5.

<code-tabs name="val" />

`metrics/det_precision`, `metrics/det_recall` y `metrics/det_hmean` puntúan solo
la localización: para que haya coincidencia basta con el solape de los
polígonos, diga lo que diga la transcripción. `metrics/e2e_precision`,
`metrics/e2e_recall` y `metrics/e2e_f1` añaden la lectura: la coincidencia
necesita el mismo solape de polígonos y una transcripción idéntica tras la
normalización NFKC y la eliminación de los espacios en blanco, y la comparación
sigue distinguiendo mayúsculas de minúsculas. `metrics/e2e_f1` es además
`fitness`, el número que lee la selección del mejor checkpoint.

`metrics/rec_1-NED` califica al reconocedor por su cuenta, sobre los pares que
la detección ya emparejó: uno menos la distancia de edición normalizada, así que
una transcripción que falla por un carácter puntúa cerca de 1 donde el F1
end-to-end la puntúa 0.

## Exportación

No hay ningún formato de exportación disponible para esta tarea. PP-OCRv5 son
dos redes avanzando juntas en lugar de un único grafo trazable, y `export()`
lanza una excepción para todos los formatos en ambas familias. Para desplegar
fuera de LibreYOLO, haz fine-tuning upstream y usa la ruta de despliegue
upstream.
