---
title: Detección de bordes
seo_title: "Detección de bordes en LibreYOLO"
description: "Predice un mapa denso de probabilidad de bordes a partir de una imagen en LibreYOLO. Convierte un checkpoint, aplica un umbral al mapa, valida con ODS y OIS, y exporta."
lead: "La detección de bordes predice la probabilidad de que cada píxel caiga sobre el contorno de un objeto. LibreYOLO la expone como la tarea edge, que devuelve un mapa denso de probabilidad sobre el lienzo de la imagen original en lugar de un conjunto de segmentos."
keywords: [detección de bordes python, detección de contornos deep learning, mapa de probabilidad de bordes, ODS OIS F-measure, DexiNed python]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Predecir un mapa de bordes
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # LibreYOLO no incluye ningún checkpoint de bordes; convierte uno antes (abajo).
        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")
        result = model(SAMPLE_IMAGE, save=True)

        edges = result.edges
        print(edges.array.shape)          # (H, W) float32 en [0, 1]
        print(edges.binary(0.5).sum())    # píxeles de borde con umbral 0.5
    - label: Elegir tu propio umbral
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")
        result = model(SAMPLE_IMAGE)

        # El mapa continuo se conserva para que el umbral siga siendo tu decisión.
        for t in (0.3, 0.5, 0.7):
            print(t, int(result.edges.binary(t).sum()))
    - label: Guardar la visualización
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")
        result = model(SAMPLE_IMAGE)

        # plot() dibuja el mapa; está definido para resultados de bordes y de normales.
        result.plot().save("edges.png")
  val:
    - label: Validar y leer las claves de las métricas
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")
        metrics = model.val(data="my-dataset.yaml", imgsz=352)

        print(metrics["metrics/ODS"])              # fitness
        print(metrics["metrics/OIS"])
        print(metrics["metrics/best_threshold"])
    - label: Cambiar el barrido y la tolerancia de emparejamiento
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")
        metrics = model.val(
            data="my-dataset.yaml",
            imgsz=352,
            edge_thresholds=(0.1, 0.2, 0.3, 0.4, 0.5),
            edge_max_dist=0.0075,
        )

        print(metrics["metrics/ODS"], metrics["metrics/best_threshold"])
  export:
    - label: Exportación
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")
        model.export(format="onnx", imgsz=352)
    - label: Usar el archivo exportado
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La factory decide según la extensión del archivo, así que un artefacto
        # exportado se carga como cualquier checkpoint y devuelve el mismo Results.
        model = LibreYOLO("weights/LibreDexiNedb-edge.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.edges.array.shape)
---

## Definición

La tarea `edge` predice una probabilidad por píxel a partir de una sola imagen
RGB: `0` significa que no es borde y `1` que sí lo es. El mapa se mantiene
continuo, así que elegir el umbral que lo convierte en una imagen binaria de
contornos queda en manos de quien llama, y el umbral adecuado depende del
dataset y del uso posterior.

Una predicción rellena `result.edges`, una carga útil `EdgeMap` que contiene un
array float32 `(H, W)` en `[0, 1]` sobre el lienzo de la imagen original.
`.array` devuelve ese mapa como NumPy y `.binary(threshold)` devuelve una
máscara booleana. `result.boxes` queda vacío, así que `conf`, `iou` y `max_det`
no tienen efecto. `Results.plot()` cubre esta tarea y dibuja el mapa
directamente.

## Modelos

Tres familias cubren `edge`.

[DexiNed](/docs/models/dexined), la Dense Extreme Inception Network, fusiona
varias salidas laterales en un único mapa de probabilidad y funciona a una
resolución nativa de 352 px.

[TEED](/docs/models/teed), el Tiny and Efficient Edge Detector, es una red
pequeña con la misma resolución nativa de 352 px, con un stride de submuestreo
de 4 frente a los 16 de DexiNed, así que acepta más valores de `imgsz`.

[LibreMODUS](/docs/models/libremodus) produce bordes estilo Canny como uno de
los objetivos de un modelo any-to-any. Necesita el extra `modus` y tu propia
cuenta autenticada de Hugging Face, y no ofrece ni `val()` ni `export()`, así
que no participa en las secciones de validación y exportación de más abajo.

## Predicción

LibreYOLO no publica ningún checkpoint de bordes. Los pesos publicados
oficialmente de DexiNed y TEED están entrenados sobre BIPED, cuyos términos de
dataset restringen el uso a fines no comerciales, así que LibreYOLO no los
replica. Convierte un checkpoint que tengas licencia para usar y después carga
el archivo convertido por su ruta:

```bash
python weights/convert_dexined_weights.py upstream.pth weights/LibreDexiNedb-edge.pt --verify
```

<code-tabs name="predict" />

El nombre del archivo tiene que llevar el sufijo de tarea `-edge` para que el
cargador lo reconozca. `imgsz` debe ser divisible por el stride de submuestreo
de la red, y LibreYOLO lanza un error claro que nombra el divisor cuando no lo
es. Consulta la [predicción](/docs/predict) para las fuentes, el streaming y el
manejo de resultados.

## Formato del dataset

La validación de bordes empareja cada imagen RGB con un mapa de un solo canal,
con el mismo nombre base y la misma resolución, más una máscara de validez
opcional.

```text
dataset/
  data.yaml
  images/
    val/scene.jpg
  edges/
    val/scene.png
  masks/
    val/scene.png
```

```yaml
path: dataset
train: images/train
val: images/val
edges_dir: edges
masks_dir: masks
nc: 1
names: {0: edge}
```

El objetivo es un PNG o TIF de un solo canal, no una visualización RGB. Los
mapas enteros se dividen por el máximo de su dtype; los mapas float ya tienen
que ser finitos y estar en `[0, 1]`. Los píxeles de la máscara cuentan como
válidos cuando son distintos de cero, y los píxeles de padding nunca
contribuyen a una métrica. `edge_invert: true` cubre las fuentes que guardan
bordes negros sobre blanco. Consulta los
[formatos de dataset](/docs/reference/dataset-formats) para el contrato
completo.

## Entrenamiento

Ninguna familia de bordes de LibreYOLO tiene implementado el entrenamiento:
`train()` lanza `NotImplementedError` en las tres. Cada página de modelo nombra
el script de conversión que convierte un checkpoint entrenado en otro sitio en
uno que LibreYOLO pueda cargar.

## Validación

`val()` informa de las F-measures al estilo BSDS. Las predicciones continuas se
adelgazan primero con una supresión de no máximos por gradiente en cuatro
direcciones, y después los píxeles de borde predichos y los del ground truth se
emparejan uno a uno dentro de una tolerancia de distancia.

<code-tabs name="val" />

`metrics/ODS` es la F-measure a escala óptima de dataset: los recuentos de
emparejamientos se agregan en todo el dataset para cada umbral, y se informa de
la mejor de esas F-measures agregadas. Es también `fitness`, el número que lee
la selección del mejor checkpoint. `metrics/OIS` es la F-measure a escala óptima
de imagen, la media sobre las imágenes de la mejor F-measure de cada una, así
que deja que cada imagen elija su propio umbral. `metrics/best_threshold` es el
único umbral que produjo el ODS, que es el que hay que reutilizar en
`edges.binary()` durante la inferencia.

Dos argumentos dan forma al barrido. `edge_thresholds` es el conjunto de
umbrales que se prueban, y por defecto va de 0.01 a 0.99 en centésimas.
`edge_max_dist` es la tolerancia de emparejamiento como fracción de la diagonal
de la imagen, con `0.0075` por defecto; un par más separado que eso no cuenta
como emparejamiento.

## Exportación

Un modelo de bordes exportado se vuelve a cargar con `LibreYOLO()` según la
extensión de su archivo, así que un archivo `.onnx` se comporta como un
checkpoint y devuelve el mismo `Results`.

<code-tabs name="export" />

La exportación de bordes usa un contrato de ejecución de resolución fija y
batch 1: se rechazan `dynamic` y cualquier `batch` distinto de 1, y el grafo
exportado emite un único mapa de probabilidad fusionado. La cobertura por
formato está en las páginas de [DexiNed](/docs/models/dexined) y
[TEED](/docs/models/teed) y en la
[matriz completa de exportación](/docs/reference/export-matrix). La
[exportación](/docs/export) enumera los argumentos que acepta cada formato.
