---
title: Estimación de profundidad
seo_title: "Estimación monocular de profundidad en LibreYOLO"
description: "Predice un mapa denso de profundidad relativa a partir de una sola imagen en LibreYOLO. Compara las familias de profundidad, interpreta sus métricas y exporta un modelo de profundidad."
lead: "La estimación de profundidad predice a qué distancia está cada píxel de la cámara usando una única imagen. LibreYOLO la expone como la tarea depth, que devuelve un mapa denso de profundidad inversa relativa sobre el lienzo de la imagen original."
keywords: [estimación de profundidad monocular python, mapa de profundidad de una imagen, modelo de profundidad relativa, depth anything libreyolo, calcular profundidad con python]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Predecir un mapa de profundidad
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")
        result = model(SAMPLE_IMAGE, save=True)

        depth = result.depth_map
        print(depth.data.shape)              # (H, W) sobre el lienzo original
        print(depth.min, depth.max, depth.mean)
    - label: Trabajar con los valores
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")
        result = model(SAMPLE_IMAGE)

        depth = result.depth_map
        raw = depth.data          # más alto es más cerca; sin unidad métrica, sin escala
        gray = depth.normalized() # reescalado a [0, 1] para visualizar
        print(raw.shape, float(gray.max()))
    - label: Una alternativa compacta
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # El mismo contrato de tarea, con una red mucho más pequeña para runtimes edge.
        model = LibreYOLO("LibreZipDepthb-depth.pt")
        result = model(SAMPLE_IMAGE)

        print(result.depth_map.data.shape)
  val:
    - label: Validar y leer las claves de las métricas
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/abs_rel"])
        print(metrics["metrics/rmse"])
        print(metrics["metrics/delta1"])   # fitness
        print(metrics["metrics/delta2"], metrics["metrics/delta3"])
  export:
    - label: Exportar
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")
        model.export(format="onnx")
    - label: Ejecutar el archivo exportado
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La factory decide según la extensión del archivo, así que un artefacto
        # exportado se carga como cualquier checkpoint y devuelve el mismo objeto Results.
        model = LibreYOLO("LibreDepthAnythingV2s-depth.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.depth_map.data.shape)
---

## Definición

La tarea `depth` predice un valor por píxel a partir de una sola imagen RGB.
LibreYOLO define ese valor como profundidad inversa relativa: más alto significa
más cerca de la cámara, y los números no llevan unidad métrica ni una escala que
se mantenga entre dos imágenes. Comparar la profundidad entre dos píxeles de la
misma predicción tiene sentido; comparar un valor con el valor de otra imagen, no.

Una predicción rellena `result.depth_map`, un payload `DepthMap` que contiene un
array `(H, W)` sobre el lienzo de la imagen original. `.min`, `.max` y `.mean`
leen los valores finitos, y `.normalized()` reescala el mapa a `[0, 1]` para
mostrarlo. `result.boxes` queda vacío, así que `conf`, `iou` y `max_det` no
tienen efecto, y `save=True` escribe una imagen del mapa con mapa de color en
lugar de una foto anotada.

## Modelos

Seis familias cubren `depth`.

[Depth Anything V2](/docs/models/depth-anything-v2) combina un encoder DINOv2 con
un decoder DPT y es aquí la opción general por defecto. La licencia decide el
tamaño tanto como la precisión: el checkpoint Small es Apache-2.0 mientras que
Base y Large son de uso no comercial, así que revisa la tabla de checkpoints de
su página antes de elegir uno.

[Depth Anything 3](/docs/models/depth-anything-3) porta el checkpoint
DA3MONO-LARGE, un transformer sin más, sin especialización arquitectónica para
profundidad.

[ZipDepth](/docs/models/zipdepth) es el nivel compacto: una CNN
reparametrizable destilada de Depth Anything V2 Large, con un segundo checkpoint
cuyo decoder evita las operaciones gather y unfold para los compiladores de NPU
que no las tienen.

[MiDaS](/docs/models/midas) es la línea de trabajo que estableció el protocolo de
profundidad relativa zero-shot con el que se miden las demás familias. Es la
única familia de profundidad que LibreYOLO no republica: pedir un checkpoint
descarga el asset oficial desde la release de GitHub de sus autores y comprueba
un SHA-256 fijado.

[LibreMODUS](/docs/models/libremodus) llega a la profundidad como uno de los
objetivos de un modelo any-to-any, y no como una cabeza dedicada. Necesita el
extra `modus` y tu propia cuenta autenticada de Hugging Face, y no ofrece ni
`val()` ni `export()`.

[SenseNova-Vision](/docs/models/sensenova-vision) genera el mapa de profundidad
como una imagen mediante un decode por difusión, desde el mismo checkpoint de 7B
que cubre sus otras seis tareas. Necesita el extra `sensenova`, y sus pesos están
restringidos a uso no comercial; la licencia está en su página.

## Predicción

Los pesos se descargan de Hugging Face en el primer uso y se cachean en local,
salvo en las dos familias señaladas arriba.

<code-tabs name="predict" />

La resolución de entrada está limitada por familia. Depth Anything V2 y Depth
Anything 3 se construyen sobre una rejilla de parches DINOv2, así que `imgsz`
debe ser divisible por 14, algo que LibreYOLO comprueba antes de ejecutar.
`Results.plot()` no cubre esta tarea; está definido solo para normales de
superficie y bordes. Consulta [predicción](/docs/predict) para fuentes, streaming
y manejo de resultados.

## Formato del dataset

La validación de profundidad empareja cada imagen con un mapa de profundidad
denso de un solo canal y la misma resolución, que se localiza sustituyendo el
directorio de profundidad en la ruta de la imagen.

```text
dataset/
  data.yaml
  images/
    val/room.jpg
  depths/
    val/room.png
```

```yaml
path: dataset
val: images/val
depths_dir: depths
nc: 1
names: {0: depth}
```

Los mapas son PNG o TIF de un solo canal, o `.npy`. Los valores son profundidad
en crudo en una unidad que el dataset mantiene consistente, y los píxeles a `0`,
negativos, NaN e infinitos marcan muestras inválidas que se excluyen de las
métricas. Los mapas enteros se dividen por `depth_scale`, que por defecto vale
`256.0`, la convención de los PNG de 16 bits; los mapas `.npy` en coma flotante
se usan tal cual. `depth_stem_suffix` y `depth_mask_suffix` cubren los datasets
que nombran de otra forma sus archivos de profundidad o sus máscaras de validez.
Consulta [formatos de dataset](/docs/reference/dataset-formats) para el contrato
completo.

## Entrenamiento

Ninguna familia de profundidad en LibreYOLO tiene implementación de
entrenamiento: `train()` lanza `NotImplementedError` en las seis. La página de
cada modelo indica el script de conversión que convierte un checkpoint entrenado
upstream en uno que LibreYOLO pueda cargar.

## Validación

`val()` ejecuta el validador de profundidad común. La profundidad relativa no
tiene escala absoluta, así que cada predicción se ajusta primero a la inversa de
su ground truth con una escala y un desplazamiento por mínimos cuadrados
calculados por imagen, y después se vuelve a invertir a profundidad. Todas las
métricas de abajo se calculan por imagen sobre ese mapa alineado y se promedian
sobre el dataset, contando solo los píxeles que el dataset marca como válidos.

<code-tabs name="val" />

`metrics/abs_rel` es el error relativo absoluto medio, el residuo dividido por la
profundidad del ground truth, y cuanto más bajo mejor. `metrics/rmse` es la raíz
del error cuadrático medio en la propia unidad de profundidad del dataset,
también cuanto más bajo mejor. `metrics/delta1`, `metrics/delta2` y
`metrics/delta3` son las precisiones por umbral: la fracción de píxeles válidos
cuya razón respecto al ground truth, tomada en la dirección que sea mayor, cae
por debajo de 1,25, de 1,25 al cuadrado y de 1,25 al cubo, así que cuanto más
alto mejor. `metrics/delta1` es además `fitness`, el número que lee la selección
del mejor checkpoint.

## Exportación

Un modelo de profundidad exportado se vuelve a cargar con `LibreYOLO()` según la
extensión de su archivo, así que un `.onnx` o un `.engine` se comporta como un
checkpoint y devuelve el mismo `Results`, con `depth_map` en lugar de boxes.

<code-tabs name="export" />

La cobertura cambia según la familia, y Depth Anything 3 rechaza cualquier
formato fuera de su conjunto validado en vez de intentar una conversión sin
validar. Revisa la página del modelo y la
[matriz completa de exportación](/docs/reference/export-matrix) antes de
comprometerte con un destino. LibreMODUS y SenseNova-Vision no exportan en
absoluto. [Exportación](/docs/export) lista los argumentos que acepta cada
formato.
