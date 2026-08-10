---
title: DINOv2
families: [dinov2]
seo_title: "DINOv2 en LibreYOLO: segmentación semántica, clasificación y embeddings"
description: "Usa DINOv2 en LibreYOLO para segmentación semántica, clasificación y embedding de imagen completa sobre el backbone DINOv2-with-Registers. Apache-2.0 de principio a fin."
lead: "DINOv2 es un vision transformer autosupervisado entrenado por Meta AI para producir características de imagen de propósito general sin etiquetas. LibreYOLO envuelve su backbone DINOv2-with-Registers para tres tareas: segmentación semántica, clasificación y embedding de imagen completa."
keywords: [DINOv2, "DINOv2 with registers", "aprendizaje autosupervisado", vision transformer, "segmentación semántica python", "embeddings de imágenes", "extracción de características", "modelo de visión preentrenado", Meta AI]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Semántica
      language: python
      code: |
        from libreyolo import SAMPLE_IMAGE
        from libreyolo.models.dinov2.model import LibreDINOv2

        # No existe ningún checkpoint alojado por LibreYOLO para esta familia:
        # esto descarga el backbone DINOv2-with-Registers-small con licencia
        # Apache-2.0 desde la organización de Meta en Hugging Face. La cabeza
        # densa arranca con inicialización aleatoria hasta que la entrenes
        # (ver Entrenamiento más abajo).
        model = LibreDINOv2(size="s", task="semantic", nb_classes=19)
        result = model(SAMPLE_IMAGE)

        mask = result.semantic_mask
        print(mask.data.shape, mask.classes)
    - label: Clasificación
      language: python
      code: |
        from libreyolo import SAMPLE_IMAGE
        from libreyolo.models.dinov2.model import LibreDINOv2

        # nb_classes= es el número de clases de tu dataset; la cabeza lineal
        # arranca con inicialización aleatoria hasta que la entrenes.
        model = LibreDINOv2(size="s", task="classify", nb_classes=10)
        result = model(SAMPLE_IMAGE)

        print(result.probs.top1, result.probs.top1conf)
    - label: Embedding
      language: python
      code: |
        from libreyolo import SAMPLE_IMAGE
        from libreyolo.models.dinov2.model import LibreDINOv2

        # Se salta todas las cabezas de tarea: el backbone por sí solo basta,
        # así que esto no necesita fine-tuning para ser útil.
        model = LibreDINOv2(size="s", task="embed")
        result = model(SAMPLE_IMAGE)

        print(result.embeddings.data.shape)   # (1, D), normalizado con L2
    - label: Embedding de un batch
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="embed")

        # Envoltorio de conveniencia: ejecuta predict() y apila todas las filas
        # en un único tensor (N, D).
        features = model.embed(["a.jpg", "b.jpg", "c.jpg"])
        print(features.shape)
  train:
    - label: Semántica
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="semantic", nb_classes=19)
        model.train(data="my-dataset.yaml", epochs=100, batch_size=4, lr=1e-4)
    - label: Clasificación
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="classify", nb_classes=10)
        model.train(data="my-dataset.yaml", epochs=100, batch_size=4, lr=1e-4)
    - label: Multi-GPU
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="semantic", nb_classes=19)
        model.train(
            data="my-dataset.yaml", epochs=100, batch_size=4, lr=1e-4,
            device="0,1",
        )
  val:
    - label: Semántica
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="semantic", nb_classes=19)
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: Clasificación
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="classify", nb_classes=10)
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
  export:
    - label: Semántica
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="semantic", nb_classes=19)
        model.export(format="onnx")
    - label: Clasificación
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="classify", nb_classes=10)
        model.export(format="onnx")
    - label: Embedding
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="embed")
        model.export(format="tflite")
    - label: Usar el archivo exportado
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La factoría enruta según la extensión del archivo, así que un
        # artefacto exportado se carga como cualquier checkpoint y devuelve el
        # mismo objeto Results. La exportación nombra el archivo a partir de la
        # tarea, aquí LibreDINOv2s-sem.onnx.
        model = LibreYOLO("LibreDINOv2s-sem.onnx")
        result = model(SAMPLE_IMAGE)
---

## Instalación

LibreDINOv2 solo se registra cuando `transformers` está instalado, la misma
dependencia opcional que RF-DETR necesita para su backbone DINOv2, así que
necesita el mismo extra.

```bash
pip install "libreyolo[rfdetr]"
```

## Predicción

LibreYOLO no publica ningún checkpoint de LibreDINOv2. Construye el wrapper
directamente en lugar de cargar un archivo: `model_path=None` (el valor por
defecto) descarga en el primer uso el backbone
`facebook/dinov2-with-registers-small` de Meta, con licencia Apache-2.0, desde
Hugging Face. `task=` selecciona qué se ejecuta encima de él.

<code-tabs name="predict" />

`task="semantic"` y `task="classify"` añaden una cabeza densa o lineal encima
del backbone; esa cabeza se inicializa de forma aleatoria y solo resulta útil
después de que la entrenes (ver [Entrenamiento](#train)). `task="embed"` se
salta todas las cabezas y devuelve el token CLS final normalizado del backbone
como una única fila de imagen completa en `result.embeddings`, así que no
necesita entrenamiento alguno. `result.boxes` siempre es `None`: ninguna de las
tres tareas produce detecciones por instancia. Consulta
[predicción](/docs/predict) para fuentes, streaming y manejo de resultados.

## Variantes

`size` selecciona el ancho del proyector al estilo RF-DETR que se superpone al
backbone, no el backbone en sí: todos los tamaños comparten el mismo encoder
DINOv2-S (small). La segmentación semántica se ejecuta con la rejilla de
parches cuadrada nativa de DINOv2; la clasificación y el embedding se ejecutan
con la resolución de clasificación, más pequeña, que se usó para entrenar el
linear probe.

## Entrenamiento

`task="semantic"` y `task="classify"` se entrenan ambos; `task="embed"` no tiene
ninguna cabeza dependiente de las clases que ajustar y lanza
`NotImplementedError` si llamas a `train()` sobre él.

<code-tabs name="train" />

Los argumentos de palabra clave principales aquí son `batch_size` y `lr`, no
`batch` y `lr0`, que son los que usa la mayoría de las demás familias; `batch` y
`lr0` se siguen aceptando y se mapean sobre ellos, pero pasar ambos lanza un
error de conflicto. `output_dir=` (por defecto `"runs/train"`) sustituye a
`project=`/`name=` como forma principal de ubicar una ejecución, aunque pasar
`project=`/`name=` directamente sigue funcionando. Consulta
[entrenamiento](/docs/train) para datasets, aumento de datos, multi-GPU y
loggers.

## Validación

`val()` devuelve un diccionario de claves `metrics/`: mIoU y precisión por píxel
para `task="semantic"`, y precisión top-1 y top-5 para `task="classify"`.
`task="embed"` no tiene ground truth contra el que puntuar y lanza
`NotImplementedError` si llamas a `val()` sobre él.

<code-tabs name="val" />

## Exportación

<export-matrix />

Cada tarea soporta un subconjunto distinto de formatos, mostrado arriba. Un
artefacto exportado se vuelve a cargar con `LibreYOLO()` según su extensión de
archivo, así que un archivo `.onnx` o `.engine` se comporta como un checkpoint y
devuelve el mismo `Results`. [Exportación](/docs/export) enumera los argumentos
que acepta cada formato.

<code-tabs name="export" />

## Licencia

<provenance-box>

La fila "Weights" de arriba nombra la licencia que aplica, Apache-2.0, pero en
realidad no se republica nada bajo la organización de LibreYOLO en Hugging Face
para esta familia: LibreYOLO no aloja ningún checkpoint propio de LibreDINOv2.
Lo que descarga `LibreDINOv2(model_path=None)` es el repositorio
`facebook/dinov2-with-registers-small` de la propia Meta, sin tocar.

</provenance-box>

## Cita

<citation-block />
