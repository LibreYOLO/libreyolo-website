---
title: CLIP
families:
  - clip
seo_title: 'CLIP en LibreYOLO: clasificación y embeddings zero-shot'
description: >-
  Usa CLIP en LibreYOLO para clasificación de imágenes zero-shot y embeddings de
  imagen y texto. Sin entrenamiento: set_classes() define el conjunto de
  etiquetas en tiempo de ejecución.
lead: >-
  CLIP es un modelo de doble torre que puntúa una imagen frente a prompts de
  texto en lugar de un conjunto fijo de etiquetas. LibreYOLO lo soporta para
  clasificación zero-shot y embeddings de imagen y texto, sin ningún paso de
  entrenamiento.
keywords:
  - CLIP
  - OpenCLIP
  - clasificación zero-shot
  - clasificar imágenes sin entrenar
  - embeddings de imágenes python
  - embeddings de texto
  - búsqueda por similitud imagen texto
  - vocabulario abierto
  - LAION-2B
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreCLIPb32-cls.pt")
        model.set_classes(["a forklift", "an empty aisle", "a spill"])
        result = model(SAMPLE_IMAGE, save=True)

        print(model.names[result.probs.top1], float(result.probs.top1conf))
    - label: CLI
      language: bash
      code: >
        # Sin una llamada a set_classes(), predict desde la CLI usa los 1.000

        # nombres de clase de ImageNet que el modelo carga por defecto.

        libreyolo predict model=LibreCLIPb32-cls.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Embeddings de imagen y texto
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreCLIPb32-cls.pt", task="embed")

        image_embed = model(SAMPLE_IMAGE).embeddings.data

        text_embed = model.embed_text("a photo of a forklift")


        # Ambos están normalizados con L2, así que un simple producto escalar es
        la similitud coseno.

        similarity = (image_embed @ text_embed.T).item()
  val:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreCLIPb32-cls.pt")


        # data es una raíz ImageFolder con un split train/; sus nombres de

        # carpeta se convierten en los prompts de clase zero-shot de esta
        ejecución.

        metrics = model.val(data="imagenette160")


        print(metrics["metrics/accuracy_top1"])

        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreCLIPb32-cls.pt data=imagenette160
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreCLIPb32-cls.pt")
        model.set_classes(["a forklift", "an empty aisle", "a spill"])
        model.export(format="onnx")

        # Las etiquetas actuales de set_classes() y la resolución de entrada
        # quedan fijadas en el grafo. Vuelve a exportar si cambias cualquiera.
    - label: CLI
      language: bash
      code: |
        # Aquí no hay llamada a set_classes(), así que esto fija las 1.000
        # clases de ImageNet por defecto con las que se carga el modelo.
        libreyolo export model=LibreCLIPb32-cls.pt format=onnx
    - label: Exportación de embeddings
      language: python
      code: |
        from libreyolo import LibreYOLO

        # task="embed" traza solo la torre de imagen; no hacen falta clases.
        model = LibreYOLO("LibreCLIPb32-cls.pt", task="embed")
        model.export(format="onnx")
source_hash: ac7cfd75ad6c0fa7
---

## Instalación

CLIP necesita su propio extra, que instala los paquetes que usa su tokenizador BPE incluido para reproducir exactamente los mismos ids de token.

```bash
pip install "libreyolo[clip]"
```

## Predicción

Los pesos se descargan de Hugging Face en el primer uso y quedan cacheados localmente.

<code-tabs name="predict" />

`set_classes()` es la primitiva que convierte esto en un clasificador de vocabulario abierto: inserta cada etiqueta en todas las plantillas de prompt, codifica y promedia los resultados, y cachea la matriz `[K, D]` resultante como cabeza del clasificador, de modo que no se recalcula en cada imagen. Vuelve a llamarla para cambiar las clases en cualquier momento. Si no la llamas, LibreCLIP se carga con los 1.000 nombres de clase de ImageNet-1k ya asignados.

Con `task="embed"`, la predicción devuelve un vector de imagen normalizado con L2 por cada entrada en lugar de probabilidades de clase, y `embed_text()` devuelve filas de texto normalizadas en el mismo espacio vectorial, así que un simple producto escalar entre ambos es la similitud coseno. `iou` no tiene efecto en ninguna de las dos tareas; no hay paso de NMS. Consulta [predicción](/docs/predict) para fuentes, streaming y manejo de resultados.

## Validación

`val()` lee los nombres de las carpetas de clase del split `train/` de un ImageFolder, llama a `set_classes()` con ellos y después mide la precisión zero-shot top-1 y top-5. La precisión depende de cómo se lean los nombres de clase como prompts, no de ninguna actualización de pesos, ya que no hay nada que entrenar. La validación solo cubre `task="classify"`; `task="embed"` no tiene validador de dataset.

<code-tabs name="val" />

## Exportación

<export-matrix />

La exportación fija el estado actual del modelo en un grafo estático. Para `task="classify"`, las últimas etiquetas que asignó `set_classes()` y la resolución en el momento de exportar quedan fijadas en una capa lineal final, de modo que el grafo ONNX o TensorRT exportado es un clasificador de imágenes `[B, K]` corriente, sin torre de texto y sin tokenizador; vuelve a exportar después de cambiar las clases o el tamaño. La exportación con `task="embed"` traza solo la torre de imagen. Ambas necesitan el opset 14 de ONNX o superior, que el exportador establece por defecto.

<code-tabs name="export" />

## Checkpoints

Todos los archivos de pesos publicados de esta familia. Ambos están convertidos a partir de los checkpoints de OpenCLIP entrenados con LAION-2B (`ViT-B-32` y `ViT-B-16`), no de ningún entrenamiento con COCO.

<checkpoint-table />

Los datos de entrenamiento de LAION-2B tienen un historial documentado de contenido CSAM (Stanford Internet Observatory, diciembre de 2023). Desde entonces LAION ha publicado Re-LAION, una reedición limpia; si vuelves a alojar estos pesos por tu cuenta, prefiere los checkpoints derivados de Re-LAION cuando estén disponibles.

## Licencia

<provenance-box></provenance-box>

## Cita

<citation-block />
