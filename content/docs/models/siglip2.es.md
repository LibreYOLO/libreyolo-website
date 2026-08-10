---
title: SigLIP2
families: [siglip2]
seo_title: "SigLIP2 en LibreYOLO: clasificación y embeddings zero-shot"
description: "Usa SigLIP2 en LibreYOLO para clasificación de imágenes zero-shot y embeddings de imagen y texto, con puntuación sigmoide multi-etiqueta. Sin entrenamiento."
lead: "SigLIP2 es un modelo de doble torre que puntúa una imagen frente a prompts de texto con una sigmoide independiente por clase, en lugar de un softmax compartido sobre un conjunto fijo de etiquetas. LibreYOLO lo soporta para clasificación zero-shot y embeddings de imagen y texto, sin ningún paso de entrenamiento."
keywords: [SigLIP2, SigLIP 2, "clasificación zero-shot", "clasificar imágenes sin entrenar", "embeddings de imágenes python", "embeddings de texto", "clasificación multietiqueta", "vocabulario abierto", "modelo multilingüe imagen texto", sigmoid loss]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreSigLIP2b16-cls.pt")
        model.set_classes(["a forklift", "an empty aisle", "a spill"])
        result = model(SAMPLE_IMAGE, save=True)

        print(model.names[result.probs.top1], float(result.probs.top1conf))
    - label: CLI
      language: bash
      code: |
        # Sin una llamada a set_classes(), predict desde la CLI usa los 1.000
        # nombres de clase de ImageNet que el modelo carga por defecto.
        libreyolo predict model=LibreSigLIP2b16-cls.pt source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg save=True
    - label: Puntuación sigmoide multi-etiqueta
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreSigLIP2b16-cls.pt")
        model.set_classes(["a dog", "a cat", "outdoors"], multi_label=True)
        r = model(SAMPLE_IMAGE)

        # Probabilidades independientes por clase: más de una, o ninguna,
        # puede puntuar alto a la vez. El softmax (el valor por defecto) las
        # normaliza en una distribución de etiqueta única, igual que LibreCLIP.
        for i, name in model.names.items():
            print(name, float(r.probs.data[i]))
    - label: Embeddings de imagen y texto
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreSigLIP2b16-cls.pt", task="embed")
        image_embed = model(SAMPLE_IMAGE).embeddings.data
        text_embed = model.embed_text("a photo of a forklift")

        # Ambos están normalizados con L2, así que un simple producto escalar es la similitud coseno.
        similarity = (image_embed @ text_embed.T).item()
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSigLIP2b16-cls.pt")

        # data es una raíz ImageFolder con un split train/; sus nombres de
        # carpeta se convierten en los prompts de clase zero-shot de esta ejecución.
        metrics = model.val(data="imagenette160")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreSigLIP2b16-cls.pt data=imagenette160
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSigLIP2b16-cls.pt")
        model.set_classes(["a forklift", "an empty aisle", "a spill"])
        model.export(format="onnx")

        # Las etiquetas actuales de set_classes() y la resolución de entrada
        # quedan fijadas en el grafo. Vuelve a exportar si cambias cualquiera.
        # multi_label debe ser False (el valor por defecto) al exportar.
    - label: CLI
      language: bash
      code: |
        # Aquí no hay llamada a set_classes(), así que esto fija las 1.000
        # clases de ImageNet por defecto con las que se carga el modelo.
        libreyolo export model=LibreSigLIP2b16-cls.pt format=onnx
    - label: Exportación de embeddings
      language: python
      code: |
        from libreyolo import LibreYOLO

        # task="embed" traza solo la torre de imagen; no hacen falta clases.
        model = LibreYOLO("LibreSigLIP2b16-cls.pt", task="embed")
        model.export(format="onnx")
---

## Instalación

SigLIP2 necesita su propio extra, que instala el paquete SentencePiece que usa su tokenizador multilingüe.

```bash
pip install "libreyolo[siglip2]"
```

## Predicción

Los pesos se descargan de Hugging Face en el primer uso y quedan cacheados localmente.

<code-tabs name="predict" />

`set_classes()` es la primitiva que convierte esto en un clasificador de vocabulario abierto: inserta cada etiqueta en todas las plantillas de prompt, codifica y promedia los resultados, y cachea la matriz `[K, D]` resultante como cabeza del clasificador, de modo que no se recalcula en cada imagen. Vuelve a llamarla para cambiar las clases en cualquier momento. Si no la llamas, LibreSigLIP2 se carga con los 1.000 nombres de clase de ImageNet-1k ya asignados.

SigLIP puntúa cada clase de forma independiente: `logit = scale * (image . text) + bias`. Por defecto, ese conjunto de logits sigue pasando por un softmax, lo que da una distribución de etiqueta única que coincide con el comportamiento de `top1`/`top5` de LibreCLIP. Pasar `multi_label=True` a `set_classes()` (o en la construcción) cambia a probabilidades sigmoides independientes, de modo que más de una clase, o ninguna, puede puntuar alto en la misma imagen. El tokenizador es un modelo SentencePiece multilingüe (vocabulario de Gemma), así que los nombres de clase en idiomas distintos del inglés funcionan igual.

Con `task="embed"`, la predicción devuelve un vector de imagen normalizado con L2 por cada entrada en lugar de probabilidades de clase, y `embed_text()` devuelve filas de texto normalizadas en el mismo espacio vectorial, así que un simple producto escalar entre ambos es la similitud coseno. `iou` no tiene efecto en ninguna de las dos tareas; no hay paso de NMS. Consulta [predicción](/docs/predict) para fuentes, streaming y manejo de resultados.

## Validación

`val()` lee los nombres de las carpetas de clase del split `train/` de un ImageFolder, llama a `set_classes()` con ellos y después mide la precisión zero-shot top-1 y top-5 con puntuación softmax. La precisión depende de cómo se lean los nombres de clase como prompts, no de ninguna actualización de pesos, ya que no hay nada que entrenar. La validación solo cubre `task="classify"`; `task="embed"` no tiene validador de dataset.

<code-tabs name="val" />

## Exportación

<export-matrix />

La exportación fija el estado actual del modelo en un grafo estático. Para `task="classify"`, las últimas etiquetas que asignó `set_classes()` y la resolución en el momento de exportar quedan fijadas en una capa lineal final con la escala y el bias aprendidos, de modo que el grafo exportado es un clasificador de imágenes `[B, K]` corriente, sin torre de texto y sin tokenizador; vuelve a exportar después de cambiar las clases o el tamaño. La exportación en modo `multi_label=True` no está implementada; vuelve a ponerlo en `False` primero. La exportación con `task="embed"` traza solo la torre de imagen. Ambas necesitan el opset 14 de ONNX o superior, que el exportador establece por defecto.

<code-tabs name="export" />

## Checkpoints

Todos los archivos de pesos publicados de esta familia. Ambos están convertidos a partir de los checkpoints `siglip2-base-patch16-256` y `siglip2-so400m-patch14-384` de Google, con licencia Apache-2.0, no de ningún entrenamiento con COCO.

<checkpoint-table />

## Licencia

<provenance-box></provenance-box>

## Cita

<citation-block />
