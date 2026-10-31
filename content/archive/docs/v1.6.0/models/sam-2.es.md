---
title: SAM 2
families:
  - sam2
seo_title: 'SAM 2: segmentación de imágenes con prompts en LibreYOLO'
description: >-
  Usa SAM 2 en LibreYOLO para segmentación con prompts de punto y de caja.
  Instala y predice con los checkpoints tiny, small, base-plus y large, bajo
  Apache-2.0.
lead: >-
  SAM 2 amplía SAM con una arquitectura de memoria en streaming construida para
  vídeo, y convierte un clic de punto o de caja en una máscara de objeto.
  LibreYOLO admite su ruta de segmentación de imágenes mediante una factoría
  LibreSAM dedicada, separada de la factoría de detectores LibreYOLO().
keywords:
  - SAM 2
  - Segment Anything
  - segmentación con prompts
  - segmentación interactiva python
  - segmentar objeto con un clic
  - prompt de punto
  - prompt de caja
  - Meta AI
  - Hiera
last_verified: 1.5.0
snippets:
  predict:
    - label: Prompts de punto y de caja
      language: python
      code: >
        from libreyolo import LibreSAM, SAMPLE_IMAGE


        # Alias de tamaño: "sam2-tiny", "sam2-small", "sam2-base-plus",

        # "sam2-large" (también las formas cortas
        "sam2-t"/"sam2-s"/"sam2-bp"/"sam2-l").

        model = LibreSAM("sam2-large")


        # Un prompt de punto: [x, y] en coordenadas de píxel, label 1 = primer
        plano.

        result = model.predict(SAMPLE_IMAGE, points=[640, 420], labels=[1])

        print(result.masks.xy)      # un polígono por máscara

        print(result.boxes.xyxy)    # caja ajustada derivada de la máscara


        # Un prompt de caja en lugar de un punto.

        result = model.predict(SAMPLE_IMAGE, bboxes=[300, 200, 900, 700])


        # Sin ningún prompt se segmenta la imagen entera (un generador

        # automático de máscaras simplificado, no el exhaustivo de referencia).

        result = model.predict(SAMPLE_IMAGE)
    - label: 'Codifica una vez, lanza muchos prompts'
      language: python
      code: >
        from libreyolo import LibreSAM2, SAMPLE_IMAGE


        # La clase específica de la familia toma el tamaño sin el prefijo
        "sam2-".

        model = LibreSAM2("large")


        # El encoder de imagen es la parte cara. set_image() lo ejecuta una vez;

        # cada llamada posterior a predict() reutiliza el embedding cacheado.

        model.set_image(SAMPLE_IMAGE)

        a = model.predict(points=[640, 420], labels=[1])

        b = model.predict(bboxes=[300, 200, 900, 700])

        model.reset_image()
source_hash: 2a3090d7ecd533b0
---

## Instalación

SAM 2 necesita el extra `sam`, que arrastra `transformers` y `timm`.

```bash
pip install "libreyolo[sam]"
```

## Predicción

`LibreSAM(...)` (o el `LibreSAM2(...)` específico de la familia) es un punto de
entrada distinto de `LibreYOLO(...)`: devuelve un segmentador guiado por
prompts en lugar de un detector, porque aquí un forward pass no significa nada
sin un prompt espacial. No hay comando de CLI `libreyolo predict` para esta
familia; usa la API de Python. Solo se admite la segmentación de imágenes; el
seguimiento con memoria de vídeo de SAM 2 queda fuera del alcance aquí.

<code-tabs name="predict" />

Un prompt de punto acepta `[x, y]` para un objeto, `[[x, y], ...]` para varios,
o arrays de numpy; `labels` marca cada punto con `1` (primer plano) o `0`
(fondo) y por defecto son todos de primer plano. Un prompt de caja toma
`[x1, y1, x2, y2]` o una lista de cajas, con una máscara por caja. Omitir
ambos prompts segmenta la imagen entera lanzando una rejilla densa de prompts
y quedándose con las máscaras de confianza que no se solapan; este modo de
«segmentarlo todo» está simplificado frente al generador automático de
máscaras de referencia y puede infrasegmentar escenas concurridas, así que un
prompt real de punto o de caja es el camino preciso. `conf` filtra por la
calidad de máscara predicha (IoU), no por una confianza de detección: pasa
`0.0` para conservar todos los candidatos. `multimask=True` devuelve las tres
máscaras de ambigüedad todo-frente-a-parte de SAM por cada prompt, en lugar de
solo la mejor. `device=` mueve el modelo y, si hay una sesión de `set_image()`
activa, su embedding cacheado. Cada máscara lleva el id de clase `0`, con
nombre `"object"`, porque una máscara guiada por prompts no tiene un conjunto
fijo de clases. `train()`, `val()`, `export()` y `track()` lanzan todos
`NotImplementedError` en esta familia: aquí lo que LibreYOLO admite es la
inferencia sobre imágenes. Consulta [predicción](/docs/predict) para los tipos
de fuente.

## Variantes

Cuatro tamaños con backbone Hiera: tiny, small, base-plus y large, todos a la
misma resolución de entrada. Todavía no hay publicado ningún benchmark de
precisión ni de latencia para esta familia, así que elegir un tamaño supone
cambiar directamente peso del encoder por calidad de máscara: tiny es el más
rápido de codificar, large el más pesado.

## Checkpoints

Todos los archivos de pesos publicados de esta familia.

<checkpoint-table />

## Licencia

<provenance-box></provenance-box>

## Cita

<citation-block />
