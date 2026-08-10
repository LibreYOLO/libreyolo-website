---
title: SAM
families: [sam]
seo_title: "SAM (Segment Anything): predecir máscaras en LibreYOLO"
description: "Usa SAM en LibreYOLO para segmentación con prompts de punto y de caja. Instala y predice con los checkpoints base, large y huge bajo Apache-2.0."
lead: "SAM (Segment Anything) convierte un clic de punto o de caja en la máscara de un objeto. LibreYOLO lo carga mediante una factoría LibreSAM dedicada, separada de la factoría de detectores LibreYOLO(), porque un modelo guiado por prompts necesita otra forma de llamada."
keywords: [SAM, Segment Anything, "segmentación con prompts", "segmentación interactiva python", "segmentar objeto con un clic", "prompt de punto", "prompt de caja", Meta AI]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Prompts de punto y de caja
      language: python
      code: |
        from libreyolo import LibreSAM, SAMPLE_IMAGE

        # "base" descarga automáticamente facebook/sam-vit-base en el primer uso.
        # Otros tamaños: "large", "huge" (también "b"/"l"/"h").
        model = LibreSAM("base")

        # Un prompt de punto: [x, y] en coordenadas de píxel, label 1 = primer plano.
        result = model.predict(SAMPLE_IMAGE, points=[640, 420], labels=[1])
        print(result.masks.xy)      # un polígono por máscara
        print(result.boxes.xyxy)    # caja ajustada derivada de la máscara

        # Un prompt de caja en lugar de un punto.
        result = model.predict(SAMPLE_IMAGE, bboxes=[300, 200, 900, 700])

        # Sin ningún prompt se segmenta la imagen entera (un generador
        # automático de máscaras simplificado, no el exhaustivo de referencia).
        result = model.predict(SAMPLE_IMAGE)
    - label: Codifica una vez, lanza muchos prompts
      language: python
      code: |
        from libreyolo import LibreSAM, SAMPLE_IMAGE

        model = LibreSAM("base")

        # El encoder de imagen es la parte cara. set_image() lo ejecuta una vez;
        # cada llamada posterior a predict() reutiliza el embedding cacheado.
        model.set_image(SAMPLE_IMAGE)
        a = model.predict(points=[640, 420], labels=[1])
        b = model.predict(bboxes=[300, 200, 900, 700])
        model.reset_image()
---

## Instalación

SAM necesita el extra `sam`, que arrastra `transformers` y `timm`.

```bash
pip install "libreyolo[sam]"
```

## Predicción

`LibreSAM(...)` es un punto de entrada distinto de `LibreYOLO(...)`: devuelve un
segmentador guiado por prompts en lugar de un detector, porque aquí un forward
pass no significa nada sin un prompt espacial. No hay comando de CLI
`libreyolo predict` para esta familia; usa la API de Python.

<code-tabs name="predict" />

Un prompt de punto acepta `[x, y]` para un objeto, `[[x, y], ...]` para varios,
o arrays de numpy; `labels` marca cada punto con `1` (primer plano) o `0`
(fondo) y por defecto son todos de primer plano. Un prompt de caja toma
`[x1, y1, x2, y2]` o una lista de cajas, con una máscara por caja. Omitir ambos
prompts segmenta la imagen entera lanzando una rejilla densa de prompts y
quedándose con las máscaras de confianza que no se solapan; este modo de
«segmentarlo todo» está simplificado frente al generador automático de máscaras
de referencia y puede infrasegmentar escenas concurridas, así que un prompt real
de punto o de caja es el camino preciso. `conf` filtra por la calidad de máscara
predicha (IoU), no por una confianza de detección: pasa `0.0` para conservar
todos los candidatos. `multimask=True` devuelve las tres máscaras de ambigüedad
todo-frente-a-parte de SAM por cada prompt, en lugar de solo la mejor. `device=`
mueve el modelo y, si hay una sesión de `set_image()` activa, su embedding
cacheado. Cada máscara lleva el id de clase `0`, con nombre `"object"`, porque
una máscara guiada por prompts no tiene un conjunto fijo de clases. `train()`,
`val()`, `export()` y `track()` lanzan todos `NotImplementedError` en esta
familia: SAM es solo de predicción en LibreYOLO, y el seguimiento en vídeo queda
fuera del alcance. Consulta [predicción](/docs/predict) para los tipos de
fuente.

## Variantes

Tres tamaños de encoder de imagen ViT: base, large y huge, todos con una entrada
fija de 1024 px. Todavía no hay publicado ningún benchmark de precisión ni de
latencia para esta familia, así que elegir un tamaño intercambia directamente
peso del encoder por calidad de máscara: base es el más rápido de codificar,
huge el más pesado.

## Licencia

<provenance-box>

LibreYOLO no aloja su propia copia de los pesos de SAM-1. `LibreSAM("base")`,
`"large"` y `"huge"` descargan directamente de los repositorios
`facebook/sam-vit-base`, `facebook/sam-vit-large` y `facebook/sam-vit-huge` de
la propia Meta en Hugging Face, cada uno etiquetado allí como Apache-2.0 de
forma independiente a LibreYOLO.

</provenance-box>

## Cita

<citation-block />
