---
title: MobileSAM
families: [mobilesam]
seo_title: "MobileSAM: segmentación ligera con prompts en LibreYOLO"
description: "Usa MobileSAM en LibreYOLO para segmentación con prompts de punto y de caja con un encoder TinyViT. Instala y predice con el checkpoint tiny bajo Apache-2.0."
lead: "MobileSAM sustituye el encoder de imagen ViT-H de SAM por un encoder TinyViT destilado, así que el mismo flujo de trabajo con prompts de punto y de caja funciona en hardware más modesto. LibreYOLO incluye un port nativo mediante una factoría LibreSAM dedicada, separada de la factoría de detectores LibreYOLO()."
keywords: [MobileSAM, Segment Anything, TinyViT, "segmentación con prompts", "segmentación interactiva python", "segmentar objeto con un clic", "prompt de punto", "segmentación ligera"]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Prompts de punto y de caja
      language: python
      code: |
        from libreyolo import LibreSAM, SAMPLE_IMAGE

        # MobileSAM tiene un único tamaño, "tiny", así que no hace falta ningún otro alias.
        model = LibreSAM("mobilesam")

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
        from libreyolo import LibreMobileSAM, SAMPLE_IMAGE

        model = LibreMobileSAM()

        # El encoder de imagen es la parte cara. set_image() lo ejecuta una vez;
        # cada llamada posterior a predict() reutiliza el embedding cacheado.
        model.set_image(SAMPLE_IMAGE)
        a = model.predict(points=[640, 420], labels=[1])
        b = model.predict(bboxes=[300, 200, 900, 700])
        model.reset_image()
---

## Instalación

MobileSAM necesita el extra `sam`: la descarga de pesos propia de LibreYOLO
sigue pasando por las herramientas de snapshot de Hugging Face de
`transformers`, aunque la inferencia se ejecute sobre un decoder nativo que no
usa `transformers`.

```bash
pip install "libreyolo[sam]"
```

## Predicción

`LibreSAM(...)` (o el `LibreMobileSAM(...)` específico de la familia) es un
punto de entrada distinto de `LibreYOLO(...)`: devuelve un segmentador guiado
por prompts en lugar de un detector, porque aquí un forward pass no significa
nada sin un prompt espacial. No hay comando de CLI `libreyolo predict` para
esta familia; usa la API de Python.

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
`NotImplementedError` en esta familia: en LibreYOLO, MobileSAM es solo de
predicción. Consulta [predicción](/docs/predict) para los tipos de fuente.

## Variantes

Un único tamaño, tiny, con una entrada fija de 1024 px: MobileSAM se publica
con un solo encoder TinyViT, en lugar de la escala base/large/huge que ofrece
SAM-1.

## Checkpoints

Todos los archivos de pesos publicados de esta familia.

<checkpoint-table />

## Licencia

<provenance-box></provenance-box>

## Cita

<citation-block />
