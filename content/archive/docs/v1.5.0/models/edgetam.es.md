---
title: EdgeTAM
families:
  - edgetam
seo_title: 'EdgeTAM: segmentación con prompts en el dispositivo con LibreYOLO'
description: >-
  Usa EdgeTAM en LibreYOLO para segmentación con prompts de punto y de caja,
  pensada para la velocidad en el dispositivo. Instala y predice con el
  checkpoint bajo Apache-2.0.
lead: >-
  EdgeTAM es una variante de SAM 2 pensada para ejecutarse en el propio
  dispositivo, construida para la velocidad de inferencia en móvil manteniendo
  el mismo flujo de trabajo con prompts de punto y de caja. LibreYOLO admite su
  ruta de segmentación de imágenes mediante una factoría LibreSAM dedicada,
  separada de la factoría de detectores LibreYOLO().
keywords:
  - EdgeTAM
  - SAM 2
  - segmentación con prompts
  - segmentación interactiva python
  - segmentar objeto con un clic
  - segmentación en el dispositivo
  - prompt de punto
  - Meta Reality Labs
last_verified: 1.5.0
snippets:
  predict:
    - label: Prompts de punto y de caja
      language: python
      code: >
        from libreyolo import LibreSAM, SAMPLE_IMAGE


        # EdgeTAM tiene un único tamaño, "edge". Alias: "edgetam", "edge-tam",

        # "edgetam-edge".

        model = LibreSAM("edgetam")


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
      code: |
        from libreyolo import LibreEdgeTAM, SAMPLE_IMAGE

        model = LibreEdgeTAM()

        # El encoder de imagen es la parte cara. set_image() lo ejecuta una vez;
        # cada llamada posterior a predict() reutiliza el embedding cacheado.
        model.set_image(SAMPLE_IMAGE)
        a = model.predict(points=[640, 420], labels=[1])
        b = model.predict(bboxes=[300, 200, 900, 700])
        model.reset_image()
source_hash: e6cce8faad18e73d
---

## Instalación

EdgeTAM necesita el extra `sam`, que arrastra `transformers` y `timm`.

```bash
pip install "libreyolo[sam]"
```

## Predicción

`LibreSAM(...)` (o el `LibreEdgeTAM(...)` específico de la familia) es un punto
de entrada distinto de `LibreYOLO(...)`: devuelve un segmentador guiado por
prompts en lugar de un detector, porque aquí un forward pass no significa nada
sin un prompt espacial. No hay comando de CLI `libreyolo predict` para esta
familia; usa la API de Python. Solo se admite la segmentación de imágenes; el
seguimiento en vídeo de EdgeTAM queda fuera del alcance aquí.

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

Un único tamaño, edge, con una resolución de entrada fija, así que elegir esta
familia frente al resto del nivel SAM es una decisión de hardware más que de
tamaño: EdgeTAM existe específicamente para la inferencia en el propio
dispositivo, con recursos limitados.

## Checkpoints

Todos los archivos de pesos publicados de esta familia.

<checkpoint-table />

## Licencia

<provenance-box></provenance-box>

## Cita

<citation-block />
