---
title: SAM 3
families: [sam3]
seo_title: "SAM 3: segmentación con prompts y por concepto en LibreYOLO"
description: "Usa SAM 3 en LibreYOLO para segmentación por punto, por caja y por concepto de texto. Instala y predice con el checkpoint large, restringido bajo la SAM License de Meta."
lead: "SAM 3 amplía SAM con un prompt de concepto en texto además de los puntos y cajas habituales, así que una frase como \"yellow school bus\" devuelve todas las instancias que coinciden. LibreYOLO soporta su camino de imagen mediante una factoría LibreSAM dedicada, separada de la factoría de detectores LibreYOLO()."
keywords: [SAM 3, Segment Anything, "segmentación con prompts", "segmentación por concepto", "segmentar con texto", "prompt de texto", "prompt de punto", "segmentar objeto con un clic", Meta AI]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Prompts de punto y de caja
      language: python
      code: |
        from libreyolo import LibreSAM, SAMPLE_IMAGE

        # "sam3" es el único tamaño ("large"); alias: "sam3", "sam-3", "sam3-large".
        model = LibreSAM("sam3")

        # Un prompt de punto: [x, y] en coordenadas de píxel, label 1 = primer plano.
        result = model.predict(SAMPLE_IMAGE, points=[640, 420], labels=[1])
        print(result.masks.xy)      # un polígono por máscara
        print(result.boxes.xyxy)    # caja ajustada derivada de la máscara

        # Un prompt de caja en lugar de un punto.
        result = model.predict(SAMPLE_IMAGE, bboxes=[300, 200, 900, 700])
    - label: Prompt de texto (concepto)
      language: python
      code: |
        from libreyolo import LibreSAM3, SAMPLE_IMAGE

        model = LibreSAM3("large")

        # Encuentra todas las instancias que coinciden con la frase, no solo un objeto.
        # text= es mutuamente excluyente con points, bboxes, labels y masks.
        result = model.predict(SAMPLE_IMAGE, text="a person")
        print(result.names)         # {0: "a person"}
        print(result.boxes.conf)    # la puntuación de detección PCS por instancia
    - label: Codifica una vez, lanza muchos prompts
      language: python
      code: |
        from libreyolo import LibreSAM3, SAMPLE_IMAGE

        model = LibreSAM3("large")

        # El encoder de imagen es la parte cara. set_image() lo ejecuta una vez;
        # cada llamada posterior a predict() reutiliza el embedding cacheado. Una
        # llamada con text= vuelve a codificar internamente, ya que el tracker y
        # el encoder de segmentación por concepto no comparten caché.
        model.set_image(SAMPLE_IMAGE)
        a = model.predict(points=[640, 420], labels=[1])
        b = model.predict(bboxes=[300, 200, 900, 700])
        model.reset_image()
---

## Instalación

SAM 3 necesita el extra `sam`, que arrastra `transformers` y `timm`.

```bash
pip install "libreyolo[sam]"
```

Los pesos están restringidos: visita
[huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3), acepta la
SAM License de Meta y luego ejecuta `hf auth login` (o define `HF_TOKEN`) antes
de la primera descarga. LibreYOLO registra un aviso de licencia la primera vez
que descarga esta familia.

## Predicción

`LibreSAM(...)` (o el `LibreSAM3(...)` específico de la familia) es un punto de
entrada distinto de `LibreYOLO(...)`: devuelve un segmentador guiado por
prompts en lugar de un detector, porque aquí un forward pass no significa nada
sin un prompt. No hay comando de CLI `libreyolo predict` para esta familia; usa
la API de Python. Solo se admite inferencia sobre imágenes; los modelos de
vídeo de SAM 3 quedan fuera del alcance aquí.

<code-tabs name="predict" />

El camino de punto y caja coincide con el resto de la familia SAM: un prompt de
punto acepta `[x, y]` para un objeto o `[[x, y], ...]` para varios, `labels`
marca cada punto con `1` (primer plano) o `0` (fondo), y un prompt de caja toma
`[x1, y1, x2, y2]` o una lista de cajas. En este camino, `conf` filtra por la
calidad de máscara predicha (IoU), no por una confianza de detección.

El camino de `text=` es lo que añade SAM 3: una cadena de concepto devuelve
todas las instancias coincidentes de la imagen mediante Promptable Concept
Segmentation, y no se puede combinar con puntos, cajas, labels o máscaras. Ahí
`conf` es la puntuación de detección PCS en lugar del IoU de la máscara;
dejarlo en su valor por defecto aplica el propio umbral de 0.3 del modelo, y
`conf=0.0` conserva todos los candidatos. El `names` devuelto asigna el id de
clase `0` a la cadena de concepto solicitada, ya que una máscara guiada por
prompts no tiene por lo demás ningún conjunto fijo de clases. `device=` mueve
el modelo y, si hay una sesión de `set_image()` activa, su embedding cacheado.
`train()`, `val()`, `export()` y `track()` lanzan todos `NotImplementedError`
en esta familia: en LibreYOLO, SAM 3 es solo de predicción, y el seguimiento en
vídeo queda fuera de alcance. Consulta [predicción](/docs/predict) para los
tipos de fuente.

## Variantes

Un único tamaño, large, con una entrada fija de 1008 px. SAM 3.1 no está
soportado: su implementación lleva una licencia propia que no se puede
incorporar a este repositorio MIT, y la versión de Transformers de la que
depende LibreYOLO todavía no carga su formato de checkpoint.

## Licencia

<provenance-box>

LibreYOLO no aloja una copia propia de los pesos de SAM 3 y no los
redistribuye. `LibreSAM("sam3")` descarga directamente del repositorio
restringido `facebook/sam3` de Meta en Hugging Face, que exige aceptar la SAM
License de Meta y autenticarse antes de la primera descarga.

</provenance-box>

## Cita

<citation-block />
