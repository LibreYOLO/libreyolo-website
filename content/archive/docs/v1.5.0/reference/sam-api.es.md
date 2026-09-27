---
title: API de segmentación con prompts
seo_title: 'API de LibreSAM: prompts, alias y firmas'
description: >-
  La factoría LibreSAM, sus alias de tamaño, los tipos de prompt de punto, de
  caja y de texto de concepto, el ciclo de vida de set_image que codifica una
  sola vez, y lo que el tier no soporta.
lead: >-
  LibreSAM es la factoría para la segmentación con prompts. Un forward pass
  necesita un prompt por imagen que se pasa en el momento de la llamada, así que
  el tier tiene su propia superficie de predict en lugar de enrutar a través del
  runner de inferencia sin prompts.
keywords:
  - LibreSAM
  - segmentación con prompts
  - prompt de punto SAM
  - prompt de caja SAM
  - set_image
  - segmentarlo todo SAM
  - extra sam de libreyolo
last_verified: 1.5.0
verification: >-
  Alias, tamaños y repositorios de la factoría leídos de
  libreyolo/models/sam/model.py, sam2.py, edgetam.py, sam3.py,
  libreyolo/models/mobilesam/model.py y libreyolo/models/picosam3/model.py.
  Contrato de prompts y valores por defecto leídos de
  libreyolo/models/sam/base.py. Intención de diseño según
  docs/adr/0007-libresam-contract.md, todo en la v1.5.0.
snippets:
  install:
    - label: bash
      language: bash
      code: |
        pip install 'libreyolo[sam]'
  usage:
    - label: Prompts de punto y de caja
      language: python
      code: |
        from libreyolo import LibreSAM, SAMPLE_IMAGE

        model = LibreSAM("base")

        r = model.predict(SAMPLE_IMAGE, points=[900, 370], labels=[1])
        print(r.masks.xy)
        print(r.boxes.xyxy)

        r = model.predict(SAMPLE_IMAGE, bboxes=[100, 100, 200, 200])
        print(len(r))
    - label: 'Codificar una vez, lanzar muchos prompts'
      language: python
      code: |
        from libreyolo import LibreSAM, SAMPLE_IMAGE

        model = LibreSAM("base")
        model.set_image(SAMPLE_IMAGE)

        a = model.predict(points=[500, 375], labels=[1])
        b = model.predict(bboxes=[100, 100, 200, 200])
        print(len(a), len(b))

        model.reset_image()
source_hash: 18e8206c10ce17fd
---

## Instalación

El tier necesita el extra `sam`.

<code-tabs name="install" />

## La factoría

```python
LibreSAM(model: str = "base", **kwargs) -> LibreSAMModel
```

`model` es un alias de tamaño, no una ruta. `**kwargs` llega al constructor de
la familia, que acepta `device` y `multimask`. Un alias desconocido lanza
`ValueError`, y el mensaje enumera todos los alias conocidos.

<code-tabs name="usage" />

## Alias

| Familia | Alias | Tamaños | Pesos |
|---|---|---|---|
| SAM-1 | `base`, `large`, `huge`, `b`, `l`, `h`, `sam-base`, `sam-large`, `sam-huge`, `sam_b`, `sam_l`, `sam_h` | `base`, `large`, `huge` | `facebook/sam-vit-base`, `-large`, `-huge` |
| SAM-2 | `sam2-tiny`, `sam2-small`, `sam2-base-plus`, `sam2-baseplus`, `sam2-large`, y las formas cortas `sam2-t`, `sam2-s`, `sam2-bp`, `sam2-l`, `sam2_t`, `sam2_s`, `sam2_bp`, `sam2_l` | `tiny`, `small`, `base-plus`, `large` | `LibreYOLO/LibreSAM2tiny`, `-small`, `-base-plus`, `-large` |
| EdgeTAM | `edgetam`, `edge-tam`, `edgetam-edge` | `edge` | `LibreYOLO/LibreEdgeTAM` |
| SAM 3 | `sam3`, `sam-3`, `sam3-large` | `large` | `facebook/sam3` |
| MobileSAM | `mobilesam`, `mobilesam-tiny`, `mobilesam_t`, `mobile-sam`, `mobile-sam-tiny` | `tiny` | `LibreYOLO/LibreMobileSAM` |
| PicoSAM3 | `picosam3`, `picosam3-pico`, `picosam3_pico`, `pico-sam3` | `pico` | `LibreYOLO/LibrePicoSAM3` |

El valor por defecto es `base`. SAM-1, SAM-2, EdgeTAM y MobileSAM trabajan
sobre un lienzo nominal de 1024 píxeles, SAM 3 sobre uno de 1008 y PicoSAM3
sobre uno de 96.

Los pesos de SAM 3 están restringidos. Se descargan de `facebook/sam3` bajo la
SAM License propia de Meta, que no es ni MIT ni Apache-2.0 y que LibreYOLO no
redistribuye. Acepta los términos en la página del repositorio y autentícate
con Hugging Face antes de cargarlo; el loader registra el aviso primero.

Las clases de familia también se exportan, así que `LibreSAM1`, `LibreSAM2`,
`LibreSAM3`, `LibreEdgeTAM`, `LibreMobileSAM` y `LibrePicoSAM3` se pueden
construir directamente con `size=`.

## predict

```python
model.predict(
    source=None,
    *,
    points=None,
    bboxes=None,
    labels=None,
    masks=None,
    text=None,
    conf=None,
    multimask=None,
    max_det=300,
    device=None,
    color_format="auto",
    points_per_side=None,
) -> Results
```

| Argumento | Por defecto | Significado |
|---|---|---|
| `source` | `None` | Imagen que segmentar; `None` reutiliza la imagen cacheada por `set_image()` |
| `points` | `None` | Prompt de punto en coordenadas de píxel |
| `bboxes` | `None` | Prompt de caja como `[x1, y1, x2, y2]`, o una lista de ellas para una máscara por caja |
| `labels` | `None` | Etiquetas de los puntos, `1` positivo y `0` negativo, con la forma que corresponda a `points`; todas positivas si se omite |
| `masks` | `None` | Reservado; pasar una lanza `NotImplementedError` |
| `text` | `None` | Prompt de concepto; solo SAM 3 |
| `conf` | `None` | Suelo del IoU de máscara predicho |
| `multimask` | `None` | Devuelve todas las máscaras de ambigüedad por prompt; por defecto toma el ajuste de construcción |
| `max_det` | `300` | Límite de máscaras devueltas |
| `device` | `None` | Mueve el modelo para esta llamada y las siguientes, invalidando los embeddings cacheados |
| `color_format` | `"auto"` | Pista de formato de color para arrays en memoria |
| `points_per_side` | `None` | Densidad de la rejilla para segmentarlo todo; por defecto 32 |

Lo que se devuelve es un `Results` normal que lleva `masks`, más unos `boxes`
ajustados derivados de esas máscaras, con la clase `0` llamada `"object"`.

## Formas de los prompts

`points` acepta las formas anidadas `[x, y]` para un objeto, `[[x, y], ...]`
para N objetos y `[[[x, y], ...], ...]` para puntos agrupados por objeto. Los
arrays de numpy funcionan en todos los sitios donde funciona una lista. Las
coordenadas son píxeles sin más sobre la imagen de origen.

Omitir todos los prompts espaciales ejecuta el modo de segmentarlo todo, un
generador automático de máscaras por rejilla con un umbral de IoU predicho y
deduplicación por IoU de caja. El `points_per_side` por defecto de 32 lanza
unas 1024 pasadas del decoder, lo que resulta lento en CPU; bájalo para uso
interactivo. El generador se salta el filtrado por stability score, el
multi-crop y la deduplicación por IoU de máscara, así que es una aproximación
al camino con prompts más que un equivalente suyo.

## Confianza

`conf` filtra por el IoU de máscara predicho, que es una puntuación de calidad
de máscara y no una confianza de detección. `None` conserva todas las máscaras
en el camino con prompts y aplica el umbral de rejilla de la familia al
segmentarlo todo. `0.0` desactiva el filtrado en cualquiera de los dos modos.

En el camino de texto de SAM 3, `conf` pasa a ser la puntuación de detección de
Promptable Concept Segmentation. Ahí `None` significa el umbral estándar de
0.3, y `0.0` conserva todos los candidatos.

## Prompts de texto

`text=` es solo de SAM 3; todas las familias de prompts espaciales lanzan
`NotImplementedError` con él. El texto es mutuamente excluyente con los puntos
y las cajas. El `names` devuelto asigna la clase `0` al concepto solicitado.
Una llamada con texto y `source=None` vuelve a codificar la imagen cacheada,
porque el tracker y el encoder de conceptos no comparten caché.

El argumento `exemplars=` está reservado para una futura extensión de
ejemplares de imagen y no está implementado.

## El ciclo de vida de codificar una sola vez

```python
model.set_image(source, color_format="auto") -> LibreSAMModel
model.reset_image() -> LibreSAMModel
```

`set_image` ejecuta una sola vez el pesado encoder de imagen y cachea los
embeddings, así que cada `predict()` posterior con `source=None` sale barato.
Ambos métodos devuelven el modelo, de modo que las llamadas se pueden
encadenar. Pasar `device=` a `predict` mueve el modelo e invalida la caché.

## PicoSAM3

PicoSAM3 solo acepta `bboxes=`. Los prompts de punto, de texto y de máscara,
`multimask` y el segmentarlo todo lanzan. La caja se expande un 10 por ciento y
se pasa por una red de ROI de 96 píxeles, y PicoSAM3 es la única familia del
tier que exporta, y solo a ONNX.

## No soportado

`train()`, `val()` y `track()` lanzan `NotImplementedError` en todas las
familias del tier. Las máscaras con prompts no tienen un conjunto fijo de
clases contra el que puntuar, así que aquí el mAP no significa nada.
`export()` lanza en SAM-1, SAM-2, SAM 3, EdgeTAM y MobileSAM.

Los caminos de vídeo y de memoria de SAM-2, SAM 3 y EdgeTAM quedan fuera del
alcance de esta versión, igual que los ejemplares de imagen y los prompts de
máscara de SAM 3.
