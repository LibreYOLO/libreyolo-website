# Spanish translation style guide — LibreYOLO docs

How to produce `<slug>.es.md` twins of the English docs in `content/docs/`.
The docs loader (`src/lib/docs.js`) picks up the twin automatically; the file
must be a complete, standalone copy of the English file with the rules below.

## Voice and register

- Neutral, international Spanish: understandable in Madrid, Mexico City and
  Buenos Aires alike. No `vosotros`; address the reader as `tú` (the standard
  register of modern Spanish developer docs), or use impersonal constructions
  where they read better.
- Technical, direct, unadorned — mirror the tone of the English source. Do not
  add, remove, summarize or "improve" content. Every paragraph, list item,
  table row, admonition and FAQ entry in the English file appears in the
  Spanish file, in the same order.
- Spell out what the English spells out; keep the same level of formality.

## What to translate

- Frontmatter: `title`, `seo_title`, `description`, `lead`, and `faq` (both
  `q` and `a`). Quote YAML strings that contain `:` or other special chars.
- Reader-visible prose values in `meta` and `verification` frontmatter blocks
  (labels like "Install" → "Instalación", prose values like "3.10 or newer" →
  "3.10 o superior"). Values marked `mono` or that are code/identifiers stay.
- Snippet and tab `label` values that are descriptive phrases ("Video and
  streams" → "Vídeo y streams", "Check a dataset" → "Comprobar un dataset").
  Labels that are proper names stay untouched: `Python`, `CLI`, `Bash`,
  library, tool and format names.
- `keywords`: **localize, don't translate literally.** Write the queries a
  Spanish-speaking developer would actually type — a mix of Spanish phrases
  and the English terms they'd realistically use (e.g.
  `"detección de objetos python"`, `"entrenar yolo dataset propio"`,
  `"exportar yolo onnx"`). Keep model names as-is.
- Body prose, headings, table cells, image alt text, admonitions, and
  `hero.caption` (the media `src`/`poster` paths stay).
- Comments **inside** code blocks and snippets (`# descarga automática...`).

## What stays exactly as in English

- All code (statements, identifiers, strings, CLI commands, file names, URLs)
  — only comments change. Never translate string literals passed to code.
- Frontmatter keys themselves, their order, and the structural fields:
  `families`, `last_verified`, `snippets` structure (keys, order, count),
  `language` values, hero/media paths.
- Internal and external link **targets** (translate link text, never hrefs).
- Model names (YOLOv9, RF-DETR, RTMDet...), product names, library names,
  format names (ONNX, TensorRT, CoreML), and heading anchors implied by them.

## Glossary (use these consistently)

| English | Spanish |
|---|---|
| library | la biblioteca (never "librería" — false friend) |
| weights | los pesos |
| checkpoint | el checkpoint |
| dataset | el dataset |
| training / to train | el entrenamiento / entrenar |
| fine-tuning / to fine-tune | el fine-tuning / hacer fine-tuning (or ajustar) |
| inference | la inferencia |
| object detection | detección de objetos |
| bounding box | el bounding box |
| instance segmentation | segmentación de instancias |
| semantic segmentation | segmentación semántica |
| pose estimation | estimación de pose |
| depth estimation | estimación de profundidad |
| embedding | el embedding |
| backbone | el backbone |
| head (of a model) | la cabeza (del modelo) |
| threshold | el umbral |
| confidence | la confianza |
| accuracy | la precisión |
| batch / batch size | el batch / el tamaño de batch |
| learning rate | el learning rate |
| data augmentation | el aumento de datos (data augmentation) |
| quantization | la cuantización |
| export / to export | la exportación / exportar |
| deployment / to deploy | el despliegue / desplegar |
| pretrained | preentrenado |
| label (annotation) | la etiqueta |
| ground truth | el ground truth |
| feature map | el mapa de características |
| open-vocabulary detection | detección de vocabulario abierto |
| edge device | el dispositivo edge |
| frame rate / FPS | los FPS |
| layer | la capa |
| loss | la loss (la función de pérdida on first mention) |

`mAP`, `mAP50`, `IoU`, `NMS`, `ONNX`, `GPU`, `CPU`, `CUDA` stay untouched.

## Formatting invariants (validated by script)

- Same number of headings, at the same levels, in the same order.
- Same number of fenced code blocks; code identical except comment lines.
- Same set of link targets.
- Same frontmatter keys as the English file — no keys added or dropped.
- Do not add translator notes, disclaimers, or a "translated from" line.
