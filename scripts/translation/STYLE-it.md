# Italian translation style guide — LibreYOLO docs

How to produce `<slug>.it.md` twins of the English docs in `content/docs/`.
The docs loader (`src/lib/docs.js`) picks up the twin automatically; the file
must be a complete, standalone copy of the English file with the rules below.

## Voice and register

- Standard modern Italian, the register of Italian developer documentation:
  address the reader as `tu` (`installa`, `esegui`, `puoi`), never `Lei` and
  never `voi`. Impersonal constructions (`si installa con`, `basta passare`)
  are fine and often read better in reference prose — alternate freely, but
  never switch to `Lei` mid-page.
- Technical, direct, unadorned — mirror the tone of the English source. Do not
  add, remove, summarize or "improve" content. Every paragraph, list item,
  table row, admonition and FAQ entry in the English file appears in the
  Italian file, in the same order.
- Avoid bureaucratic Italian. Write `per` not `al fine di`, `usa` not
  `provvedere all'utilizzo di`, `dopo` not `successivamente a`, `con` not
  `mediante`. Prefer verbs to nominalizations (`quando esporti` beats `in fase
  di esportazione`) and keep sentences roughly as long as the English ones.
- Spell out what the English spells out; keep the same level of formality.
- Write accents and elisions properly: `è`, `perché`, `più`, `così`, `qual è`
  (no apostrophe), `un'API` / `un'immagine` (feminine elision) but `un backbone`
  / `un embedding` (masculine, no apostrophe).

## What to translate

- Frontmatter: `title`, `seo_title`, `description`, `lead`, and `faq` (both
  `q` and `a`). Quote YAML strings that contain `:` or other special chars.
- Reader-visible prose values in `meta` and `verification` frontmatter blocks
  (labels like "Install" → "Installazione", prose values like "3.10 or newer" →
  "3.10 o superiore"). Values marked `mono` or that are code/identifiers stay.
- Snippet and tab `label` values that are descriptive phrases ("Video and
  streams" → "Video e stream", "Check a dataset" → "Controllare un dataset").
  Labels that are proper names stay untouched: `Python`, `CLI`, `Bash`,
  library, tool and format names.
- `keywords`: **localize, don't translate literally.** Write the queries an
  Italian-speaking developer would actually type — a mix of Italian phrases
  and the English terms they'd realistically use (e.g.
  `"object detection python"`, `"addestrare yolo dataset personalizzato"`,
  `"esportare yolo onnx"`). Italian developers search in English more often
  than in Italian for model and task names: keep those in English inside
  `keywords` even where the body prose translates them. Model names stay as-is.
- Body prose, headings, table cells, image alt text, admonitions, and
  `hero.caption` (the media `src`/`poster` paths stay).
- Comments **inside** code blocks and snippets (`# download automatico...`).

## What stays exactly as in English

- All code (statements, identifiers, strings, CLI commands, file names, URLs)
  — only comments change. Never translate string literals passed to code.
- Frontmatter keys themselves, their order, and the structural fields:
  `families`, `last_verified`, `snippets` structure (keys, order, count),
  `language` values, hero/media paths.
- Internal and external link **targets** (translate link text, never hrefs).
- Model names (YOLOv9, RF-DETR, RTMDet...), product names, library names,
  format names (ONNX, TensorRT, CoreML), and heading anchors implied by them.
- Numbers inside metrics, code and CLI arguments keep the English decimal
  point: `0.001 mAP`, `lr0=0.004`, `conf=0.25`. Only decimals in running prose
  may take the Italian comma, so in practice they almost never change.
- Loanwords this project keeps in English (see the glossary): they are written
  without italics, without quotes, and **invariable in the plural** — `i
  dataset`, `i checkpoint`, `i bounding box`, `gli embedding`, `i layer`, `i
  backbone`, `le pipeline`, `i benchmark`. Never `i datasets`, `gli
  embeddings`, `i layers`.

## Glossary (use these consistently)

| English | Italian |
|---|---|
| library | la libreria (unlike Spanish, `libreria` is correct Italian here) |
| weights | i pesi |
| checkpoint | il checkpoint |
| dataset | il dataset |
| training / to train | l'addestramento / addestrare |
| fine-tuning / to fine-tune | il fine-tuning / fare fine-tuning (or affinare) |
| inference | l'inferenza |
| to predict / prediction | fare una predizione / la predizione |
| object detection | il rilevamento di oggetti (object detection) |
| bounding box | il bounding box |
| instance segmentation | la segmentazione di istanze |
| semantic segmentation | la segmentazione semantica |
| panoptic segmentation | la segmentazione panottica |
| pose estimation | la stima della posa |
| keypoint | il keypoint |
| depth estimation | la stima della profondità |
| embedding | l'embedding |
| backbone | il backbone |
| neck | il neck |
| head (of a model) | la testa (del modello) |
| layer | il layer |
| feature map | la feature map |
| threshold | la soglia |
| confidence | la confidenza (il punteggio di confidenza) |
| accuracy | l'accuratezza (never `precisione` — that is precision) |
| precision / recall | la precisione / il recall |
| batch / batch size | il batch / la dimensione del batch |
| epoch | l'epoca (le epoche) |
| learning rate | il learning rate (il tasso di apprendimento on first mention) |
| weight decay | il weight decay |
| warmup | il warmup |
| loss | la loss (la funzione di perdita on first mention) |
| data augmentation | la data augmentation (l'aumento dei dati) |
| mixed precision | la precisione mista |
| early stopping | l'early stopping |
| quantization | la quantizzazione |
| export / to export | l'esportazione / esportare |
| deployment / to deploy | il deployment / mettere in produzione |
| pretrained | preaddestrato |
| from scratch | da zero |
| label (annotation) | l'etichetta |
| ground truth | il ground truth |
| open-vocabulary detection | il rilevamento a vocabolario aperto |
| edge device | il dispositivo edge |
| frame rate / FPS | gli FPS |
| anchor-free | anchor-free (invariable adjective) |
| mask | la maschera |
| oriented bounding box | il box orientato |
| tiled inference | l'inferenza a tasselli |
| explainability | la spiegabilità |
| benchmark | il benchmark |
| pipeline | la pipeline |
| framework | il framework |
| release | la release (or il rilascio) |
| default | predefinito / di default |
| flag | il flag |
| task | il task |

`mAP`, `mAP50`, `IoU`, `NMS`, `ONNX`, `GPU`, `CPU`, `CUDA` stay untouched.

Terms kept in English on purpose, because Italian developer prose keeps them
and the Italian calque reads as a translation: `checkpoint`, `dataset`,
`bounding box`, `embedding`, `backbone`, `neck`, `layer`, `feature map`,
`learning rate`, `weight decay`, `warmup`, `loss`, `data augmentation`, `early
stopping`, `fine-tuning`, `deployment`, `ground truth`, `benchmark`,
`pipeline`, `framework`, `flag`, `task`, `edge`, `open source`, `machine
learning`, `computer vision`. Terms deliberately translated, because the
Italian is at least as common in real usage: `weights` → `pesi`, `training` →
`addestramento`, `inference` → `inferenza`, `threshold` → `soglia`,
`quantization` → `quantizzazione`, `export` → `esportazione`, `label` →
`etichetta`, `mask` → `maschera`, `epoch` → `epoca`, `accuracy` →
`accuratezza`, `mixed precision` → `precisione mista`.

## Formatting invariants (validated by script)

- Same number of headings, at the same levels, in the same order.
- Same number of fenced code blocks; code identical except comment lines.
- Same set of link targets.
- Same frontmatter keys as the English file — no keys added or dropped.
- Do not add translator notes, disclaimers, or a "translated from" line.
