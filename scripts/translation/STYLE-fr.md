# French translation style guide — LibreYOLO docs

How to produce `<slug>.fr.md` twins of the English docs in `content/docs/`.
The docs loader (`src/lib/docs.js`) picks up the twin automatically; the file
must be a complete, standalone copy of the English file with the rules below.

## Voice and register

- Standard metropolitan French, the register of professional technical
  documentation. Address the reader as **`vous`** everywhere — never `tu`, and
  never `on` used as a disguised second person. Imperatives take the `vous`
  form: « Installez », « Lancez », « Exportez ». Bare-verb English headings
  become infinitives: "Train" → « Entraîner », "Validate" → « Valider ».
- Technical, direct, unadorned — mirror the tone of the English source. Do not
  add, remove, summarize or "improve" content. Every paragraph, list item,
  table row, admonition and FAQ entry in the English file appears in the
  French file, in the same order.
- Spell out what the English spells out; keep the same level of formality.
  Resist the French reflex to inflate: no `il convient de`, no `il est à noter
  que`, no `permet de` padding where the English simply says "does".
- Headings and labels use **sentence case** — capital on the first word and on
  proper nouns only. "Export & deploy" → « Exporter et déployer », not
  « Exporter et Déployer ».
- **Spacing before double punctuation.** French requires a space before `:`
  `;` `?` `!` and inside guillemets. Use a **non-breaking space** (U+00A0)
  before `: ; ? !`, after `«`, before `»`, and between a number and its unit
  or symbol (`35 $`, `100 %`, `640 px`). An ordinary space is a tolerable
  fallback; **no space at all is an error**.
- **Quotation marks.** Use `« … »` for quoted prose and for scare quotes
  ("research license" → « licence de recherche »). Do not use `""` or `“”` in
  body prose. Inside a double-quoted YAML scalar, guillemets need no escaping
  — that is part of why they are preferred here.
- **Apostrophes.** Use the straight apostrophe `'` (U+0027), not `’`, in both
  Markdown and frontmatter: it keeps diffs clean and matches the English
  sources. Consequence — any YAML value containing an apostrophe must be
  **double-quoted** (`lead: "L'inférence..."`), never single-quoted.
- **None of this typography enters code.** Inside fenced code blocks, inline
  `` `code` ``, YAML keys, file names, CLI flags, URLs and link targets there
  is never a space before `:`, never a guillemet, never a non-breaking space.
  `title: RTMDet`, `metrics/mAP50-95`, `https://…`, `data="my-dataset.yaml"`
  and `[prediction](/docs/predict)` stay byte-identical to the English.
- **Numbers stay as written in English.** Keep the dot decimal separator
  (`0.001 mAP`, `lr0=0.004`, `3.10`) so prose agrees with the code beside it.
  Do not switch to a comma, do not insert thousands separators.

## What to translate

- Frontmatter: `title`, `seo_title`, `description`, `lead`, and `faq` (both
  `q` and `a`). Quote YAML strings that contain `:` or other special chars —
  in French that is nearly all of them, because of the space-before-colon rule.
- Reader-visible prose values in `meta` and `verification` frontmatter blocks
  (labels like "Install" → « Installation », prose values like "3.10 or newer"
  → « 3.10 ou plus récent »). Values marked `mono` or that are
  code/identifiers stay.
- Snippet and tab `label` values that are descriptive phrases ("Video and
  streams" → « Vidéo et flux », "Check a dataset" → « Vérifier un dataset »,
  "Use the exported file" → « Utiliser le fichier exporté »). Labels that are
  proper names stay untouched: `Python`, `CLI`, `Bash`, library, tool and
  format names.
- `keywords`: **localize, don't translate literally.** Write the queries a
  French-speaking developer would actually type — a mix of French phrases and
  the English terms they'd realistically use (e.g.
  `"détection d'objets python"`, `"entraîner yolo sur son propre dataset"`,
  `"exporter yolo onnx"`, `"yolo licence mit"`). Keep model names as-is.
- Body prose, headings, table cells, image alt text, admonitions, and
  `hero.caption` (the media `src`/`poster` paths stay).
- Comments **inside** code blocks and snippets (`# téléchargement automatique
  ...`). Keep them short enough not to wrap the line.

## What stays exactly as in English

- All code (statements, identifiers, strings, CLI commands, file names, URLs)
  — only comments change. Never translate string literals passed to code.
- Frontmatter keys themselves, their order, and the structural fields:
  `families`, `last_verified`, `snippets` structure (keys, order, count),
  `language` values, hero/media paths.
- Internal and external link **targets** (translate link text, never hrefs).
- Model names (YOLOv9, RF-DETR, RTMDet...), product names, library names,
  format names (ONNX, TensorRT, CoreML), and heading anchors implied by them.
- Component tags in the body — `<code-tabs name="predict" />`,
  `<export-matrix />`, `<checkpoint-table />`, `<provenance-box></provenance-box>`,
  `<citation-block />` — including their attribute values.

## Glossary (use these consistently)

| English | French |
|---|---|
| library | la bibliothèque (never « librairie » — false friend) |
| framework | le framework |
| weights | les poids |
| weights file | le fichier de poids |
| checkpoint | le checkpoint (keep English; masculine, invariable) |
| dataset | le dataset (keep English; « jeu de données » only where a gloss helps) |
| training / to train | l'entraînement / entraîner |
| fine-tuning / to fine-tune | le fine-tuning / faire du fine-tuning (or affiner) |
| pretrained | pré-entraîné |
| from scratch | à partir de zéro |
| inference | l'inférence |
| inference engine | le moteur d'inférence |
| prediction / to predict | la prédiction / prédire |
| validation / to validate | la validation / valider |
| object detection | la détection d'objets |
| bounding box | la bounding box (or la boîte englobante — pick one per page) |
| oriented bounding box | la boîte orientée |
| mask | le masque |
| instance segmentation | la segmentation d'instances |
| semantic segmentation | la segmentation sémantique |
| panoptic segmentation | la segmentation panoptique |
| pose estimation | l'estimation de pose |
| keypoint | le point clé |
| depth estimation | l'estimation de profondeur |
| surface normals | les normales de surface |
| open-vocabulary detection | la détection à vocabulaire ouvert |
| text recognition / OCR | la reconnaissance de texte / l'OCR |
| matting | le matting |
| denoising | le débruitage |
| embedding | l'embedding (keep English; masculine) |
| clustering | le clustering |
| retrieval | la recherche par similarité |
| backbone | le backbone (keep English) |
| neck | le neck (keep English) |
| head (of a model) | la tête (du modèle) |
| layer | la couche |
| feature map | la carte de caractéristiques |
| features | les caractéristiques |
| anchor / anchor-free | l'ancre / sans ancres |
| threshold | le seuil |
| confidence | la confiance / le score de confiance |
| accuracy | l'exactitude (never « précision » — that word is taken by `precision`) |
| precision (metric) | la précision |
| recall | le rappel |
| batch / batch size | le batch / la taille de batch |
| epoch | l'époque (the `epochs=` argument itself stays as code) |
| learning rate | le learning rate (gloss once as « taux d'apprentissage » if useful) |
| warmup | le warmup |
| optimizer | l'optimiseur |
| weight decay | le weight decay |
| loss | la loss (la fonction de perte on first mention) |
| overfitting | le surapprentissage |
| early stopping | l'early stopping |
| mixed precision | la précision mixte |
| data augmentation | l'augmentation de données |
| label (annotation) | l'étiquette |
| class | la classe |
| ground truth | la vérité terrain |
| benchmark | le benchmark |
| quantization | la quantification (never « quantisation ») |
| export / to export | l'export / exporter |
| deployment / to deploy | le déploiement / déployer |
| runtime | le runtime |
| pipeline | le pipeline |
| edge device | l'appareil edge |
| edge (as in edge-to-cloud) | l'edge |
| embedded system | le système embarqué |
| frame rate / FPS | les FPS / la cadence |
| tiled inference | l'inférence par tuiles |
| streaming | le streaming (or le flux, for a video source) |
| tracking | le suivi |
| release | la version |
| logger | le logger |
| open source | l'open source (invariable, no hyphen) |
| license | la licence (`MIT License` as a proper name stays) |

Terms deliberately **kept in English**: `checkpoint`, `dataset`, `batch`,
`embedding`, `backbone`, `neck`, `learning rate`, `weight decay`, `warmup`,
`early stopping`, `benchmark`, `pipeline`, `runtime`, `framework`,
`clustering`, `matting`, `edge`, `open source`, `fine-tuning`. These are what
French practitioners actually say; a French calque reads as a
back-translation. `bounding box` and `loss` may go either way — choose once
per page and stay with it.

`mAP`, `mAP50`, `IoU`, `NMS`, `ONNX`, `GPU`, `CPU`, `CUDA`, `SaaS` and `API`
stay untouched, and abbreviations take no French plural `s` (`les GPU`, not
`les GPUs`).

## Formatting invariants (validated by script)

- Same number of headings, at the same levels, in the same order.
- Same number of fenced code blocks; code identical except comment lines.
- Same set of link targets.
- Same frontmatter keys as the English file — no keys added or dropped.
- Do not add translator notes, disclaimers, or a "translated from" line.
