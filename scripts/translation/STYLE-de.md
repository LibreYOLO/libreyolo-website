# German translation style guide for LibreYOLO docs

How to produce `<slug>.de.md` twins of the English docs in `content/docs/`.
The docs loader (`src/lib/docs.js`) picks up the twin automatically; the file
must be a complete, standalone copy of the English file with the rules below.

## Voice and register

- Standard German (Bundesdeutsch), the register of a modern developer
  handbook. Address the reader as **`du`** everywhere, never `Sie` and never
  the impersonal `man` used as a disguised second person. `du` is what German
  developer documentation actually uses today (Python-Doku, MDN, Docker,
  Hugging Face), and switching mid-page reads as sloppy. Imperatives take the
  `du` form: „Installiere“, „Starte“, „Exportiere“.
- Write `du`, `dein`, `dich` in lowercase. Capitalized `Du`/`Dein` is letter
  style, not documentation style.
- Technical, direct, unadorned, mirroring the tone of the English source. Do
  not add, remove, summarize or „improve“ content. Every paragraph, list item,
  table row, admonition and FAQ entry in the English file appears in the German
  file, in the same order.
- Spell out what the English spells out; keep the same level of formality.
  Resist the German reflex to inflate: no `es sei darauf hingewiesen, dass`, no
  `im Rahmen von`, no `ermöglicht es dir, X zu tun` where the English simply
  says „does X“.
- **Headings use the Nominalstil.** A bare English verb heading becomes a noun,
  not an imperative: „Install“ becomes „Installation“, „Predict“ becomes
  „Vorhersage“, „Train“ becomes „Training“, „Validate“ becomes „Validierung“,
  „Export“ stays „Export“, „Licensing“ becomes „Lizenzierung“, „Citation“
  becomes „Zitieren“. Section headings in the body prose of the site (UI
  strings) may keep the verb form where the English is clearly a button.
- **NO EM DASHES.** The site bans the em dash „—“ outright (see `AGENTS.md`);
  German is not one of the two exempt languages. Where the English uses an em
  dash, use a comma, a colon, parentheses or two sentences. Do not substitute
  the spaced en dash „ – “ (the German Gedankenstrich) either: it is the same
  stylistic move under a different codepoint. The plain hyphen `-` is
  unaffected and stays essential, because German compounds need it constantly
  (`Open-Source-Projekt`, `Bounding-Box-Format`).
- **Noun capitalization applies to loanwords too.** Every noun is capitalized,
  including borrowed ones: `das Embedding`, `der Checkpoint`, `die Bounding
  Box`, `das Backbone`, `das Deployment`, `die Feature Map`, `das Early
  Stopping`, `der Batch`. Adjectives and verbs derived from them stay lowercase
  (`vortrainiert`, `exportieren`, `quantisiert`). The only exception is a token
  that is code: `conf`, `iou`, `train()` and `pretrained` keep their source
  spelling inside backticks.
- **Compounds with English loanwords: durchkoppeln.** A standalone multi-word
  English term keeps its space and its capitals (`die Bounding Box`, `die
  Feature Map`, `das Early Stopping`, `die Learning Rate`). As soon as it
  enters a compound, with a German element or another English one, the whole
  compound is written as one unit with hyphens between every part:
  `das Bounding-Box-Format`, `die Feature-Map-Extraktion`,
  `die Class-Activation-Map-Methode`, `das Edge-Device`, `die Batch-Größe`,
  `der Open-Vocabulary-Detektor`, `die ONNX-Export-Pipeline`,
  `das LibreYOLO-Modell`. Never leave a mixed compound open
  (`Bounding Box Format` is wrong) and never fuse it without hyphens
  (`Boundingboxformat` is wrong). A single-word loanword may fuse directly:
  `die Trainingspipeline`, `die Modelldatei`, `die Batchgröße` reads worse than
  `die Batch-Größe`, so prefer the hyphen whenever the seam is hard to read.
- **Genitive and dative of borrowed nouns.** Masculine and neuter borrowings
  take `-s` in the genitive singular: `des Checkpoints`, `des Backbones`,
  `des Datasets`, `des Embeddings`, `des Batches`. Plurals are the English `-s`
  plural: `die Checkpoints`, `die Embeddings`, `die Backbones`, `die Feature
  Maps`; `Batch` takes `die Batches`. Dative plural adds no extra `-n` to a
  plural that already ends in `-s`: `mit den Embeddings`, not
  `mit den Embeddingsn`. Feminine borrowings are invariable in the singular:
  `der Bounding Box`, `die Pipeline`, `der Pipeline`.
- **Prefer a German verb over a conjugated English one.** `deployen` and
  `getunt` are ugly in a participle; write `ausrollen`, `in Betrieb nehmen`,
  `Fine-Tuning durchführen` instead. `exportieren`, `trainieren`, `validieren`
  and `quantisieren` are already German and are used freely.
- **Quotation marks.** Use `„…“` (U+201E, U+201C) for quoted prose and for
  scare quotes („Forschungslizenz“). Do not use `""` or `“”` in body prose.
  Inside a double-quoted YAML scalar the German quotes need no escaping, which
  is part of why they are preferred here.
- **Use `ß`**, not `ss`, per German and Austrian orthography: `größer`,
  `heißt`, `Größe`, `schließlich`. Swiss `ss` spelling is out of scope.
- **None of this typography enters code.** Inside fenced code blocks, inline
  `` `code` ``, YAML keys, file names, CLI flags, URLs and link targets there
  is never a German quote, never an added hyphen, never a capitalized
  identifier. `title: RTMDet`, `metrics/mAP50-95`, `https://…`,
  `data="my-dataset.yaml"` and `[prediction](/docs/predict)` stay
  byte-identical to the English.
- **Numbers stay as written in English.** Keep the dot decimal separator
  (`0.001 mAP`, `lr0=0.004`, `3.10`) so prose agrees with the code beside it.
  Do not switch to a comma, do not insert thousands separators. Between a
  number and a unit or symbol use a non-breaking space (U+00A0): `100 %`,
  `35 $`, `640 px`, `300 Epochen`.

## What to translate

- Frontmatter: `title`, `seo_title`, `description`, `lead`, and `faq` (both
  `q` and `a`). Quote YAML strings that contain `:` or other special chars.
- Reader-visible prose values in `meta` and `verification` frontmatter blocks
  (labels like "Install" becoming „Installation“, prose values like "3.10 or
  newer" becoming „3.10 oder neuer“). Values marked `mono` or that are
  code/identifiers stay.
- Snippet and tab `label` values that are descriptive phrases ("Video and
  streams" becoming „Video und Streams“, "Check a dataset" becoming „Datensatz
  prüfen“, "Use the exported file" becoming „Die exportierte Datei nutzen“).
  Labels that are proper names stay untouched: `Python`, `CLI`, `Bash`,
  library, tool and format names.
- `keywords`: **localize, don't translate literally.** Write the queries a
  German-speaking developer would actually type, a mix of German phrases and
  the English terms they would realistically use (e.g.
  `"objekterkennung python"`, `"yolo mit eigenem datensatz trainieren"`,
  `"yolo nach onnx exportieren"`, `"yolo mit lizenz kommerziell nutzen"`).
  Keep model names as-is, and keep the queries lowercase, the way people type.
- Body prose, headings, table cells, image alt text, admonitions, and
  `hero.caption` (the media `src`/`poster` paths stay).
- Comments **inside** code blocks and snippets (`# wird beim ersten Aufruf
  automatisch geladen`). Keep them short enough not to wrap the line, and
  remember German runs roughly 15 to 25 % longer than English: shorten the
  wording rather than let a comment overflow.

## What stays exactly as in English

- All code (statements, identifiers, strings, CLI commands, file names, URLs).
  Only comments change. Never translate string literals passed to code.
- Frontmatter keys themselves, their order, and the structural fields:
  `families`, `last_verified`, `snippets` structure (keys, order, count),
  `language` values, hero/media paths.
- Internal and external link **targets** (translate link text, never hrefs).
- Model names (YOLOv9, RF-DETR, RTMDet...), product names, library names,
  format names (ONNX, TensorRT, CoreML), and heading anchors implied by them.
- Component tags in the body, `<code-tabs name="predict" />`,
  `<export-matrix />`, `<checkpoint-table />`,
  `<provenance-box></provenance-box>` and `<citation-block />`, including
  their attribute values.

## Glossary (use these consistently)

| English | German |
|---|---|
| library | die Bibliothek (never „die Library“) |
| framework | das Framework |
| weights | die Gewichte |
| weights file | die Gewichtsdatei |
| checkpoint | der Checkpoint (keep English; masc., gen. `des Checkpoints`) |
| dataset | der Datensatz (`das Dataset` when naming a file or a product, e.g. `das Dataset my-dataset.yaml`) |
| training / to train | das Training / trainieren |
| fine-tuning / to fine-tune | das Fine-Tuning / Fine-Tuning durchführen (or nachtrainieren) |
| pretrained | vortrainiert |
| from scratch | von Grund auf neu |
| inference | die Inferenz |
| inference engine | die Inferenz-Engine |
| prediction / to predict | die Vorhersage / vorhersagen |
| validation / to validate | die Validierung / validieren |
| object detection | die Objekterkennung |
| detector | der Detektor |
| bounding box | die Bounding Box (compound: `das Bounding-Box-Format`) |
| oriented bounding box | die orientierte Box (in prose also „gedrehte Box“) |
| mask | die Maske |
| instance segmentation | die Instanzsegmentierung |
| semantic segmentation | die semantische Segmentierung |
| panoptic segmentation | die panoptische Segmentierung |
| pose estimation | die Pose-Schätzung |
| keypoint | der Keypoint (der Schlüsselpunkt on first mention if a gloss helps) |
| depth estimation | die Tiefenschätzung |
| surface normals | die Oberflächennormalen |
| open-vocabulary detection | die Open-Vocabulary-Erkennung |
| text recognition / OCR | die Texterkennung / OCR |
| matting | das Matting |
| denoising | das Entrauschen (die Rauschunterdrückung) |
| embedding | das Embedding (keep English; neut., pl. `die Embeddings`) |
| clustering | das Clustering |
| retrieval | das Retrieval (die Ähnlichkeitssuche as a gloss) |
| backbone | das Backbone (Duden also allows `der`; pick one per page) |
| neck | das Neck |
| head (of a model) | der Head (des Modells); „der Kopf“ only where the metaphor is being explained |
| layer | die Schicht (the `layer` argument itself stays as code) |
| feature map | die Feature Map (compound: `die Feature-Map-Extraktion`) |
| features | die Features (die Merkmale) |
| anchor / anchor-free | der Anchor / ankerfrei |
| threshold | der Schwellenwert |
| confidence | die Confidence (der Konfidenzwert for the score) |
| accuracy | die Accuracy (never „Präzision“, that word is taken by `precision`) |
| precision (metric) | die Precision (die Präzision) |
| recall | der Recall (die Trefferquote) |
| batch / batch size | der Batch / die Batch-Größe |
| epoch | die Epoche |
| learning rate | die Lernrate (`die Learning Rate` is equally idiomatic; pick one per page) |
| warmup | das Warmup (die Aufwärmphase) |
| optimizer | der Optimizer |
| weight decay | das Weight Decay |
| loss | der Loss (die Verlustfunktion on first mention) |
| overfitting | das Overfitting (die Überanpassung) |
| early stopping | das Early Stopping |
| mixed precision | Mixed Precision (gemischte Präzision) |
| data augmentation | die Datenaugmentierung (Data Augmentation) |
| label (annotation) | das Label (pl. `die Labels`); die Annotation for the file |
| class | die Klasse |
| ground truth | die Ground Truth (die Referenzdaten) |
| benchmark | der Benchmark |
| quantization | die Quantisierung (never „Quantifizierung“) |
| export / to export | der Export / exportieren |
| deployment / to deploy | das Deployment / ausrollen (in Betrieb nehmen) |
| runtime | die Runtime (die Laufzeitumgebung) |
| pipeline | die Pipeline |
| edge device | das Edge-Device (das Edge-Gerät) |
| edge (as in edge-to-cloud) | das Edge |
| embedded system | das eingebettete System |
| frame rate / FPS | die Bildrate / FPS |
| tiled inference | die gekachelte Inferenz |
| streaming | das Streaming (der Stream for a single source) |
| tracking | das Tracking |
| release | das Release (die Version) |
| logger | der Logger |
| task | die Aufgabe (der Task where it names the library's `task=` concept) |
| open source | Open Source (adjective: quelloffen; compound: `das Open-Source-Projekt`) |
| license | die Lizenz (`MIT License` as a proper name stays; `die MIT-Lizenz` in prose) |

That is 71 pairs. Terms deliberately **kept in English**: `Checkpoint`,
`Embedding`, `Backbone`, `Neck`, `Head`, `Batch`, `Feature Map`, `Bounding
Box`, `Early Stopping`, `Weight Decay`, `Warmup`, `Optimizer`, `Benchmark`,
`Pipeline`, `Runtime`, `Framework`, `Clustering`, `Matting`, `Retrieval`,
`Edge`, `Ground Truth`, `Overfitting`, `Deployment`, `Fine-Tuning`,
`Open Source`, `Accuracy`, `Precision`, `Recall`, `Confidence`. German
practitioners say these in English; a full German calque reads as a
back-translation and is often ambiguous (`Genauigkeit` collapses accuracy and
precision into one word, which the metrics tables cannot afford). `Datensatz`,
`Lernrate`, `Schwellenwert`, `Schicht` and `Gewichte` are the reverse case:
the German word is what people actually say, so use it.

`mAP`, `mAP50`, `IoU`, `NMS`, `ONNX`, `GPU`, `CPU`, `CUDA`, `SaaS` and `API`
stay untouched. Abbreviations take the English plural where German has settled
on it (`die GPUs`, `die APIs`), and inflect with a hyphen in compounds
(`die GPU-Auslastung`, `der ONNX-Graph`).

## Formatting invariants (validated by script)

- Same number of headings, at the same levels, in the same order.
- Same number of fenced code blocks; code identical except comment lines.
- Same set of link targets.
- Same frontmatter keys as the English file, no keys added or dropped.
- Do not add translator notes, disclaimers, or a „translated from“ line.
