# Indonesian translation style guide — LibreYOLO docs

How to produce `<slug>.id.md` twins of the English docs in `content/docs/`.
The docs loader (`src/lib/docs.js`) picks up the twin automatically; the file
must be a complete, standalone copy of the English file with the rules below.

## Voice and register

- Standard Bahasa Indonesia, *baku* but not stiff: the register of Indonesian
  developer documentation and good technical writing, not of a government
  circular. Use `bahasa Indonesia baku` spelling and grammar (KBBI / EYD V),
  but keep sentences short and plain.
- **Address the reader impersonally by default, and use `Anda` only when a
  pronoun is genuinely required.** Justification: Indonesian imperatives take
  no subject, so most instructional sentences in these docs need no pronoun at
  all (`Jalankan perintah berikut`, `Pasang paketnya lebih dulu`,
  `Ekspor sekali, lalu jalankan di mana saja`). That is the natural register of
  Indonesian dev docs and it avoids the politeness problem entirely. When a
  pronoun cannot be avoided (possession, contrast, direct second person), use
  `Anda`, capitalized: `dataset Anda`, `hasil kerja Anda tetap milik Anda`.
  Never `kamu` (too informal for reference docs), never `engkau`/`kau`, never
  `saudara`, never `kalian`. Do not mix: a page is impersonal plus `Anda`, and
  never switches to `kamu` halfway through.
- For the writer's own side, prefer impersonal too. Use `kami` only where the
  English says "we" about the project; never `kita` (inclusive "we"), which
  wrongly folds the reader into the maintainers.
- Technical, direct, unadorned. Mirror the tone of the English source. Do not
  add, remove, summarize or "improve" content. Every paragraph, list item,
  table row, admonition and FAQ entry in the English file appears in the
  Indonesian file, in the same order.
- Avoid the inflated formal register: write `untuk` not `dalam rangka`, `pakai`
  or `gunakan` not `mempergunakan`, `setelah` not `setelah dilakukannya`,
  `dengan` not `dengan menggunakan` when `dengan` alone is enough. Prefer verbs
  to nominalizations (`saat mengekspor` beats `pada saat dilakukannya proses
  ekspor`).
- Keep active voice where the English is active. Indonesian `di-` passives are
  correct and common in technical prose (`bobot diunduh otomatis`), so use them
  where they read better, but do not turn every English active sentence passive.
- Spell out what the English spells out; keep the same level of formality.

### Plurals: Indonesian does not inflect for number

Indonesian nouns have no plural form. A bare noun is number-neutral, and the
English `-s` must be dropped, including on English loanwords kept in the text.

- Correct: `dataset`, `checkpoint`, `bounding box`, `embedding`, `layer`,
  `backbone`, `benchmark`, `flag`, `task`, `model`, `pipeline`.
- **Never** `datasets`, `checkpoints`, `embeddings`, `layers`, `models-model`.
- Add a quantifier only when the count actually matters and context does not
  already supply it: `beberapa dataset`, `semua checkpoint`, `dua family`,
  `setiap layer`, `banyak model`. Note that the quantified noun still stays
  singular in form.
- Reduplication (`model-model`, `berkas-berkas`) marks diversity or plurality
  emphatically and reads heavy in reference prose. Use it rarely, never on an
  English loanword (`dataset-dataset` is wrong here), and never together with a
  quantifier (`beberapa model-model` is ungrammatical).
- The English plural in a heading or table header usually becomes a bare noun:
  "Checkpoints" is `Checkpoint`, "Variants" is `Varian`, "Export formats" is
  `Format ekspor`.

### Affixation on English loanwords

Indonesian verb morphology (`meN-`, `di-`, `-kan`, `-i`, `peN-an`) attaches
only to bases written in Indonesian orthography. The rule for this project:

1. If the term has a settled, KBBI-assimilated Indonesian spelling, use it and
   affix it normally: `ekspor` → `mengekspor`, `diekspor`, `pengeksporan`;
   `impor` → `mengimpor`; `instal` → `menginstal`; `validasi` → `memvalidasi`;
   `kuantisasi` → `mengkuantisasi`. Write **`mengekspor`**, one word, no hyphen.
2. If the term is kept in English (see the glossary), **do not glue an
   Indonesian prefix onto it.** `meng-export`, `meng-train`, `di-deploy`,
   `di-export`, `nge-run` are chat-register forms and are not *baku*; they never
   appear in these docs.
3. Instead, either fall back to rule 1's assimilated word, or use a light verb
   plus the English noun: `melakukan fine-tuning`, `menjalankan inference`,
   `melakukan deployment`, `menerapkan model ke produksi`, `menyimpan
   checkpoint`. For passives, the same: `dilatih` (not `di-train`),
   `diekspor` (not `di-export`), `dijalankan` (not `di-run`).
4. Do not invent an assimilation the language has not made. There is no
   `mendataset`, no `mengembedding`, no `mengbackbone`.

So: **`mengekspor`, never `meng-export`.** `melatih model`, never
`men-train model`.

## What to translate

- Frontmatter: `title`, `seo_title`, `description`, `lead`, and `faq` (both
  `q` and `a`). Quote YAML strings that contain `:` or other special chars.
- Reader-visible prose values in `meta` and `verification` frontmatter blocks
  (labels like "Install" → "Instalasi", prose values like "3.10 or newer" →
  "3.10 atau lebih baru"). Values marked `mono` or that are code/identifiers
  stay.
- Snippet and tab `label` values that are descriptive phrases ("Video and
  streams" → "Video dan stream", "Check a dataset" → "Memeriksa dataset").
  Labels that are proper names stay untouched: `Python`, `CLI`, `Bash`,
  library, tool and format names.
- `keywords`: **localize, don't translate literally.** Write the queries an
  Indonesian-speaking developer would actually type, which in practice is a mix
  of Indonesian phrases and English technical terms (for example
  `"deteksi objek python"`, `"cara training yolo dataset sendiri"`,
  `"export yolo ke onnx"`, `"object detection python"`). Indonesian developers
  search in English for task and model names far more often than in Indonesian:
  keep those English inside `keywords` even where the body prose translates
  them. Model names stay as-is. Search queries are also the one place where the
  colloquial `training` / `export` noun forms are appropriate.
- Body prose, headings, table cells, image alt text, admonitions, and
  `hero.caption` (the media `src`/`poster` paths stay).
- Comments **inside** code blocks and snippets (`# unduhan otomatis...`).

## What stays exactly as in English

- All code (statements, identifiers, strings, CLI commands, file names, URLs).
  Only comments change. Never translate string literals passed to code.
- Frontmatter keys themselves, their order, and the structural fields:
  `families`, `last_verified`, `snippets` structure (keys, order, count),
  `language` values, hero/media paths.
- Internal and external link **targets** (translate link text, never hrefs).
- Model names (YOLOv9, RF-DETR, RTMDet...), product names, library names,
  format names (ONNX, TensorRT, CoreML), and heading anchors implied by them.
- Numbers inside metrics, code and CLI arguments keep the English decimal
  point: `0.001 mAP`, `lr0=0.004`, `conf=0.25`. Indonesian normally writes
  decimals with a comma and thousands with a period, but that convention is
  **not** applied to metrics, versions, hyperparameters or anything that
  mirrors a code value, because it would make the docs disagree with the
  output the reader sees. Prices and plain counts in running prose may follow
  Indonesian convention.
- Loanwords this project keeps in English (see the glossary). They are written
  without italics, without quotes, and with no plural `-s`: `dataset`,
  `checkpoint`, `bounding box`, `embedding`, `backbone`, `head`, `feature map`,
  `learning rate`, `batch`, `loss`, `flag`, `task`, `pipeline`, `framework`,
  `benchmark`, `ground truth`, `early stopping`, `mixed precision`,
  `fine-tuning`, `open source`, `machine learning`, `computer vision`.

## Glossary (use these consistently)

| English | Indonesian |
|---|---|
| library | library (`pustaka` only if a native word is unavoidable) |
| framework | framework |
| weights | bobot |
| weights file | berkas bobot |
| checkpoint | checkpoint (never `titik simpan`) |
| dataset | dataset (never `datasets`, never `kumpulan data` in headings) |
| training / to train | pelatihan / melatih (`training` acceptable in keywords) |
| fine-tuning / to fine-tune | fine-tuning / melakukan fine-tuning |
| from scratch | dari nol |
| inference | inferensi |
| to predict / prediction | memprediksi / prediksi |
| object detection | deteksi objek |
| bounding box | bounding box (`kotak pembatas` on first mention if helpful) |
| oriented bounding box | bounding box berorientasi |
| instance segmentation | segmentasi instance (**never** `instansi`, which means a government agency) |
| semantic segmentation | segmentasi semantik |
| panoptic segmentation | segmentasi panoptik |
| mask | mask (avoid `masker`, which is a face mask) |
| pose estimation | estimasi pose |
| keypoint | keypoint (`titik kunci` on first mention if helpful) |
| depth estimation | estimasi kedalaman |
| embedding | embedding |
| backbone | backbone |
| neck | neck |
| head (of a model) | head (avoid `kepala`) |
| layer | lapisan (`layer` is also fine and common; pick one per page) |
| feature map | feature map |
| threshold | ambang batas (`ambang` after first mention) |
| confidence | skor keyakinan (confidence) |
| accuracy | akurasi |
| precision / recall | presisi / recall |
| batch / batch size | batch / ukuran batch |
| epoch | epoch |
| learning rate | learning rate (`laju pembelajaran` on first mention) |
| weight decay | weight decay |
| warmup | warmup |
| loss | loss (avoid `fungsi kerugian`, which reads as accounting) |
| data augmentation | augmentasi data |
| mixed precision | mixed precision |
| early stopping | early stopping |
| quantization | kuantisasi |
| export / to export | ekspor / mengekspor |
| deployment / to deploy | deployment (penerapan) / menerapkan, menjalankan di produksi |
| pretrained | pretrained (`model yang sudah dilatih` when a phrase reads better) |
| label (annotation) | label |
| annotation | anotasi |
| ground truth | ground truth |
| class | kelas |
| image | gambar (`citra` only in an explicitly academic passage) |
| pixel | piksel |
| open-vocabulary detection | deteksi open-vocabulary |
| edge device | perangkat edge |
| frame rate / FPS | FPS |
| anchor-free | anchor-free (invariable) |
| tiled inference | inferensi berbasis tile |
| explainability / interpretability | explainability / interpretability |
| benchmark | benchmark |
| pipeline | pipeline |
| release | rilis |
| default | default (`bawaan` as an adjective: `nilai bawaan`) |
| flag | flag |
| task | task (avoid `tugas`, which reads as homework) |
| open source | open source (`sumber terbuka` only in legal-flavored prose) |
| proprietary / closed source | proprietary / closed source |
| machine learning / computer vision | machine learning / computer vision |
| to download / to install | mengunduh / memasang, menginstal |

`mAP`, `mAP50`, `IoU`, `NMS`, `ONNX`, `GPU`, `CPU`, `CUDA`, `API`, `SaaS` stay
untouched.

Terms kept in English on purpose, because Indonesian developer prose keeps them
and the Indonesian calque reads like a translation exercise: `dataset`,
`checkpoint`, `bounding box`, `embedding`, `backbone`, `neck`, `head`,
`feature map`, `learning rate`, `weight decay`, `warmup`, `loss`, `batch`,
`epoch`, `early stopping`, `mixed precision`, `fine-tuning`, `ground truth`,
`benchmark`, `pipeline`, `framework`, `flag`, `task`, `edge`, `recall`,
`open source`, `machine learning`, `computer vision`. Terms deliberately
translated, because the Indonesian form is at least as common in real usage:
`weights` → `bobot`, `training` → `pelatihan`, `inference` → `inferensi`,
`threshold` → `ambang batas`, `quantization` → `kuantisasi`, `export` →
`ekspor`, `label` → `label`, `annotation` → `anotasi`, `accuracy` → `akurasi`,
`precision` → `presisi`, `class` → `kelas`, `image` → `gambar`, `pixel` →
`piksel`, `layer` → `lapisan`.

## Formatting invariants (validated by script)

- Same number of headings, at the same levels, in the same order.
- Same number of fenced code blocks; code identical except comment lines.
- Same set of link targets.
- Same frontmatter keys as the English file, with no keys added or dropped.
- **No em dashes.** This site bans the em dash `—` in every locale except
  Chinese and Russian, where it is grammar rather than style. Indonesian is not
  an exception: never introduce `—` into an `.id.md` file, not even where the
  English source uses one. Rewrite instead, with a comma, a colon, parentheses,
  or a full stop. The en dash `–` is likewise not used as punctuation.
- Indonesian quotation marks are the straight double quotes `"..."` used by the
  English source. Do not switch to guillemets or to curly low quotes.
- Do not add translator notes, disclaimers, or a "translated from" line.
