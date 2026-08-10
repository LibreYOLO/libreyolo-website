# Russian translation style guide — LibreYOLO docs

How to produce `<slug>.ru.md` twins of the English docs in `content/docs/`.
The docs loader (`src/lib/docs.js`) picks up the twin automatically; the file
must be a complete, standalone copy of the English file with the rules below.

## Voice and register

- Neutral technical Russian. Address the reader as `вы`, always lowercase —
  never `Вы`, not in headings, buttons, FAQ answers or admonitions. Capitalized
  `Вы` is letter-writing register and reads as an ad in developer docs.
- Prefer impersonal and infinitive constructions wherever they read better than
  a second person: «Веса скачиваются с Hugging Face при первом запуске и
  кэшируются локально», «Чтобы обучить модель на своём датасете, ...». English
  passives usually come out as Russian reflexive verbs (`скачиваются`,
  `кэшируются`, `определяется`), not as a chain of nominalizations.
- Never `мы`: the docs do not speak as a team. English "we recommend" becomes
  «лучше», «рекомендуется» or a plain imperative.
- Match the tone of good Russian developer documentation, not academic prose.
  Banned clerical register: `данный` (write `этот`), `является` (drop the copula
  or use a dash), `осуществляется` / `производится` (write the plain verb),
  `в качестве` (write `как`), `позволяет осуществить` (write `позволяет` + verb
  or just the verb), `с целью` (write `чтобы`).
- Technical, direct, unadorned — mirror the tone of the English source. Do not
  add, remove, summarize or "improve" content. Every paragraph, list item,
  table row, admonition and FAQ entry in the English file appears in the
  Russian file, in the same order.
- Spell out what the English spells out; keep the same level of formality.
- Headings are sentence case: capitalize the first word and proper names only
  («Обучение», «Экспорт», «Проверка на валидации»), never Title Case.
- Section headings that are single English verbs (Install, Predict, Train,
  Validate, Export) become verbal nouns in Russian («Установка»,
  «Предсказание», «Обучение», «Валидация», «Экспорт»), not imperatives.
  Buttons and CTA-style labels stay imperative («Начать», «Читать документацию»).

## What to translate

- Frontmatter: `title`, `seo_title`, `description`, `lead`, and `faq` (both
  `q` and `a`). Quote YAML strings that contain `:` or other special chars —
  Russian sentences hit this often («Установка: что нужно»).
- Reader-visible prose values in `meta` and `verification` frontmatter blocks
  (labels like "Install" → «Установка», prose values like "3.10 or newer" →
  «3.10 или новее»). Values marked `mono` or that are code/identifiers stay.
- Snippet and tab `label` values that are descriptive phrases ("Video and
  streams" → «Видео и потоки», "Check a dataset" → «Проверка датасета»).
  Descriptive labels take a verbal noun, not an imperative. Labels that are
  proper names stay untouched: `Python`, `CLI`, `Bash`, library, tool and
  format names.
- `keywords`: **localize, don't translate literally.** Write the queries a
  Russian-speaking developer would actually type — a mix of Russian phrases
  and the English terms they'd realistically use, usually lowercase and often
  in Latin script (e.g. `"детекция объектов python"`,
  `"обучить yolo на своём датасете"`, `"экспорт yolo в onnx"`,
  `"yolo инференс на видео"`). Keep model names as-is.
- Body prose, headings, table cells, image alt text, admonitions, and
  `hero.caption` (the media `src`/`poster` paths stay).
- Comments **inside** code blocks and snippets (`# веса скачиваются...`).

## What stays exactly as in English

- All code (statements, identifiers, strings, CLI commands, file names, URLs)
  — only comments change. Never translate string literals passed to code.
- Frontmatter keys themselves, their order, and the structural fields:
  `families`, `last_verified`, `snippets` structure (keys, order, count),
  `language` values, hero/media paths.
- Internal and external link **targets** (translate link text, never hrefs).
- Model names (YOLOv9, RF-DETR, RTMDet...), product names, library names,
  format names (ONNX, TensorRT, CoreML), and heading anchors implied by them.
- Argument and metric names as written in the source: `conf`, `iou`, `imgsz`,
  `lr0`, `metrics/mAP50-95`, `-seg`, `NotImplementedError`. In prose they keep
  Latin script and back-ticks and are never transliterated.

## Latin script, transliteration and declension

Three buckets. Decide by this list, not by feel — the same term must not appear
transliterated on one page and in Latin script on the next.

**Translated into Russian** (a settled Russian word exists): веса, обучение,
дообучение, слой, голова, признаки, карта признаков, порог, уверенность,
точность, полнота, эпоха, маска, класс, метка, разметка, скорость обучения,
смешанная точность, ранняя остановка, функция потерь, оптимизатор, экспорт,
развёртывание, среда выполнения, задержка, пропускная способность,
компьютерное зрение, машинное обучение, открытый исходный код.

**Transliterated into Cyrillic** (the community says it this way; a "proper"
translation would read as a back-translation exercise): инференс, чекпойнт,
датасет, эмбеддинг, бэкбон, батч, аугментация, квантизация, пайплайн, фреймворк,
бенчмарк, релиз, репозиторий, тензор, кэш, стриминг, маттинг, центроид,
микроконтроллер. These decline normally: «из чекпойнта», «в датасете»,
«по батчам».

**Kept in Latin script**: model and family names (YOLOv9, YOLOX, RF-DETR,
RTMDet, RTMDet-Ins, SAM, MiDaS, LibreYOLO), format and runtime names (ONNX,
TensorRT, CoreML, OpenVINO, TFLite, PyTorch, CUDA), tool and site names (Python,
NumPy, OpenCV, PIL, Hugging Face, GitHub), licenses (MIT, AGPL, Apache-2.0),
metric and algorithm abbreviations (mAP, mAP50, mAP50-95, IoU, NMS, AP, FPS),
hardware abbreviations (GPU, CPU, TPU, NPU, RAM, VRAM), API identifiers, and
`neck` (no settled Russian term; `шея` is not used in practice).

Declension of Latin-script names:

- Do not decline the name and do not bolt a Cyrillic ending onto it. Write
  «в YOLOv9», «для RTMDet», «из ONNX», «экспорт в TensorRT» — never
  «YOLOv9-е», «RTMDetом», «ONNX-ом», «ONNXа».
- When the sentence needs a case the bare name cannot carry, put a Russian
  generic noun in front and decline that: «в модели YOLOv9», «в формате ONNX»,
  «в семействе RTMDet», «на устройствах с CUDA».
- Hyphenated attributive compounds are correct and preferred over paraphrase:
  `ONNX-модель`, `TensorRT-движок`, `edge-устройство`, `Python-скрипт`,
  `CLI-команда`. The hyphen joins two nouns; it never carries a case ending.
- Agreement: a bare model or family name behaves as masculine inanimate
  («RTMDet поддерживает», «RTMDet был портирован из mmdetection»). If the
  sentence needs another gender, lead with the generic noun and let it govern
  («библиотека LibreYOLO поддерживает…», «модель RTMDet-Ins обучать нельзя»).
- Latin abbreviations take no Cyrillic plural marker: «две GPU», «несколько
  GPU» — never «GPUs», «GPU-шки».
- Version strings, size suffixes and file suffixes stay in code voice and are
  never inflected: `1.5.0`, размеры `t`–`x`, суффикс `-seg`.

## Punctuation and typography

- Quotation marks in prose are `« »`; nested quotes are `„ “`. Never use `" "`
  in Russian prose. Straight quotes stay inside code, YAML values and
  identifiers, where they are syntax.
- Scare quotes from the English source carry over as `« »`: "see" → «видит».
- Em dash `—` with spaces around it for parentheticals and for the dropped
  copula («LibreYOLO — библиотека под MIT»). Do not substitute a hyphen.
- Write `ё` consistently where the standard spelling has it: `развёртывание`,
  `повёрнутые рамки`, `чётко`, `своём`, `ещё`.
- Decimal separators stay as in the English source (`0.001 mAP`, `lr0=0.004`,
  `0.5`). Do not switch to a comma: these values mirror code, tables and
  metric output, and a comma breaks copy-paste and diffing.
- Ranges use an en dash without spaces: `300–500`, `t`–`x`.
- Ordinary spaces only. Do not insert non-breaking spaces or typographic
  thin spaces — they are invisible in review and noisy in diffs.
- Lists keep the English punctuation pattern; do not convert the source's
  full stops into semicolons.

## Glossary (use these consistently)

| English | Russian |
|---|---|
| library | библиотека (never «либа» in docs) |
| framework | фреймворк |
| repository | репозиторий |
| open source | открытый исходный код |
| computer vision | компьютерное зрение |
| machine learning | машинное обучение |
| state-of-the-art | передовой (SOTA only in `keywords`) |
| weights | веса (plural; «файл весов», never «вес») |
| checkpoint | чекпойнт (not «контрольная точка») |
| dataset | датасет (not «набор данных») |
| training / to train | обучение / обучать (models are «обучают», not «тренируют») |
| fine-tuning / to fine-tune | дообучение / дообучать (fine-tuning in brackets on first mention if the source stresses the term) |
| pretrained | предобученный |
| from scratch | с нуля |
| inference | инференс («запустить инференс», never «инферить») |
| prediction / to predict | предсказание / предсказывать |
| object detection | детекция объектов (not «обнаружение объектов») |
| detector | детектор |
| bounding box | ограничивающая рамка on first mention, then рамка («бокс» only next to code) |
| instance segmentation | сегментация экземпляров |
| semantic segmentation | семантическая сегментация |
| panoptic segmentation | паноптическая сегментация |
| mask | маска |
| pose estimation | оценка позы |
| keypoints | ключевые точки |
| depth estimation | оценка глубины |
| embedding | эмбеддинг |
| open-vocabulary detection | детекция с открытым словарём |
| oriented bounding box (OBB) | повёрнутая рамка (OBB) |
| backbone | бэкбон |
| neck | neck (Latin; no settled Russian term) |
| head (of a model) | голова (модели) |
| layer | слой |
| feature map | карта признаков |
| features | признаки |
| anchor / anchor-free | якорь / без якорей (anchor-free) |
| loss (function) | функция потерь (in code and metric names — loss) |
| optimizer | оптимизатор |
| learning rate | скорость обучения (in code — `lr0`, `lr`) |
| scheduler / warmup | планировщик / прогрев |
| epoch | эпоха |
| batch / batch size | батч / размер батча |
| data augmentation | аугментация данных |
| mixed precision | смешанная точность |
| early stopping | ранняя остановка |
| threshold | порог |
| confidence | уверенность (confidence score — «оценка уверенности») |
| precision | точность (the precision metric) |
| recall | полнота |
| accuracy | точность (quality in general); as a metric name — accuracy, left in Latin so it does not collide with precision |
| label (annotation) | метка (the whole set — «разметка») |
| ground truth | эталонная разметка (ground truth on first mention) |
| class | класс |
| quantization | квантизация |
| export / to export | экспорт / экспортировать |
| deployment / to deploy | развёртывание / разворачивать («деплой» only in informal copy) |
| runtime | среда выполнения (runtime) |
| edge device | периферийное устройство / edge-устройство |
| latency | задержка |
| throughput | пропускная способность |
| frame rate / FPS | частота кадров / FPS |
| benchmark | бенчмарк |
| pipeline | пайплайн |
| release | релиз |
| tiled inference | потайловый инференс |
| stream / streaming | поток / стриминг |
| cache / to cache | кэш / кэшировать |
| model zoo | каталог моделей |
| license | лицензия |

`mAP`, `mAP50`, `mAP50-95`, `IoU`, `NMS`, `ONNX`, `GPU`, `CPU`, `CUDA`, `FPS`
stay untouched.

## Formatting invariants (validated by script)

- Same number of headings, at the same levels, in the same order.
- Same number of fenced code blocks; code identical except comment lines.
- Same set of link targets.
- Same frontmatter keys as the English file — no keys added or dropped.
- Do not add translator notes, disclaimers, or a "translated from" line.
