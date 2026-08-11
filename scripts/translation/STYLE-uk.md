# Ukrainian translation style guide: LibreYOLO docs

How to produce `<slug>.uk.md` twins of the English docs in `content/docs/`.
The docs loader (`src/lib/docs.js`) picks up the twin automatically; the file
must be a complete, standalone copy of the English file with the rules below.

This guide is written for Ukrainian on its own terms. Do not build it, or the
pages, by routing English through Russian: Ukrainian ML writing has its own
established vocabulary, and a Russian-shaped Ukrainian is the single most
obvious tell to the readers this locale exists for.

## Voice and register

- Standard literary Ukrainian, 2019 orthography (`проєкт`, `кешування`,
  `авдиторія` where it applies), in the register of professional technical
  documentation. Neutral and matter-of-fact, never chatty, never bureaucratic.
- Address the reader as **`ви`, lowercase**, always. Capitalized `Ви` belongs
  to personal letters and formal correspondence and looks wrong in a docs page.
  Verbs agree in the plural: `встановіть`, `запустіть`, `експортуйте`.
- Prefer impersonal and infinitive constructions wherever they read better than
  a second person. `щоб навчити модель, запустіть...`, `для експорту достатньо
  одного рядка`, `ваги завантажуються автоматично`. Ukrainian technical prose
  leans on these; forcing `ви` into every sentence reads like a bad dub.
- Use the impersonal `-но / -то` forms for completed actions instead of a
  passive chain: `ваги завантажено автоматично`, `модель навчено на COCO`, not
  `ваги були завантажені`.
- **Bare-verb English headings become verbal nouns.** `Install` becomes
  `Встановлення`, `Predict` becomes `Передбачення`, `Train` becomes
  `Навчання`, `Validate` becomes `Валідація`, `Export` becomes `Експорт`,
  `Variants` becomes `Варіанти`, `Checkpoints` becomes `Контрольні точки`,
  `Licensing` becomes `Ліцензування`, `Citation` becomes `Цитування`. Verbal
  nouns are the Ukrainian norm for section titles; imperative headings are not.
- Headings and labels use **sentence case**: capital on the first word and on
  proper nouns only. `Export & deploy` becomes `Експорт і розгортання`, not
  `Експорт і Розгортання`.
- Technical, direct, unadorned. Mirror the tone of the English source. Do not
  add, remove, summarize or "improve" content. Every paragraph, list item,
  table row, admonition and FAQ entry in the English file appears in the
  Ukrainian file, in the same order.
- Spell out what the English spells out; keep the same level of formality. No
  padding: no `слід зазначити, що`, no `є таким, що дозволяє`, no `здійснює
  виконання` where the English simply says "runs".
- **Avoid the common calques.** These are the ones that actually show up in
  translated ML docs:

  | not this | this |
  |---|---|
  | даний (meaning "this") | цей, ця, це |
  | слідуючий / наступний (meaning "the following") | такий, наведений нижче |
  | в залежності від | залежно від |
  | по замовчуванню | типово, за замовчуванням |
  | на протязі | протягом |
  | у якості (meaning "as") | як |
  | представляє собою | є, or rewrite with a real verb |
  | включає в себе | містить, охоплює |
  | виключити (meaning "turn off") | вимкнути |
  | відмінити | скасувати |
  | співпадати | збігатися |
  | вияснити | з'ясувати |
  | приймати участь | брати участь |
  | більш швидший | швидший |
  | у випадку якщо | якщо, у разі |

- `download` is `завантажити`, `upload` is `вивантажити`. Never let both
  collapse into `завантажити` on the same page.
- **Numbers stay exactly as in the English.** Keep the decimal point in every
  metric, hyperparameter and version (`0.001 mAP`, `lr0=0.004`, `mAP50-95`,
  `v1.5.0`). Ukrainian prose normally takes a decimal comma, but these values
  sit next to code that uses a point, and mixing separators on one page is
  worse than the inconsistency with general Ukrainian typography.
- Currency and units follow the source: `$35 boards` becomes `плати за 35
  доларів`, `640` stays `640`.

## What to translate

- Frontmatter: `title`, `seo_title`, `description`, `lead`, and `faq` (both
  `q` and `a`). Quote YAML strings that contain `:` or other special chars.
- Reader-visible prose values in `meta` and `verification` frontmatter blocks
  (labels like "Install" become `Встановлення`, prose values like "3.10 or
  newer" become `3.10 або новіша`). Values marked `mono` or that are
  code/identifiers stay.
- Snippet and tab `label` values that are descriptive phrases ("Video and
  streams" becomes `Відео та потоки`, "Check a dataset" becomes `Перевірити
  датасет`, "Use the exported file" becomes `Використати експортований файл`).
  Labels that are proper names stay untouched: `Python`, `CLI`, `Bash`,
  library, tool and format names.
- `keywords`: **localize, don't translate literally.** Write the queries a
  Ukrainian-speaking developer would actually type, which is a mix of Ukrainian
  phrases and the English terms they really use (e.g.
  `"виявлення об'єктів python"`, `"детекція об'єктів yolo"`,
  `"навчити yolo на власному датасеті"`, `"експорт yolo onnx"`,
  `"yolo ліцензія mit"`). Both `виявлення об'єктів` and `детекція об'єктів`
  are worth having across a page's keyword list, because both are typed.
  Keep model names as-is.
- Body prose, headings, table cells, image alt text, admonitions, and
  `hero.caption` (the media `src`/`poster` paths stay).
- Comments **inside** code blocks and snippets (`# ваги завантажуються...`).

## What stays exactly as in English

- All code (statements, identifiers, strings, CLI commands, file names, URLs).
  Only comments change. Never translate string literals passed to code.
- Frontmatter keys themselves, their order, and the structural fields:
  `families`, `last_verified`, `snippets` structure (keys, order, count),
  `language` values, hero/media paths.
- Internal and external link **targets** (translate link text, never hrefs).
- Model names (YOLOv9, RF-DETR, RTMDet, RTMDet-Ins, YOLOX...), product names,
  library names, format names (ONNX, TensorRT, CoreML), and heading anchors
  implied by them.
- Component tags in the body: `<code-tabs name="predict" />`,
  `<export-matrix />`, `<checkpoint-table />`,
  `<provenance-box></provenance-box>`, `<citation-block />`, including their
  attribute values.
- Metric and API identifiers quoted from the library:
  `metrics/mAP50-95(M)`, `result.masks`, `train()`, `NotImplementedError`,
  `pretrained=False`, `weight_decay`.

## Glossary (use these consistently)

| English | Ukrainian |
|---|---|
| library | бібліотека |
| framework | фреймворк |
| weights | ваги (pl.) |
| weights file | файл ваг |
| checkpoint | контрольна точка |
| model | модель |
| dataset | датасет |
| training / to train | навчання / навчати |
| fine-tuning / to fine-tune | донавчання / донавчати |
| pretrained | попередньо навчений |
| from scratch | з нуля |
| epoch | епоха |
| inference | інференс |
| prediction / to predict | передбачення / передбачати |
| validation / to validate | валідація / валідувати |
| object detection | виявлення об'єктів (детекція об'єктів in keywords) |
| detector | детектор |
| bounding box | обмежувальна рамка (short: рамка) |
| oriented bounding box | орієнтована рамка |
| mask | маска |
| instance segmentation | сегментація екземплярів |
| semantic segmentation | семантична сегментація |
| panoptic segmentation | паноптична сегментація |
| pose estimation | оцінювання пози |
| keypoints | ключові точки |
| depth estimation | оцінювання глибини |
| embedding | ембединг (pl. ембединги) |
| backbone | бекбон |
| head (of a model) | голова (моделі) |
| layer | шар |
| feature / features | ознака / ознаки |
| feature map | карта ознак |
| activation | активація |
| threshold | поріг |
| confidence | впевненість (оцінка впевненості) |
| accuracy | правильність (accuracy) |
| precision (metric) | точність |
| recall | повнота |
| confusion matrix | матриця плутанини |
| batch / batch size | батч / розмір батча |
| learning rate | швидкість навчання (the arg stays `lr0`) |
| loss | функція втрат (втрати) |
| overfitting | перенавчання |
| early stopping | рання зупинка |
| data augmentation | аугментація даних |
| mixed precision | змішана точність |
| quantization | квантування |
| export / to export | експорт / експортувати |
| deployment / to deploy | розгортання / розгортати |
| label (annotation) | мітка |
| annotation | анотація |
| ground truth | еталонна розмітка (еталонні дані) |
| class | клас |
| open-vocabulary detection | виявлення з відкритим словником |
| edge device | edge-пристрій (at the edge: на периферії) |
| frame rate / FPS | частота кадрів / FPS |
| latency | затримка |
| resolution | роздільна здатність |
| pipeline | пайплайн |
| runtime | середовище виконання |
| benchmark | бенчмарк |
| open source | відкритий код |
| machine learning | машинне навчання |
| computer vision | комп'ютерний зір |
| explainability / interpretability | пояснюваність / інтерпретовність |
| heatmap | теплова карта |
| flag (CLI) | прапорець |
| model zoo | каталог моделей |

`mAP`, `mAP50`, `mAP50-95`, `IoU`, `NMS`, `ONNX`, `GPU`, `CPU`, `CUDA`, `API`,
`CLI`, `SaaS`, `MIT`, `AGPL` stay untouched.

### The three buckets

Every term on a page falls into one of three buckets. When a term is not in
the glossary, decide which bucket it belongs to before inventing anything.

1. **Translated to Ukrainian.** Anything with a settled Ukrainian equivalent:
   `ваги`, `шар`, `поріг`, `карта ознак`, `функція втрат`, `навчання`,
   `виявлення об'єктів`, `обмежувальна рамка`, `сегментація`, `маска`,
   `епоха`, `квантування`, `розгортання`, `мітка`, `точність`, `повнота`,
   `аугментація даних`, `машинне навчання`, `комп'ютерний зір`,
   `відкритий код`. This is the default bucket. Reach for it first.
2. **Transliterated to Cyrillic.** Borrowings that Ukrainian practitioners
   actually say out loud and that have no clean native equivalent:
   `датасет`, `батч`, `ембединг`, `бекбон`, `інференс`, `пайплайн`,
   `бенчмарк`, `фреймворк`, `тензор`, `токен`, `реліз`, `тайл`, `кеш`,
   `скрипт`. Transliterated terms decline normally (`з датасету`,
   `у батчі`, `розмір батча`).
3. **Kept in Latin.** Names and identifiers, which are never transliterated:
   model and family names (`LibreYOLO`, `YOLOv9`, `RF-DETR`, `RTMDet`,
   `YOLOX`, `Ultralytics`), formats (`ONNX`, `TensorRT`, `CoreML`,
   `OpenVINO`, `TFLite`), tools and libraries (`Python`, `PyTorch`, `NumPy`,
   `OpenCV`, `PIL`, `Hugging Face`, `CUDA`), metric and hardware
   abbreviations (`mAP`, `IoU`, `NMS`, `FPS`, `GPU`, `CPU`), everything
   inside code, and the two architecture terms with no usable Ukrainian
   form yet: `neck` and `matting`. Never write `Ультралітікс`, `Пайторч`
   or `ОННХ`.

Do not stack a gloss on every borrowing. Gloss a term at most once per page,
on first mention, and only when the Ukrainian form is the less familiar one:
`донавчання (fine-tuning)`, then `донавчання` from there on.

## Model names, declension and case

Latin-script names do not decline. The case is carried by a generic Ukrainian
noun placed in front of the name, and the name itself stays in its exact
upstream spelling.

- Wrong: `у RTMDet'і`, `RTMDet-ом`, `ваги YOLOv9-а`, `з ONNX'у`,
  `налаштування LibreYOLO'а`.
- Right: `модель RTMDet`, `у моделі RTMDet`, `за допомогою моделі RTMDet`,
  `ваги моделі YOLOv9`, `у форматі ONNX`, `налаштування бібліотеки LibreYOLO`.

Useful carrier nouns: `модель`, `детектор`, `сімейство`, `архітектура`,
`формат`, `бібліотека`, `пакет`, `контрольна точка`, `файл`, `клас`, `метод`.

- A bare name is fine where no ending is needed at all, that is, as a
  nominative subject or after a preposition that takes the nominative-looking
  form in practice: `RTMDet підтримує виявлення та сегментацію екземплярів`,
  `LibreYOLO працює під ліцензією MIT`. As soon as the sentence would demand
  an ending, add the carrier noun instead.
- Agreement follows the carrier noun, not the name:
  `модель RTMDet навчена`, `сімейство RTMDet підтримує`. With a bare name,
  agree with the implied noun `модель` (feminine): `RTMDet підтримує` is
  neutral and safe, so prefer verbs that need no gender marking.
- Suffixed variants keep their upstream form whole: `RTMDet-Ins`,
  `LibreRTMDets-seg.pt`, `mAP50-95(M)`. Never split them across a Ukrainian
  ending.
- Abbreviations behave the same way: `на GPU`, `у форматі ONNX`,
  `поріг NMS`, `значення mAP`. Never `на GPU-шці`, never `по IoU`.
- No possessive `'s` calques: `API бібліотеки LibreYOLO`, not
  `LibreYOLO's API` and not `LibreYOLO-ів API`.
- File suffixes and extensions stay in code style and never take endings:
  `файл .onnx`, `файл .pt`, not `.onnx'у`.

## Quotation marks, apostrophes and dashes

- **Quotation marks.** Use `« … »` for quoted prose and for scare quotes
  (`"research license"` becomes `«дослідницька ліцензія»`). For a quote inside
  a quote, use the lower/upper pair `„ … “`. Do not use `""` or `“ ”` in body
  prose. Inside a double-quoted YAML scalar, guillemets need no escaping,
  which is part of why they are preferred here.
- **Apostrophes.** Ukrainian needs the apostrophe as a letter-level sign
  (`об'єкт`, `комп'ютер`, `п'ять`, `з'ясувати`). Use the straight apostrophe
  `'` (U+0027), not `’`: it keeps diffs clean and matches the English sources.
  Consequence: any YAML value containing an apostrophe must be
  **double-quoted** (`lead: "Виявлення об'єктів..."`), never single-quoted.
- **No em dashes.** This site bans `—` in every locale it applies to, and
  Ukrainian is one of them. That is a real cost here, because the dash is
  grammar in Ukrainian and not only punctuation, so the fix is to **rewrite
  the sentence, not to delete the character**. Leaving `RTMDet це
  одностадійний детектор` behind is worse than the dash was.

  For the copula (`X — це Y`), in order of preference:

  1. **Give the sentence a real verb.** `RTMDet виявляє об'єкти за один
     прохід`, `LibreYOLO надає єдиний API для трьох архітектур`.
  2. **Use `є` with the instrumental.** `RTMDet є одностадійним детектором`.
     Correct and slightly formal; fine in a `lead` or a definition, but do not
     use it three times in a row.
  3. **Use a colon** when a definition or an enumeration follows.
     `RTMDet: одностадійний детектор без якорів`. Good for headings, leads
     and table cells.
  4. **Apposition with commas.** `RTMDet, одностадійний детектор без якорів,
     працює з тим самим об'єктом Results`.

  For a parenthetical aside where the English used a dash, use commas or
  parentheses. For an intro to a list, use a colon. For ranges, use the plain
  hyphen the English already uses (`mAP50-95`, `t` through `x`).

  Do not substitute an en dash `–` as a workaround, and do not carry an em
  dash across from the English source: if the English paragraph has one, the
  Ukrainian paragraph must be restructured around it.

  The ban stops at the code boundary. Inside fenced code blocks, inline
  `` `code` ``, YAML keys, file names, CLI flags, URLs and link targets,
  nothing changes, and verbatim English quotations keep their own punctuation.
- **None of this typography enters code.** Guillemets, apostrophes and
  rewrites apply to prose only.

## Formatting invariants (validated by script)

- Same number of headings, at the same levels, in the same order.
- Same number of fenced code blocks; code identical except comment lines.
- Same set of link targets.
- Same frontmatter keys as the English file, with no keys added or dropped.
- Do not add translator notes, disclaimers, or a "translated from" line.
- Run `node scripts/translation/validate.mjs uk` before committing, then
  `node scripts/translation/sync-check.mjs --stamp` once the twin is final.
