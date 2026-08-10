# Arabic translation style guide — LibreYOLO docs

How to produce `<slug>.ar.md` twins of the English docs in `content/docs/`.
The docs loader (`src/lib/docs.js`) picks up the twin automatically; the file
must be a complete, standalone copy of the English file with the rules below.

Arabic is the only right-to-left locale the site serves: `localeDir` in
`src/i18n/routing.js` puts `dir="rtl"` on `<html>`, and every paragraph you
write becomes an RTL paragraph with Latin identifiers embedded in it. That is
where Arabic twins break, not in the vocabulary. Read **Bidirectional text**
below before writing a line.

## Voice and register

- Modern Standard Arabic (فصحى معاصرة) in its technical register: the way a
  working engineer writes for other engineers. Not literary Arabic — no سجع, no
  rhetorical build-up, no `إن` / `لقد` openings that add nothing, no synonym
  pairs where the English has one word.
- Neutral across regions: it must read the same in Cairo, Riyadh, Casablanca
  and Amman. No dialect (مش، عايز، بدك), no country-specific vocabulary
  (حاسوب, not كمبيوتر only in some markets; ملف, not دوسيه).
- Address the reader in the masculine singular second person — the default of
  Arabic developer docs (`ثبِّت`, `شغِّل`, `لاحظ أن`). Never write paired forms
  (`اكتب/اكتبي`), never the dual or plural. Use impersonal constructions
  (`يمكن`, `يكفي أن`, `يُستحسن`) where they read better than an imperative.
- Prefer short verb-first sentences for instructions and the internal passive
  for descriptions (`تُنزَّل الأوزان`, `يُصدَّر النموذج`) over `تم` + مصدر
  periphrasis (`تم تنزيل الأوزان`), which is longer and reads like a press
  release.
- Break long إضافة chains. Three or more genitive links in a row
  (`عملية تحويل صيغة ملف نموذج الكشف`) is the main way Arabic technical prose
  becomes unreadable; use `من` / `لـ` / a relative clause instead.
- Technical, direct, unadorned — mirror the tone of the English source. Do not
  add, remove, summarize or "improve" content. Every paragraph, list item,
  table row, admonition and FAQ entry in the English file appears in the
  Arabic file, in the same order.
- Keep hedges as hedges: "has not been checked" → `لم يُتحقَّق منه`, not
  `غير مدعوم`. Do not add exclamation marks, blessings or enthusiasm the
  English does not have.
- Tashkeel only to disambiguate a genuinely ambiguous word (`مُدخَل` vs
  `مَدخَل`, `مُدرَّب` vs `مدرِّب`). Never vocalize whole sentences.
- Spell out what the English spells out; keep the same level of formality.

## What to translate

- Frontmatter: `title`, `seo_title`, `description`, `lead`, and `faq` (both
  `q` and `a`). Quote YAML strings that contain `:` or other special chars.
  Prefer the block-list form for `keywords` over the flow form `[a, b, c]`, so
  an Arabic comma `،` or a stray `,` inside a phrase can never split an entry.
- Reader-visible prose values in `meta` and `verification` frontmatter blocks
  (labels like "Install" → `التثبيت`, "Writes" → `المخرجات`, "Loads back" →
  `إعادة التحميل`, "Shapes" → `الأشكال`; prose values like "3.10 or newer" →
  `3.10 أو أحدث`). Values marked `mono` or that are code/identifiers stay.
- Snippet and tab `label` values that are descriptive phrases ("Video and
  streams" → `الفيديو والبثوث`, "Check a dataset" → `فحص مجموعة بيانات`,
  "Instance segmentation" → `تجزئة النُسَخ`, "Use the exported file" →
  `استخدام الملف المُصدَّر`). Labels that are proper names stay untouched:
  `Python`, `CLI`, `Bash`, library, tool and format names.
- `keywords`: **localize, don't translate literally.** Write the queries an
  Arabic-speaking developer would actually type, which is usually an Arabic
  phrase next to the English term they really use, and often pure English
  (e.g. `كشف الأجسام python`, `تدريب yolo على بياناتي`, `تصدير yolo onnx`,
  `rtmdet بدون mmdetection`). Keep model names as-is, and never transliterate
  one into Arabic script — `RTMDet`, never `آر تي إم ديت`.
- Body prose, headings, table cells, image alt text, admonitions, and
  `hero.caption` (the media `src`/`poster` paths stay).
- Comments **inside** code blocks and snippets (`# تُنزَّل الأوزان تلقائيًا عند
  أول استخدام`). Keep them short, and see the bidi rules: no sentence-final
  `.` after a Latin identifier, and no Arabic-Indic digits.

## What stays exactly as in English

- All code (statements, identifiers, strings, CLI commands, file names, URLs)
  — only comments change. Never translate string literals passed to code.
- **No Arabic-script character ever enters code.** Not an Arabic letter, not an
  Arabic comma `،`, semicolon `؛`, question mark `؟`, percent sign `٪`, decimal
  separator `٫`, thousands separator `٬`, not an Arabic-Indic digit
  (`٠١٢٣٤٥٦٧٨٩` or `۰۱۲۳۴۵۶۷۸۹`), and not a bidi control character — inside a
  code block, an inline `code span`, a path, a flag, a version string, a URL or
  a `mono` frontmatter value. Any of these breaks the validator's byte
  comparison and, worse, breaks copy-paste for the reader.
- **Western digits (0-9) everywhere, prose included.** Arabic-Indic digits are
  normal in the Mashriq, but Arabic technical writing and the whole Maghreb use
  Western digits, and the docs are full of `640`, `mAP50-95`, `1.5.0` and
  `0.001` that must match the code beside them. One numeral system per page.
- Frontmatter keys themselves, their order, and the structural fields:
  `families`, `last_verified`, `snippets` structure (keys, order, count),
  `language` values, hero/media paths.
- Internal and external link **targets** (translate link text, never hrefs).
- Model names (YOLOv9, RF-DETR, RTMDet...), product names, library names,
  format names (ONNX, TensorRT, CoreML), and heading anchors implied by them.
- Custom components (`<code-tabs />`, `<export-matrix />`,
  `<checkpoint-table />`, `<provenance-box>`, `<citation-block />`) and their
  attributes — including `name="predict"`, which is a key, not prose.

## Bidirectional text

The single highest-risk part of this locale. Everything here is about the
Unicode Bidirectional Algorithm (UAX #9) deciding where a Latin run, a number
or a punctuation mark lands inside an RTL paragraph.

**Why it breaks.** Punctuation and symbols — space `.` `,` `:` `;` `-` `/`
`(` `)` `[` `]` `"` `'` — are *neutral*: they take their direction from the
strong characters on either side. A sentence ending
`… يُحمَّل الملف LibreRTMDets.pt.` can render with the final period at the far
left of the line, detached from the sentence it ends. Digits are *weak*: `640`
glued to a Latin word joins the Latin run, the same `640` next to Arabic does
not. Nothing in the source file changes; only the display order does.

**Rule 1 — backtick every Latin run in prose.** Any identifier, model file,
flag, path, format name, metric key, argument or CLI fragment that appears in
running prose goes in an inline code span: `` `LibreRTMDets-seg.pt` ``,
`` `conf` ``, `` `metrics/mAP50-95(B)` ``, `` `format="onnx"` ``. The site
stylesheet already pins `pre`, `code`, `kbd`, `samp` to `direction: ltr` and
left alignment under `[dir='rtl']` (`src/app/globals.css`), so a code span is
the supported, reviewable way to isolate a Latin run. It also keeps the
validator's link and comment regexes on familiar ground. (If an *inline* span
still scrambles in the browser, the fix is `unicode-bidi: isolate` in the
stylesheet — `direction` alone has no effect on inline boxes. Fix it in CSS,
never by inserting invisible characters into content.)

**Rule 2 — punctuation belongs to the Arabic sentence.** Put the sentence-final
mark *after* the closing backtick, never inside it, and never move it to where
it "looks right" in your editor.
Good: `` أعِد تحميل النموذج عبر `LibreYOLO()`. ``
Bad: `` أعِد تحميل النموذج عبر `LibreYOLO().` ``
The logical order you type is correct even when the rendered order surprises
you; the renderer, not the author, decides the visual position of the period.

**Rule 3 — put a space between `و` and a Latin word.** Arabic normally attaches
the conjunction with no space, but `وYOLOv9` puts a strong RTL letter flush
against a strong LTR run with no separator, which is exactly the boundary that
scrambles and is unreadable besides. Write `YOLOX و YOLOv9 و RF-DETR`. The same
goes for the definite article and prepositions: rewrite `الـ backbones` and
`بـ ONNX` as `شبكات backbone` and `إلى ONNX`.

**Rule 4 — when a Unicode isolate is genuinely needed.** Only in running Arabic
prose, only for a Latin or numeric run that cannot be backticked (a bare
product name inside a heading, a version number wedged between two Arabic
clauses) and only after trying to restructure the sentence first — moving the
Latin run away from a punctuation boundary fixes most cases with no invisible
characters at all. When you do need one, wrap the run in
U+2066 LEFT-TO-RIGHT ISOLATE … U+2069 POP DIRECTIONAL ISOLATE, or place a
single U+200F RIGHT-TO-LEFT MARK after it. Never use the deprecated embedding
controls U+202A–U+202E: they do not isolate and they leak into the rest of the
paragraph. Record in the PR description that you added one, since nobody can
see it in a diff.

**Rule 5 — the hard rule.** Code fences, inline code spans, link targets, file
paths, flags, version strings, `mono` frontmatter values and every frontmatter
identifier are **never** reordered, never re-punctuated, never given
Arabic-Indic digits and never given a bidi control character. Not one. A single
U+200F inside a fence is an instant validator FAIL, is invisible in review, and
silently corrupts the command the reader copies.

**Rule 6 — link syntax is written in logical order.** `[الوثائق](/docs/predict)`
is correct even when your editor draws the `](` pair backwards. That is a
display artifact of an editor without bidi support, not a bug in the file. Do
not "fix" it by moving brackets — you will produce a file that renders wrong
everywhere else. Verify on the rendered page, and use
`git diff --word-diff` when a diff looks scrambled.

**Rule 7 — never mirror the code.** Prose flows RTL; code blocks stay LTR and
left-aligned. Do not reverse CLI argument order, do not reorder table columns,
do not flip a numbered list, do not swap `xyxy` to `yxyx` because it "reads
backwards". Benchmark and metric tables stay left-to-right (the site provides
`data-ltr` / `.table-ltr` for this).

**Rule 8 — neutrals between two Latin runs are safe; between Latin and Arabic
they are not.** `ONNX / TensorRT` is fine. `(B)` and `(M)` are fine inside a
code span. A lone `(` sitting between an Arabic word and a Latin one is the
classic scrambled-parenthesis bug: backtick the Latin run or move the
parenthesis so both its sides are the same script.

**Rule 9 — never paste text back from a terminal, a PDF viewer or a
bidi-unaware editor.** Some of them store what they displayed rather than what
was typed, which permanently reverses the run order in the file. Type Arabic in
an editor you trust, and check the built page rather than the source view.

Save as UTF-8 without a BOM. A BOM at the head of a `.ar.md` file breaks
frontmatter parsing before any of the above matters.

## Glossary (use these consistently)

The third column is the honest one: whether a working Arabic-speaking ML
engineer actually says the Arabic term or reaches for the English one. Where
English dominates, keep the English term in Latin script, unitalicized, and
gloss it in Arabic **once** on first mention — `checkpoint` (نقطة الحفظ) — never
twice on the same page.

| English | Arabic | What engineers actually use |
|---|---|---|
| library | المكتبة | Arabic. Universal, no competition. |
| framework | إطار العمل | Arabic in writing; `framework` in speech. Use Arabic. |
| weights | الأوزان | Arabic. Settled. |
| weights file | ملف الأوزان | Arabic. |
| checkpoint | نقطة الحفظ | **English dominates.** Gloss once, then `checkpoint`. |
| dataset | مجموعة البيانات | Arabic in writing, `dataset` in speech. Use Arabic. |
| training / to train | التدريب / يُدرِّب | Arabic. Settled. |
| fine-tuning / to fine-tune | الضبط الدقيق | **English dominates** in speech. Gloss once, then Arabic. |
| inference | الاستدلال | Arabic in writing; `inference` in speech. Use Arabic. |
| prediction / to predict | التنبؤ | Arabic. Settled. |
| validation / to validate | التحقق | Arabic. Settled. |
| object detection | كشف الأجسام | Arabic. Settled (`اكتشاف الكائنات` also occurs; pick ours). |
| bounding box | المربع المحيط | Arabic understood, English common. Gloss once. |
| oriented bounding box | المربع المحيط المائل | Arabic; `OBB` stays Latin. |
| segmentation | التجزئة | Arabic. Settled (`التقسيم` also; pick ours). |
| instance segmentation | تجزئة النُسَخ | **English dominates.** Gloss once. |
| semantic segmentation | التجزئة الدلالية | Arabic. Settled. |
| panoptic segmentation | التجزئة الشاملة | No settled Arabic; gloss once as (panoptic). |
| mask | القناع / الأقنعة | Arabic. Settled. |
| pose estimation | تقدير الوضعية | Arabic. Settled. |
| keypoints | النقاط المفتاحية | Arabic. Settled. |
| depth estimation | تقدير العمق | Arabic. Settled. |
| classification | التصنيف | Arabic. Settled. |
| embedding | متجه تمثيلي | **English dominates hard.** Keep `embedding`; gloss once. |
| backbone | — | **English only.** `العمود الفقري` reads as anatomy. |
| head (of a model) | — | **English only.** Use `head`; `رأس النموذج` only if a noun phrase is unavoidable. |
| neck | — | **English only.** |
| layer | الطبقة | Arabic. Settled. |
| loss / loss function | الخسارة / دالة الخسارة | Arabic. Settled. |
| threshold | العتبة | Arabic. Settled. |
| confidence (score) | الثقة / درجة الثقة | Arabic. Settled. |
| accuracy | الدقة | Arabic — but see the precision ruling below. |
| precision / recall (metrics) | — | **Keep Latin.** `الدقة` is already accuracy and `الاستدعاء` is not read instantly; metric keys are `metrics/` identifiers anyway. |
| batch / batch size | الدُّفعة / حجم الدُّفعة | Arabic in prose; `batch=16` untouched in code. |
| epoch | دورة | **English dominates**; `حقبة` is a dictionary word nobody says. Write `300 دورة`, keep `epochs=300`. |
| learning rate | معدل التعلّم | Arabic. Settled. |
| warmup | الإحماء | Split; Arabic is fine, `warmup` common. |
| optimizer | المُحسِّن | Arabic; optimizer names (`AdamW`, `SGD`) stay Latin. |
| data augmentation | زيادة البيانات | **English dominates** in speech. Gloss once, then Arabic. |
| quantization / to quantize | التكميم | Arabic in writing; `INT8`, `FP16` stay Latin. |
| half / mixed precision | نصف الدقة / الدقة المختلطة | Arabic. |
| export / to export | التصدير / يُصدِّر | Arabic. Settled. |
| deployment / to deploy | النشر / يَنشُر | Arabic. Settled — never `التوزيع`. |
| pretrained | مُدرَّب مسبقًا | Arabic. Settled. |
| from scratch | من الصفر | Arabic. |
| label (class label) | التسمية | Arabic; `label` stays Latin when naming label files/keys. |
| annotation | التوصيف | Arabic in writing; `annotation` common in speech. |
| ground truth | القيم المرجعية | **English dominates.** `الحقيقة الأرضية` is a calque that reads as geology. Keep `ground truth`, gloss once. |
| feature / feature map | السمة / خريطة السمات | Arabic in writing; `feature map` often kept. Use Arabic. |
| open-vocabulary detection | الكشف بمفردات مفتوحة | No settled Arabic; gloss once as (open-vocabulary). |
| anchor-free | — | **English only.** Gloss once as (بلا مراسٍ) if the sentence needs it. |
| edge device / edge | جهاز طرفي / الحوسبة الطرفية | Arabic is settled for `edge computing`; `edge` alone stays common. |
| runtime | بيئة التشغيل | Arabic. Settled. |
| latency | زمن الاستجابة | Arabic. Settled. |
| throughput | معدل المعالجة | Split; Arabic acceptable. |
| real-time | الزمن الحقيقي | Arabic. Settled. |
| frame / FPS | الإطار / `FPS` | Arabic for frame, `FPS` always Latin. |
| pipeline | خط المعالجة | **English dominates.** Keep `pipeline` unless the sentence needs a noun. |
| preprocessing / postprocessing | المعالجة المسبقة / المعالجة اللاحقة | Arabic. Settled. |
| tensor | موتِّر | **English dominates** in ML; `موتِّر` is the mathematician's word. Keep `tensor`. |
| tiled / sliced inference | الاستدلال بالتقسيم | No settled Arabic; gloss once. |
| distillation / to distill | التقطير | Arabic in ML writing. |
| early stopping | الإيقاف المبكر | Arabic. Settled. |
| resume (training) | استئناف التدريب | Arabic. Settled. |
| multi-GPU | التدريب متعدد البطاقات | Arabic; `GPU` stays Latin. |
| model family | عائلة النماذج | Arabic. |
| model zoo | معرض النماذج | Arabic; `حديقة النماذج` is a literal calque — do not use. |
| variant / size | الحجم | Arabic; the size letters `t`…`x` stay Latin. |
| benchmark | اختبار الأداء | Arabic in prose; the nav label stays as the product uses it. |
| upstream | المشروع الأصلي | Split; `upstream` common. Gloss once. |
| repository / repo | المستودع | Arabic. Settled. |
| issue / pull request | — | **English only.** GitHub UI nouns are never translated. |
| license | الرخصة | Arabic; `MIT`, `AGPL-3.0`, `Apache-2.0` stay Latin and hyphenated. |
| open source | مفتوح المصدر | Arabic. Settled. |
| commercial use | الاستخدام التجاري | Arabic. Settled. |
| flag / argument | مُعامل | Arabic in prose; the flag itself stays in a code span. |
| dependency | الاعتمادية / الاعتماديات | Arabic. Settled. |

`mAP`, `mAP50`, `mAP50-95`, `AP`, `IoU`, `NMS`, `ONNX`, `TensorRT`, `CoreML`,
`OpenVINO`, `TFLite`, `GPU`, `CPU`, `NPU`, `CUDA`, `FP16`, `INT8`, `px` and
`FPS` stay untouched, always in Latin script and Western digits.

## Formatting invariants (validated by script)

Run `node scripts/translation/validate.mjs ar [section/slug ...]` before
committing. It compares the twin against its English source and fails on drift.

- Same number of headings, at the same levels, in the same order.
- Same number of fenced code blocks; code identical except comment lines. An
  Arabic-Indic digit, an Arabic comma `،` or an invisible bidi control
  character inside a fence is the most common way an Arabic twin fails this.
- Same set of link targets.
- Same frontmatter keys as the English file — no keys added or dropped;
  `families`, `last_verified` and `layout` byte-identical; `hero.src` and
  `hero.poster` byte-identical.
- Same snippet structure: same group keys, same order, same count, same
  `language` values. Only `label` and in-code comments change.
- Do not add translator notes, disclaimers, or a "translated from" line.
