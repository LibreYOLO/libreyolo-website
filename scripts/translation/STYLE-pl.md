# Polish translation style guide — LibreYOLO docs

How to produce `<slug>.pl.md` twins of the English docs in `content/docs/`.
The docs loader (`src/lib/docs.js`) picks up the twin automatically; the file
must be a complete, standalone copy of the English file with the rules below.

## Voice and register

- Neutral technical Polish. **Impersonal is the default register**, second
  person informal is the exception, reserved for direct instructions, buttons
  and CTA labels. Justification: Polish second-person forms are gendered in the
  past tense and in predicative adjectives (`zrobiłeś` / `zrobiłaś`,
  `jesteś gotowy` / `jesteś gotowa`), so a docs page written throughout in the
  second person has to guess the reader's gender on every other sentence.
  Impersonal constructions carry no gender and are what Polish developer
  documentation actually reads like.
- The impersonal toolkit: `-no` / `-to` forms (`wagi pobrano`), reflexive
  passives (`wagi są pobierane`, `architektura jest wykrywana automatycznie`),
  modal impersonals (`można`, `należy`, `warto`, `wystarczy`), and bare
  infinitives after them (`aby wytrenować model na własnym zbiorze danych,
  należy...`). Prefer these over a chain of nominalizations.
- Where the English is a plain imperative aimed at the reader
  ("Install the package", "Run the example"), use the informal second-person
  imperative: `Zainstaluj pakiet`, `Uruchom przykład`. Never write the pronoun
  `Ty` and never capitalize `Ci`, `Twój`, `Tobie` mid-sentence — capitalized
  address is letter register and reads as an ad in developer docs.
- Never `Pan` / `Pani` (too formal, wrong audience) and never `my` (`polecamy`,
  `zbudowaliśmy`): the docs do not speak as a team. English "we recommend"
  becomes `zalecane jest`, `lepiej` or a plain imperative.
- Avoid clerical Polish: `dokonać eksportu` (write `wyeksportować`),
  `w celu` (write `aby`), `posiada` (write `ma`), `celem` (write `aby`),
  `przedmiotowy` / `niniejszy` (write `ten`), `poprzez` where plain `przez` or
  an instrumental case will do.
- Technical, direct, unadorned — mirror the tone of the English source. Do not
  add, remove, summarize or "improve" content. Every paragraph, list item,
  table row, admonition and FAQ entry in the English file appears in the
  Polish file, in the same order.
- Spell out what the English spells out; keep the same level of formality.
- Headings are sentence case: capitalize the first word and proper names only
  (`Trenowanie`, `Eksport`, `Walidacja`), never Title Case.
- Section headings that are single English verbs (Install, Predict, Train,
  Validate, Export) become verbal nouns (`Instalacja`, `Predykcja`,
  `Trenowanie`, `Walidacja`, `Eksport`), not imperatives. Buttons and CTA-style
  labels stay imperative (`Zacznij`, `Przeczytaj dokumentację`).

## What to translate

- Frontmatter: `title`, `seo_title`, `description`, `lead`, and `faq` (both
  `q` and `a`). Quote YAML strings that contain `:` or other special chars.
- Reader-visible prose values in `meta` and `verification` frontmatter blocks
  (labels like "Install" → `Instalacja`, prose values like "3.10 or newer" →
  `3.10 lub nowszy`). Values marked `mono` or that are code/identifiers stay.
- Snippet and tab `label` values that are descriptive phrases ("Video and
  streams" → `Wideo i strumienie`, "Check a dataset" → `Sprawdzenie zbioru
  danych`). Descriptive labels take a verbal noun, not an imperative. Labels
  that are proper names stay untouched: `Python`, `CLI`, `Bash`, library, tool
  and format names.
- `keywords`: **localize, don't translate literally.** Write the queries a
  Polish-speaking developer would actually type — a mix of Polish phrases and
  the English terms they'd realistically use, lowercase, often with the model
  name in Latin script (e.g. `"detekcja obiektów python"`,
  `"trenowanie yolo na własnych danych"`, `"eksport yolo do onnx"`,
  `"yolo inferencja wideo"`). Keep model names as-is.
- Body prose, headings, table cells, image alt text, admonitions, and
  `hero.caption` (the media `src`/`poster` paths stay).
- Comments **inside** code blocks and snippets (`# wagi pobierane automatycznie...`).

## What stays exactly as in English

- All code (statements, identifiers, strings, CLI commands, file names, URLs)
  — only comments change. Never translate string literals passed to code.
- Frontmatter keys themselves, their order, and the structural fields:
  `families`, `last_verified`, `snippets` structure (keys, order, count),
  `language` values, hero/media paths.
- Internal and external link **targets** (translate link text, never hrefs).
- Model names (YOLOv9, RF-DETR, RTMDet, RTMDet-Ins, SAM, MiDaS, LibreYOLO),
  product names, library names, format and runtime names (ONNX, TensorRT,
  CoreML, OpenVINO, TFLite, PyTorch, CUDA), and heading anchors implied by them.
- Argument and metric names as written in the source: `conf`, `iou`, `imgsz`,
  `lr0`, `epochs`, `batch`, `metrics/mAP50-95`, `-seg`, `NotImplementedError`.
  In prose they keep Latin script and back-ticks and are never Polonized.
- Metric and algorithm abbreviations: `mAP`, `mAP50`, `mAP50-95`, `IoU`, `NMS`,
  `AP`, `FPS`. Hardware abbreviations: `GPU`, `CPU`, `TPU`, `NPU`, `RAM`,
  `VRAM`. Licenses: `MIT`, `AGPL`, `Apache-2.0`.
- Architecture-part terms with no settled Polish equivalent: `backbone`, `neck`,
  `anchor-free`, `matting`, `early stopping`. Use them bare, in Latin script,
  and let a Polish generic noun carry the grammar (`blok backbone`,
  `detektor anchor-free`).

## Latin script, Polonization and declension

Three buckets. Decide by this list, not by feel — the same term must not appear
Polonized on one page and in bare Latin script on the next.

**Translated into Polish** (a settled Polish term exists): wagi, trenowanie,
dostrajanie, warstwa, głowica, cechy, mapa cech, próg, pewność, precyzja,
dokładność, epoka, maska, klasa, etykieta, adnotacje, współczynnik uczenia,
mieszana precyzja, funkcja straty, optymalizator, eksport, wdrożenie,
środowisko uruchomieniowe, opóźnienie, przepustowość, wizja komputerowa,
uczenie maszynowe, otwarty kod źródłowy, zbiór danych, pamięć podręczna.

**Borrowed and declined as Polish nouns** (the community says it this way; a
"proper" translation would read as a back-translation exercise): checkpoint,
inferencja, embedding, batch, augmentacja, kwantyzacja, pipeline, framework,
benchmark, tensor, streaming, centroid, mikrokontroler. These take normal
Polish endings: `z checkpointu`, `w batchu`, `kilka embeddingów`,
`w pipelinie`. Loanwords ending in a silent `-e` take the apostrophe before the
ending, per Polish orthography: `pipeline'u`, `pipeline'y`.

**Kept in Latin script and never inflected**: everything in the previous
section (model, format, tool and license names, API identifiers, metric and
hardware abbreviations, `backbone`, `neck`).

### Declension of Latin-script names

- **Do not decline the name itself and do not bolt a Polish ending onto it.**
  Write `w YOLOv9`, `dla RTMDet`, `z ONNX`, `eksport do TensorRT`. Never
  `YOLOv9-a`, `RTMDetem`, `ONNX-em`, `ONNX-a`, `YOLOv9'a`, `w RTMDecie`.
- **Use a generic Polish noun as the case carrier** whenever the sentence needs
  a case the bare name cannot show. This is the standard fix and it is always
  available: `model YOLOv9` (`w modelu YOLOv9`, `dla modelu YOLOv9`),
  `w formacie ONNX`, `do formatu ONNX`, `w rodzinie RTMDet`,
  `biblioteka LibreYOLO` (`w bibliotece LibreYOLO`), `plik ONNX`,
  `silnik TensorRT`, `na urządzeniach z CUDA`, `w środowisku PyTorch`.
- The apostrophe declension Polish normally allows for foreign names
  (`Google'a`) is **not** used here. These names appear next to code and file
  names, and an apostrophe glued to an identifier breaks copy-paste and reads
  as a typo. Reach for the generic-noun carrier instead.
- Polish does not form Russian-style hyphen compounds. Write `plik ONNX`,
  `skrypt Pythona`, `polecenie CLI`, `urządzenie brzegowe` — not `ONNX-plik`
  or `edge-urządzenie`.
- Fully assimilated tool names **do** decline, because they are ordinary Polish
  nouns now: `Python` → `w Pythonie`, `Pythona`; `GitHub` → `na GitHubie`;
  `Docker` → `w Dockerze`. This applies only to names written in Polish
  orthography in running prose, never inside back-ticks.
- Agreement: a bare model or family name behaves as masculine inanimate
  (`RTMDet obsługuje`, `RTMDet został przeniesiony z mmdetection`). If the
  sentence needs another gender, lead with the generic noun and let it govern
  (`biblioteka LibreYOLO obsługuje...`, `rodzina RTMDet-Ins nie ma ścieżki
  trenowania`).
- Latin abbreviations take no Polish plural or case marker: `dwa GPU`,
  `kilka GPU`, `na dwóch GPU` — never `GPU-ki`, `GPUs`, `GPU-ów`.
- Version strings, size suffixes and file suffixes stay in code voice and are
  never inflected: `1.5.0`, rozmiary `t`–`x`, sufiks `-seg`.

### Counted nouns and plural forms

Polish agrees a counted noun with the number in three patterns, and getting
this wrong is the single most visible translation error on a page full of
counts:

| Number | Form | Example |
|---|---|---|
| 1 | nominative singular | `1 model`, `1 rodzina`, `1 warstwa` |
| 2–4, and any number ending in 2, 3, 4 **except** 12–14 | nominative plural | `2 modele`, `24 modele`, `102 modele` |
| 0, 5–21, anything ending in 5–9 or 0, and 12–14 | genitive plural | `5 modeli`, `12 modeli`, `30 modeli`, `0 modeli` |
| decimals | genitive singular | `1,5 modelu`, `0,5 rodziny` |

- In docs prose the number is a literal, so simply write the correct form:
  "Five sizes, `t` through `x`" becomes "Pięć rozmiarów, od `t` do `x`", because
  *pięć* takes the genitive plural *rozmiarów*, not *rozmiary*.
- **Never hand-write one form for an interpolated count.** In UI strings under
  `messages/pl.json`, where the number arrives at render time, use ICU
  `plural` with all four CLDR categories and keep the placeholder name inside
  each branch so it survives placeholder validation:

  ```
  {count, plural, one {{count} rodzina} few {{count} rodziny} many {{count} rodzin} other {{count} rodziny}}
  ```

  `few` is the 2–4 pattern, `many` is the 0/5+ pattern, `other` is the decimal
  pattern (genitive singular).
- If a string cannot take ICU plurals for structural reasons, fall back to the
  label form, which sidesteps agreement entirely: `rodziny: {count}`,
  `liczba warstw: {count}`. Do not fall back to a bare guessed form.
- Numerals also govern the verb: the genitive-plural pattern takes a neuter
  singular verb (`5 modeli obsługuje`), the 2–4 pattern takes a plural verb
  (`2 modele obsługują`). Rephrase rather than fight this when the count is
  interpolated.

## Glossary (use these consistently)

| English | Polish |
|---|---|
| library | biblioteka |
| framework | framework (declines: `we frameworku`) |
| repository | repozytorium (`repo` only in informal UI copy) |
| open source | open source (`otwarty kod źródłowy` in running prose) |
| computer vision | wizja komputerowa |
| machine learning | uczenie maszynowe |
| state-of-the-art | najnowocześniejszy (SOTA only in `keywords`) |
| weights | wagi (plural; `plik wag`, never `waga` for the file) |
| checkpoint | checkpoint (declines; not `punkt kontrolny`) |
| dataset | zbiór danych (`dataset` only next to `data=` and CLI args) |
| training / to train | trenowanie / trenować (models are `trenowane`, not `szkolone`) |
| fine-tuning / to fine-tune | dostrajanie / dostroić (fine-tuning in brackets on first mention) |
| pretrained | wstępnie wytrenowany |
| from scratch | od zera |
| inference | inferencja (`uruchomić inferencję`) |
| prediction / to predict | predykcja / przewidywać |
| object detection | detekcja obiektów (`wykrywanie obiektów` acceptable once for SEO) |
| detector | detektor |
| bounding box | ramka ograniczająca on first mention, then ramka (`bounding box` only next to code) |
| instance segmentation | segmentacja instancji |
| semantic segmentation | segmentacja semantyczna |
| panoptic segmentation | segmentacja panoptyczna |
| mask | maska |
| pose estimation | estymacja pozy |
| keypoints | punkty kluczowe |
| depth estimation | estymacja głębi |
| embedding | embedding (declines: `embeddingi`, `embeddingów`) |
| open-vocabulary detection | detekcja z otwartym słownikiem |
| oriented bounding box (OBB) | obrócona ramka (OBB) |
| backbone | backbone (Latin; `kręgosłup` is not used) |
| neck | neck (Latin; no settled Polish term) |
| head (of a model) | głowica (modelu) |
| layer | warstwa |
| feature map | mapa cech |
| features | cechy |
| anchor / anchor-free | kotwica / bez kotwic (anchor-free) |
| loss (function) | funkcja straty (in code and metric names — loss) |
| optimizer | optymalizator |
| learning rate | współczynnik uczenia (in code — `lr0`, `lr`) |
| scheduler / warmup | harmonogram / rozgrzewka |
| epoch | epoka |
| batch / batch size | batch / rozmiar batcha |
| data augmentation | augmentacja danych |
| mixed precision | mieszana precyzja |
| early stopping | early stopping (`wczesne zatrzymanie` in brackets on first mention) |
| threshold | próg |
| confidence | pewność (confidence score — `wskaźnik pewności`) |
| precision | precyzja (the precision metric) |
| recall | recall (left in Latin; `czułość` collides with sensitivity) |
| accuracy | dokładność (quality in general); as a metric name — accuracy, left in Latin so it does not collide with precyzja |
| label (annotation) | etykieta (the whole set — adnotacje) |
| ground truth | dane referencyjne (ground truth on first mention) |
| class | klasa |
| quantization | kwantyzacja |
| export / to export | eksport / eksportować |
| deployment / to deploy | wdrożenie / wdrażać (`deploy` only in informal copy) |
| runtime | środowisko uruchomieniowe (runtime) |
| edge device | urządzenie brzegowe (edge) |
| latency | opóźnienie |
| throughput | przepustowość |
| frame rate / FPS | liczba klatek na sekundę / FPS |
| benchmark | benchmark |
| pipeline | pipeline (declines: `w pipelinie`) |
| release | wydanie |
| tiled inference | inferencja kafelkowa |
| stream / streaming | strumień / streaming |
| cache / to cache | pamięć podręczna / zapisywać w pamięci podręcznej |
| model zoo | katalog modeli |
| license | licencja |
| explainability | wyjaśnialność |
| heatmap | mapa cieplna |

`mAP`, `mAP50`, `mAP50-95`, `IoU`, `NMS`, `ONNX`, `GPU`, `CPU`, `CUDA`, `FPS`
stay untouched.

## Formatting invariants (validated by script)

- Same number of headings, at the same levels, in the same order.
- Same number of fenced code blocks; code identical except comment lines.
- Same set of link targets.
- Same frontmatter keys as the English file — no keys added or dropped.
- Do not add translator notes, disclaimers, or a "translated from" line.
- **No em dashes.** The site bans `—` outright (see `AGENTS.md`); Polish is not
  one of the two exceptions. Do not substitute the Polish półpauza `–` either:
  where the English source uses a dash for a parenthetical or a dropped copula,
  recast with a comma, parentheses or a colon. "LibreYOLO — an MIT-licensed
  library" becomes `LibreYOLO, biblioteka na licencji MIT` or
  `LibreYOLO to biblioteka na licencji MIT`; "no anchors — one prior per cell"
  becomes `bez kotwic (jeden prior na komórkę)`. The en dash stays legal as a
  range separator only: `300–500`, `t`–`x`, `mAP50-95` (the last is a code
  identifier and keeps its hyphen).
- Quotation marks in prose are `„ ”`; nested quotes are `« »`. Never use `" "`
  in Polish prose. Straight quotes stay inside code, YAML values and
  identifiers, where they are syntax. Scare quotes from the English source
  carry over as `„ ”`: "sees" → `„widzi”`.
- Decimal separators stay as in the English source (`0.001 mAP`, `lr0=0.004`,
  `0.5`). Do not switch to a comma: these values mirror code, tables and metric
  output, and a comma breaks copy-paste and diffing. Polish decimal commas are
  correct only in prose numbers that are not code values.
- Ordinary spaces only. Do not insert non-breaking spaces after Polish single
  letter words (`w`, `i`, `z`, `a`, `o`) — they are invisible in review and
  noisy in diffs.
- Lists keep the English punctuation pattern; do not convert the source's full
  stops into semicolons.
