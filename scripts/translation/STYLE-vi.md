# Vietnamese translation style guide — LibreYOLO docs

How to produce `<slug>.vi.md` twins of the English docs in `content/docs/`.
The docs loader (`src/lib/docs.js`) picks up the twin automatically; the file
must be a complete, standalone copy of the English file with the rules below.

## Voice and register

- Neutral technical Vietnamese, the register of a working engineer writing for
  other engineers. Address the reader as `bạn`, or drop the pronoun entirely
  and use an impersonal construction where that reads better (`Cần cài thêm...`,
  `Trọng số được tải về...`). Never `anh/chị`, `quý khách`, `các bạn` or
  `chúng ta` for the reader; they turn developer docs into a sales page or a
  classroom lecture.
- Refer to the library as `LibreYOLO` or `thư viện`. Use `chúng tôi` only where
  the English says "we"; the English docs almost never do, so neither should
  the Vietnamese.
- Technical, direct, unadorned — mirror the tone of the English source. Do not
  add, remove, summarize or "improve" content. Every paragraph, list item,
  table row, admonition and FAQ entry in the English file appears in the
  Vietnamese file, in the same order.
- Vietnamese runs longer than English. That is fine; do not compress two
  English sentences into one Vietnamese sentence, and do not pad a short one.
  Sentence and paragraph counts should track the source.
- Do not add sentence-final particles (`nhé`, `nha`, `đấy`, `ạ`) or exclamation
  marks the English does not have. Keep hedges as hedges: "has not been
  checked" → `chưa được kiểm chứng`, not `không hỗ trợ`.
- **Correct diacritics are mandatory.** Never write unaccented Vietnamese
  ("khong ho tro", "trong so", "huan luyen") anywhere, including inside code
  comments, YAML frontmatter, `keywords`, and alt text. Unaccented Vietnamese
  is chat shorthand, it is ambiguous, and it will be treated as a broken file.
  Watch the pairs that only diacritics separate: `mô hình` (model) vs `mồ hình`
  (nothing), `độ sâu` (depth) vs `đô sâu`, `phân đoạn` vs `phần đoạn`,
  `lớp` (layer) vs `lóp`, `ảnh` (image) vs `anh` (he/brother).
- Use `đ`/`Đ`, not `d`. Use the Vietnamese quotation marks `"..."` as in the
  English source (straight quotes are fine inside YAML); do not switch to
  `«...»` or `『...』`.
- **Never use an em dash `—`.** This site bans it in every locale except
  Chinese and Russian, where it is grammar rather than style. Vietnamese has no
  such need. Where the English uses an em dash for a parenthetical break, use a
  comma, a colon, parentheses, or split the sentence. Where it introduces a
  gloss or an explanation, use `:`. A hyphen `-` in compound identifiers,
  version strings and model names is unaffected.

## What to translate

- Frontmatter: `title`, `seo_title`, `description`, `lead`, and `faq` (both
  `q` and `a`). Quote YAML strings that contain `:` or other special chars.
- Reader-visible prose values in `meta` and `verification` frontmatter blocks
  (labels like "Install" → `Cài đặt`, "Writes" → `Kết quả ghi ra`, "Reloads
  with" → `Tải lại bằng`; prose values like "3.10 or newer" → `3.10 trở lên`).
  Values marked `mono` or that are code/identifiers stay.
- Snippet and tab `label` values that are descriptive phrases ("Video and
  streams" → `Video và luồng`, "Check a dataset" → `Kiểm tra tập dữ liệu`,
  "Instance segmentation" → `Phân đoạn thực thể`, "Use the exported file" →
  `Dùng tệp đã xuất`). Labels that are proper names stay untouched: `Python`,
  `CLI`, `Bash`, library, tool and format names.
- `keywords`: **localize, don't translate literally.** Write the queries a
  Vietnamese-speaking developer would actually type. In practice that is a
  Vietnamese phrase mixed with the English terms people really search with, and
  it is often typed without diacritics in the wild — but our `keywords` values
  are still written **with** correct diacritics (e.g.
  `"phát hiện đối tượng python"`, `"train yolo với dataset riêng"`,
  `"xuất yolo sang onnx"`, `"rtmdet không cần mmdetection"`). Keep model names
  as-is. Do not invent a textbook phrase nobody searches for.
- Body prose, headings, table cells, image alt text, admonitions, and
  `hero.caption` (the media `src`/`poster` paths stay).
- Comments **inside** code blocks and snippets (`# tự động tải về ở lần chạy
  đầu tiên`, `# tọa độ xyxy`). Code comments take diacritics like any other
  prose, and no sentence-final period.

## What stays exactly as in English

- All code (statements, identifiers, strings, CLI commands, file names, URLs)
  — only comments change. Never translate string literals passed to code.
- Frontmatter keys themselves, their order, and the structural fields:
  `families`, `last_verified`, `snippets` structure (keys, order, count),
  `language` values, hero/media paths.
- Internal and external link **targets** (translate link text, never hrefs).
- Model names (YOLOv9, RF-DETR, RTMDet...), product names, library names,
  format names (ONNX, TensorRT, CoreML), and heading anchors implied by them.
- Custom components (`<code-tabs />`, `<export-matrix />`,
  `<checkpoint-table />`, `<provenance-box>`, `<citation-block />`) and their
  attributes — including `name="predict"`, which is a key, not prose.
- Metrics, units and hardware: `mAP`, `mAP50`, `mAP50-95`, `IoU`, `NMS`, `FPS`,
  `px`, `FP16`, `INT8`, `GPU`, `CPU`, `NPU`, `CUDA`, `Apple Silicon`,
  `Raspberry Pi`, `Jetson`.

Vietnamese technical writing keeps a **very high** proportion of English terms,
much higher than Spanish or French, and pretending otherwise produces text no
Vietnamese engineer would write. Be explicit about which side of the line a
term falls on:

- **Always English, never translated:** `backbone`, `head`, `neck`, `anchor`,
  `bounding box`, `checkpoint`, `batch`, `learning rate`, `epoch`, `loss`,
  `embedding`, `feature map`, `ground truth`, `token`, `overfitting`,
  `benchmark`, `pipeline`, `runtime`, `wheel`, `repo`, `issue`, `commit`,
  `stream`, `matting`, `early stopping`, `mixed precision`, `copyleft`, `SaaS`.
  A settled Vietnamese equivalent either does not exist or exists only in
  textbooks, and using it makes the sentence harder to read, not easier.
- **Always Vietnamese, never left in English:** `trọng số` (weights),
  `huấn luyện` (training), `tinh chỉnh` (fine-tuning), `mô hình` (model),
  `ảnh` (image), `nhãn` (label), `ngưỡng` (threshold),
  `độ tin cậy` (confidence), `độ chính xác` (accuracy), `độ trễ` (latency),
  `triển khai` (deployment), `xuất` (export), `lớp` (layer),
  `độ phân giải` (resolution), `đầu vào`/`đầu ra` (input/output),
  `thư viện` (library), `giấy phép` (license).
- **Gloss once, then use English thereafter.** This is the default for
  everything in between. On its **first** appearance in a page, write the
  Vietnamese rendering followed by the English in parentheses, then use the
  bare English for the rest of the page:
  `suy luận (inference)` on first mention, then `inference`;
  `tập dữ liệu (dataset)` then `dataset`;
  `lượng tử hóa (quantization)` then `quantization`;
  `tăng cường dữ liệu (data augmentation)` then `data augmentation`.
  Never gloss the same term twice on one page, and never gloss inside a code
  comment, a heading, or a `keywords` entry. If a term appears only once on the
  page, gloss it there and be done.
  The inverse form (`inference (suy luận)`) is also acceptable when the
  surrounding sentence is already dense with English; pick one direction per
  term per page and keep it.
- Do not invent a Vietnamese term for something the English source itself
  treats as a proper noun (`Results` object → `đối tượng Results`, not
  `đối tượng Kết quả`).

## Glossary (use these consistently)

| English | Vietnamese |
|---|---|
| library | thư viện |
| model | mô hình |
| weights | trọng số |
| checkpoint | checkpoint (English; `bản checkpoint` when a classifier is needed) |
| dataset | tập dữ liệu (gloss once, then `dataset`) |
| training / to train | huấn luyện / huấn luyện |
| fine-tuning / to fine-tune | tinh chỉnh (fine-tuning) / tinh chỉnh |
| pretrained | được huấn luyện sẵn (pretrained) |
| inference | suy luận (gloss once, then `inference`) |
| to predict / prediction | dự đoán / dự đoán |
| object detection | phát hiện đối tượng |
| bounding box | bounding box (English) |
| instance segmentation | phân đoạn thực thể |
| semantic segmentation | phân đoạn ngữ nghĩa |
| panoptic segmentation | phân đoạn toàn cảnh |
| pose estimation | ước lượng tư thế |
| keypoint | keypoint (English) |
| depth estimation | ước lượng độ sâu |
| classification | phân loại |
| class (category) | lớp đối tượng (never bare `lớp`, which is `layer` here) |
| embedding | embedding (English) |
| backbone | backbone (English) |
| head (of a model) | head (English) |
| layer | lớp |
| threshold | ngưỡng |
| confidence | độ tin cậy |
| accuracy | độ chính xác |
| precision | precision (English; `độ chính xác` is already `accuracy`) |
| recall | recall (English) |
| batch / batch size | batch / kích thước batch |
| epoch | epoch (English) |
| learning rate | learning rate (English) |
| loss | loss (English; `hàm mất mát` only where the English says "loss function") |
| data augmentation | tăng cường dữ liệu (gloss once, then `data augmentation`) |
| mixed precision | mixed precision (English) |
| early stopping | early stopping (English) |
| to resume (training) | tiếp tục (huấn luyện) |
| quantization | lượng tử hóa (gloss once, then `quantization`) |
| export / to export | xuất / xuất (mô hình) |
| deployment / to deploy | triển khai / triển khai |
| label (annotation) | nhãn |
| to annotate / annotation | gán nhãn / nhãn |
| ground truth | ground truth (English) |
| feature / feature map | đặc trưng / feature map (English) |
| open-vocabulary detection | phát hiện với từ vựng mở (open-vocabulary) |
| edge device | thiết bị biên (edge device) |
| frame rate / FPS | tốc độ khung hình / FPS |
| latency | độ trễ |
| resolution | độ phân giải |
| input / output | đầu vào / đầu ra |
| image / video / stream | ảnh / video / luồng (stream) |
| mask | mặt nạ (mask) |
| oriented bounding box | hộp xoay (OBB) |
| text recognition (OCR) | nhận dạng văn bản |
| surface normals | pháp tuyến bề mặt |
| matting | matting (English) |
| restoration | phục hồi ảnh |
| mesh | lưới 3D |
| repository / repo | repo (English) |
| open source | mã nguồn mở |
| license | giấy phép |
| pipeline | pipeline (English) |
| runtime | runtime (English) |
| benchmark | benchmark (English) |

`mAP`, `mAP50`, `IoU`, `NMS`, `ONNX`, `GPU`, `CPU`, `CUDA` stay untouched.

## Formatting invariants (validated by script)

- Same number of headings, at the same levels, in the same order.
- Same number of fenced code blocks; code identical except comment lines.
- Same set of link targets.
- Same frontmatter keys as the English file — no keys added or dropped.
- No em dash `—` anywhere in the Vietnamese file, prose or frontmatter.
- Correct diacritics everywhere, including `keywords` and code comments. A file
  containing unaccented Vietnamese prose is a failed translation, not a draft.
- Do not add translator notes, disclaimers, or a "translated from" line.
