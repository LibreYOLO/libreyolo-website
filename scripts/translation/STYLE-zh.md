# Chinese translation style guide — LibreYOLO docs

How to produce `<slug>.zh.md` twins of the English docs in `content/docs/`.
The docs loader (`src/lib/docs.js`) picks up the twin automatically; the file
must be a complete, standalone copy of the English file with the rules below.

Simplified Chinese (zh-CN) only. Every rule here is derived from the published
Chinese articles in `content/articles/*.zh.md`; the docs must read like they
came from the same hand. Where those articles disagree with each other, this
guide picks one form and that form wins — see "Standardizations" at the end.

## Voice and register

- Mainland Simplified Chinese, the register of a working engineer writing for
  other engineers: 简体中文, 直接、克制、不啰嗦. Address the reader as `你`
  (used throughout the articles: 「如果你搜索的是…」、「你必须手动指定版本」),
  or drop the pronoun where Chinese reads better without it. Never `您` — the
  articles never use it, and it makes developer docs sound like a sales page.
- Technical, direct, unadorned — mirror the tone of the English source. Do not
  add, remove, summarize or "improve" content. Every paragraph, list item,
  table row, admonition and FAQ entry in the English file appears in the
  Chinese file, in the same order.
- Do not add 语气助词 (`吧`、`呢`、`啦`) or exclamation marks that the English
  does not have. Keep hedges as hedges: "has not been checked" → 尚未验证,
  not 不支持.
- Chinese is denser than English; that is fine. Do not pad a short English
  sentence into a long Chinese one, and do not merge two English sentences
  into one Chinese sentence — the paragraph and sentence count should track
  the source closely.
- Spell out what the English spells out; keep the same level of formality.

## What to translate

- Frontmatter: `title`, `seo_title`, `description`, `lead`, and `faq` (both
  `q` and `a`). Quote YAML strings that contain `:` or other special chars —
  and note that a full-width `：` does not need quoting but a half-width one
  inside a code fragment does.
- Reader-visible prose values in `meta` and `verification` frontmatter blocks
  (labels like "Install" → 安装, "Writes" → 输出, "Reloads with" → 重新加载方式;
  prose values like "3.10 or newer" → 3.10 或更高版本). Values marked `mono`
  or that are code/identifiers stay.
- Snippet and tab `label` values that are descriptive phrases ("Video and
  streams" → 视频与流, "Check a dataset" → 检查数据集, "Instance segmentation"
  → 实例分割, "Use the exported file" → 使用导出的文件). Labels that are proper
  names stay untouched: `Python`, `CLI`, `Bash`, library, tool and format names.
- `keywords`: **localize, don't translate literally.** Write the queries a
  Chinese-speaking developer would actually type — usually a Chinese phrase
  plus the English term they would really use, with no punctuation between
  (e.g. `目标检测 python`, `yolo 训练自己的数据集`, `yolo 导出 onnx`,
  `rtmdet 不用 mmdetection`). Keep model names as-is. Do not translate a
  keyword into a phrase nobody searches for (`边界框` is textbook Chinese but
  `检测框` is what people type).
- Body prose, headings, table cells, image alt text, admonitions, and
  `hero.caption` (the media `src`/`poster` paths stay).
- Comments **inside** code blocks and snippets (`# 首次运行时自动下载`,
  `# xyxy 坐标`, `# 把上色后的深度图写入磁盘`). Code comments take full-width
  `，：` but no sentence-final `。`, matching the articles.

## What stays exactly as in English

- All code (statements, identifiers, strings, CLI commands, file names, URLs)
  — only comments change. Never translate string literals passed to code.
  Never let a full-width character (`，。：（）“”「」`) into a code block, an
  inline `code span`, a path, a flag, or a version string.
- Frontmatter keys themselves, their order, and the structural fields:
  `families`, `last_verified`, `snippets` structure (keys, order, count),
  `language` values, hero/media paths.
- Internal and external link **targets** (translate link text, never hrefs).
- Model names (YOLOv9, RF-DETR, RTMDet...), product names, library names,
  format names (ONNX, TensorRT, CoreML), and heading anchors implied by them.
- Custom components (`<code-tabs />`, `<export-matrix />`,
  `<checkpoint-table />`, `<provenance-box>`, `<citation-block />`) and their
  attributes — including `name="predict"`, which is a key, not prose.

## Keeping English inline vs translating

The articles are consistent about this, and the rule is: **a term gets a
Chinese rendering only when a settled Chinese rendering exists.** Otherwise it
stays in Latin script, unitalicized, spaced as described below.

- **Always Latin:** model, family, framework, format, package, CLI, flag,
  extension, metric and hardware names (`RTMDet`, `RF-DETR`, `PyTorch`,
  `ONNX`, `TensorRT`, `NCNN`, `mmcv`, `pip`, `libreyolo`, `.tflite`, `mAP`,
  `IoU`, `NMS`, `FPS`, `px`, `INT8`, `FP16`, `GPU`, `CPU`, `NPU`, `CUDA`,
  `Apple Silicon`, `Hailo-8L`).
- **Latin because Chinese has no settled term** (attested in the articles):
  `copyleft`, `wheel`, `head`, `memory bank`, `batch-size`, `agent`,
  `transformer`, `SaaS`, `issue`, `star`.
- **Translate — do not leave in Latin:** 权重, 推理, 训练, 微调, 数据集, 量化,
  导出, 部署, 骨干, 蒸馏, 标注, 精度, 置信度, 延迟, 仓库, 上游, 运行时.
- **Gloss once, in full-width parentheses, on first mention only:**
  译名（原文）— 旷视（Megvii）、无锚框（anchor-free）、注册表（registry）、
  上色（colorize）、企业许可（Enterprise License）、强 copyleft（传染性）.
  Never gloss the same term twice on one page.
- **Foreign company / product names:** use the established Chinese name when
  one exists and the articles use it (旷视、百度、英伟达、高通、树莓派、
  中科大、北京大学、清华大学); keep Latin when no established Chinese name is
  in use (Roboflow, Deci, Meta, Google, Hugging Face, Intellindust, Jabra,
  Ultralytics). Product/SKU names stay Latin even under a Chinese company
  name: 英伟达 the company, but `NVIDIA T4`, `Jetson`, `QCS6490` the parts.
- Do not invent a Chinese term for something the English source itself treats
  as a proper noun (`Results` object → `Results` 对象, not 结果对象).

## Spacing, punctuation and script

These are hard rules; the published articles satisfy them with zero exceptions
(checked mechanically across all nine `*.zh.md` articles).

- **One half-width space between a Chinese character and an adjacent Latin or
  numeric run, on both sides.** 在 COCO 上、640 px、2024 年 4 月、每帧 19.0 毫秒、
  达到 52.6 FPS、`pip install libreyolo` 会安装…. This includes inline code
  spans and bold/emphasis runs: 采用 **AGPL-3.0** 许可.
- **No space between Latin/digits and full-width punctuation.** `AGPL-3.0，`
  not `AGPL-3.0 ，`. Full-width punctuation already carries its own sidebearing.
- **No space inside full-width brackets** and none between a Chinese character
  and full-width punctuation: 无锚框（anchor-free）检测器.
- **Full-width punctuation in prose:** `，。：；？！、（）——`. The enumeration
  comma `、` separates list items inside a sentence; `，` does not. Use the
  double em dash `——` (two characters) for parenthetical breaks, matching the
  English em dash, and never a single `—`.
- **Half-width punctuation inside code, paths, identifiers, versions, file
  names, URLs and inline code spans**, always: `format="tflite"`,
  `metrics/mAP50-95(B)`, `libreyolo[coreml]`, `v3.3.0`, `weights/x.pt`.
  A full-width character here breaks the validator's code comparison and, worse,
  breaks copy-paste for the reader.
- **Quotation marks: 「」 (and 『』 when nested).** This is the house style of
  the existing articles. Never use ASCII `"…"` in prose — it appears in two
  older articles and is being retrofitted. Use 「」 for scare quotes, quoted
  search queries and quoted document titles: 搜索「YOLOv8 可以免费商用吗」、
  标注为「TFLite (LiteRT)」.
- Numbers, versions and units stay in Arabic numerals and Latin: 五种尺寸 for a
  counted noun in flowing prose is fine, but keep `1.5.0`, `300 epochs` → 300 轮,
  `0.001 mAP`, `20 多个模型家族` in figures.
- Bullet lists that are one syntactic sentence in English keep the English
  punctuation shape: items end in `，` and the last in `。` (as in the article
  lists). Standalone bullets take `。` or no terminal mark, consistently within
  one list.
- No Traditional characters, no Taiwan/HK vocabulary (软件 not 軟體, 视频 not
  影片, 默认 not 預設, 内存 not 記憶體, 分辨率 not 解析度).

## Glossary (use these consistently)

"Attested" cites where the term is already used in `content/articles/*.zh.md`.
Rows marked *(no precedent)* are not used in the articles; they are the
recommended rendering and become binding from here on.

| English | Chinese | Attested in |
|---|---|---|
| library | 库 (`MIT 许可的 YOLO 库`) | best-ultralytics-alternatives, simplest-way-to-run-depth-anything-v2 |
| weights | 权重 | all articles (26 uses) |
| checkpoint | 检查点; gloss once as 检查点（checkpoint） | rtmdet-without-mmdetection, best-ultralytics-alternatives, depth-anything (Latin `checkpoint` also occurs — see Standardizations) |
| inference | 推理 | all articles (16 uses) |
| to fine-tune / fine-tuning | 微调 (`LoRA/DoRA 微调`) | rtmdet-without-mmdetection, best-ultralytics-alternatives |
| training / to train | 训练 | all articles |
| validation / to validate | 验证 | litert-vs-tensorflow-lite, depth-anything |
| prediction / to predict | 预测 (`跑一次预测`) | yolo-nas-with-libreyolo |
| dataset | 数据集 (`YOLO 格式数据集`) | best-ultralytics-alternatives |
| bounding box / boxes | 检测框 | yolox, yolo-nas, depth-anything |
| oriented bounding box | 旋转框 | best-ultralytics-alternatives, depth-anything |
| object detection | 目标检测 | cvpr-2026, yolo-commercial-license, depth-anything, mentions |
| instance segmentation | 实例分割 | best-ultralytics-alternatives |
| semantic segmentation | 语义分割 | best-ultralytics-alternatives |
| panoptic segmentation | 全景分割 | best-ultralytics-alternatives |
| segmentation | 分割 | all articles (25 uses) |
| pose (estimation) | 姿态 / 姿态估计 | all articles (19 uses) |
| classification | 分类 | yolo-commercial-license, best-ultralytics-alternatives |
| keypoint | 关键点 | best-ultralytics-alternatives |
| depth estimation / monocular depth | 深度估计 / 单目深度 | depth-anything |
| gaze estimation | 视线估计 | best-ultralytics-alternatives |
| tracking / multi-object tracking | 跟踪 / 多目标跟踪 | best-ultralytics-alternatives |
| quantization / to quantize | 量化 (`量化为 INT8`) | cvpr-2026, best-ultralytics-alternatives |
| export / to export | 导出 | all articles |
| deployment / to deploy | 部署 | rtmdet, best-ultralytics-alternatives, cvpr-2026 |
| latency | 延迟 | best-ultralytics-alternatives |
| accuracy | 精度 | rtmdet, yolox, yolo-nas, best-ultralytics-alternatives |
| confidence (score) | 置信度（分数） | rtmdet, yolox, yolo-nas |
| backbone | 骨干 (`ViT 骨干`、`DINO 骨干`) | best-ultralytics-alternatives |
| head (of a model) | `head` — stays in Latin (`各种 head`) | best-ultralytics-alternatives |
| loss / loss function | 损失函数 | best-ultralytics-alternatives |
| distillation / to distill | 蒸馏 | best-ultralytics-alternatives (10 uses) |
| pretraining / pretrained | 预训练 | best-ultralytics-alternatives |
| self-supervised | 自监督 | best-ultralytics-alternatives |
| data augmentation | 数据增强 | rtmdet-without-mmdetection, best-ultralytics-alternatives |
| test-time augmentation (TTA) | 测试时增强（TTA） | best-ultralytics-alternatives |
| label / annotation; unlabeled | 标注；无标注 | best-ultralytics-alternatives, litert |
| feature / features | 特征 (`DINOv3 特征`) | best-ultralytics-alternatives |
| tensor | 张量 | yolox, best-ultralytics-alternatives |
| preprocessing | 预处理 | yolox, litert |
| normalization | 归一化 | depth-anything |
| colormap / to colorize | 色彩映射 / 上色（colorize） | depth-anything |
| device placement | 设备分配 | depth-anything |
| anchor-free | 无锚框（anchor-free） | yolox, best-ultralytics-alternatives |
| open-vocabulary | 开放词汇 | best-ultralytics-alternatives |
| end-to-end | 端到端 | cvpr-2026, best-ultralytics-alternatives |
| on-device | 端侧 | litert-vs-tensorflow-lite |
| edge device / edge | 边缘设备 / 边缘 | best-ultralytics-alternatives, cvpr-2026 |
| Raspberry Pi | 树莓派 | best-ultralytics-alternatives, cvpr-2026 |
| plain CPU | 普通 CPU | 8 uses across the boilerplate footer |
| frame / ms per frame; FPS | 帧 / 每帧 X 毫秒；FPS | cvpr-2026 |
| model family | 模型家族 / 家族 | best-ultralytics-alternatives, yolo-commercial-license |
| variant / size | 变体 / 尺寸 | yolox, yolo-nas, rtmdet |
| parameter count | 参数量 | yolox, yolo-nas |
| gradient accumulation | 梯度累积 | best-ultralytics-alternatives |
| layer freezing | 层冻结 | best-ultralytics-alternatives |
| resume (training) | 断点续训 | best-ultralytics-alternatives |
| multi-GPU | 多卡训练 | best-ultralytics-alternatives |
| embedded NMS / model metadata | 内嵌 NMS / 模型元数据 | best-ultralytics-alternatives |
| sliced inference | 切片推理 | best-ultralytics-alternatives |
| license (in `X-licensed`) | 采用 X 许可 | all articles (103 uses of 许可) |
| license (standalone noun) | 许可证 (`关于许可证的说明`) | rtmdet, yolox, yolo-nas, depth-anything |
| permissive (license) | 宽松许可 | yolo-commercial-license, best-ultralytics-alternatives |
| copyleft | `copyleft`, glossed once as 强 copyleft（传染性） | yolo-commercial-license, best-ultralytics-alternatives |
| commercial use / free for commercial use | 商用 / 可免费商用 | all articles (24 uses) |
| Enterprise License | 企业许可（Enterprise License） | yolo-commercial-license |
| vendor | 厂商 | yolo-commercial-license, best-ultralytics-alternatives |
| upstream | 上游 | rtmdet, yolo-nas, depth-anything |
| repository / repo | 仓库 | all articles (23 uses) |
| runtime | 运行时 | litert, rtmdet, yolox |
| stack | 技术栈 | 12 uses across six articles |
| dependency | 依赖 | litert, yolox, yolo-nas |
| build from source | 从源码编译 | rtmdet-without-mmdetection |
| wheel | `wheel` — stays in Latin (`预编译 wheel`) | rtmdet, best-ultralytics-alternatives |
| registry | 注册表（registry） | rtmdet-without-mmdetection |
| config file | 配置文件 | best-ultralytics-alternatives (rtmdet writes `config 文件` — see Standardizations) |
| encoder | 编码器 | depth-anything FAQ, best-ultralytics-alternatives (body text writes `encoder` — see Standardizations) |
| toolchain / compiler | 工具链 / 编译器 | rtmdet, cvpr-2026 |
| benchmark | 基准测试 | cvpr-2026 |
| baseline | 基线 | best-ultralytics-alternatives |
| reproduce (a paper) | 复现 | best-ultralytics-alternatives |
| deprecated | 弃用 | litert-vs-tensorflow-lite |
| unmaintained / abandoned | 无人维护 / 已弃置 | yolox, yolo-nas, best-ultralytics-alternatives |
| actively maintained | 持续维护 | best-ultralytics-alternatives, yolo-nas |
| real-time | 实时 | all articles (21 uses) |
| telemetry | 遥测 | best-ultralytics-alternatives |
| support matrix | 支持矩阵 | litert, rtmdet |
| threshold | 阈值 | *(no precedent)* |
| confidence threshold / IoU threshold | 置信度阈值 / IoU 阈值 | *(no precedent)* |
| batch / batch size | 批大小 (keep `batch=16` in code; `batch-size` stays Latin when naming the upstream issue) | yolo-nas uses Latin `batch-size`; 批大小 *(no precedent)* |
| epoch | 轮 (`300 轮`) | *(no precedent)* |
| learning rate | 学习率 | *(no precedent)* |
| warmup | 预热 | *(no precedent)* |
| postprocessing | 后处理 | *(no precedent)* |
| feature map | 特征图 | *(no precedent)* |
| embedding | 嵌入向量 (note: 嵌入 alone means "to embed" in these articles) | *(no precedent)* |
| ground truth | 真值（ground truth） | *(no precedent)* |
| half precision | 半精度 | *(no precedent)* |
| precision / recall (metrics) | 查准率 / 查全率 | *(no precedent)* — do not reuse 精度, which is reserved for accuracy |

`mAP`、`mAP50`、`mAP50-95`、`AP`、`IoU`、`NMS`、`ONNX`、`TensorRT`、`CoreML`、
`OpenVINO`、`NCNN`、`TFLite`、`GPU`、`CPU`、`NPU`、`CUDA`、`FP16`、`INT8`、`px`、
`FPS` stay untouched.

## Formatting invariants (validated by script)

Run `node scripts/translation/validate.mjs zh [section/slug ...]` before
committing. It compares the twin against its English source and fails on drift.

- Same number of headings, at the same levels, in the same order.
- Same number of fenced code blocks; code identical except comment lines.
  A stray full-width `，` or `（` inside a code block is the most common way a
  Chinese twin fails this check.
- Same set of link targets.
- Same frontmatter keys as the English file — no keys added or dropped;
  `families`, `last_verified` and `layout` byte-identical; `hero.src` and
  `hero.poster` byte-identical.
- Same snippet structure: same group keys, same order, same count, same
  `language` values. Only `label` and in-code comments change.
- Do not add translator notes, disclaimers, or a "translated from" line.

## Standardizations (the articles disagree; these rulings win)

Where the published `*.zh.md` articles contradict each other, docs follow the
ruling below. The articles should be retrofitted to match.

1. **checkpoint** — 检查点 in prose and headings, glossed once as
   检查点（checkpoint）. (`checkpoint` in Latin appears 6×; 检查点 3×; rtmdet and
   depth-anything use 检查点 in the FAQ and `checkpoint` in the body of the
   *same file*.)
2. **Quotation marks** — 「」. (Six articles use 「」; litert and
   yolo-commercial-license use ASCII `"…"`. No article uses `“”`. If the project
   would rather follow GB/T 15834 and use `“”`, change it here once and retrofit
   all nine articles — do not mix.)
3. **Apache-2.0** — hyphenated, always, and likewise `AGPL-3.0`, `GPL-3.0`,
   `CC-BY-NC-4.0`, `BSD-3-Clause`. (rtmdet and yolox write `Apache 2.0`;
   best-ultralytics and yolo-commercial-license write `Apache-2.0`;
   depth-anything uses both forms in one file.)
4. **encoder** — 编码器. (depth-anything writes 编码器 in the FAQ and `encoder`
   in the body of the same file.)
5. **config file** — 配置文件. (rtmdet writes `config 文件`.)
6. **bit-exact** — 逐位一致. (rtmdet writes 逐位一致 in the FAQ and 逐比特一致 in
   the body of the same file.)
7. **pipeline** — 流水线. (rtmdet writes 管线 in the FAQ and 流水线 in the body.)
8. **plain CPU** — 普通 CPU. (`纯 CPU` appears twice against eight.)
9. **Raspberry Pi** — 树莓派. (cvpr-2026 writes `Raspberry Pi 5` in the FAQ and
   树莓派 5 in the body of the same file.)
10. **NVIDIA the company** — 英伟达; the parts stay Latin (`NVIDIA T4`,
    `Jetson`). (yolo-nas writes `NVIDIA` in the FAQ and 英伟达 in the body.)
11. **license** — 许可 in attributive use (采用 MIT 许可), 许可证 only as a
    standalone noun or in a formal license title
    (GNU Affero 通用公共许可证 v3.0). Do not write 授权 or 协议.
12. **"Star it on GitHub" boilerplate** — 在 GitHub 上点个 star (five articles)
    rather than 给它加星 (two).
