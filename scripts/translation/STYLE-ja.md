# Japanese translation style guide — LibreYOLO docs

How to produce `<slug>.ja.md` twins of the English docs in `content/docs/`.
The docs loader (`src/lib/docs.js`) picks up the twin automatically; the file
must be a complete, standalone copy of the English file with the rules below.

Japanese (ja-JP) only. There is no published Japanese corpus on this site yet,
so unlike `STYLE-zh.md` nothing here is "attested": every ruling below is
prescriptive and becomes binding from the first `*.ja.md` page onward. If you
disagree with a call, change it here once and retrofit, rather than diverging
in a single file.

## Voice and register

- **Register: ですます体. Use it for all body prose, and never mix it with
  である/だ体 inside a page.** Justification: these are user-facing product
  docs, not a paper or a spec. ですます is what the Japanese-language docs a
  LibreYOLO reader already uses are written in (Microsoft Learn, AWS, Google
  Cloud, PyTorch and Hugging Face community translations), and it is the
  default of the JTF 日本語標準スタイルガイド for 説明文 aimed at users.
  である体 reads as a 論文/規格書 voice and, next to the plain-spoken English
  source, would come across as colder and more institutional than the original.
  The one place である is acceptable is a fixed idiom inside a quoted title.
- **体言止め (noun-final) is not a register violation and is preferred where
  the English itself is a fragment**: headings, table cells, `title`, snippet
  and tab `label`s, badges, admonition labels, image alt text, `hero.caption`,
  and the short blurbs that are noun phrases in English ("Contours and
  boundaries." → 「輪郭と境界。」). A fragment in English stays a fragment in
  Japanese; a full sentence in English becomes a ですます sentence.
- Code comments take neither ですます nor a sentence-final `。`. Write them as
  短い体言止め or plain 連体/連用 form: `# 初回実行時に自動ダウンロード`,
  `# xyxy 座標`. This matches how Japanese developers actually comment code.
- Technical, direct, unadorned: mirror the tone of the English source. Do not
  add, remove, summarize or "improve" content. Every paragraph, list item,
  table row, admonition and FAQ entry in the English file appears in the
  Japanese file, in the same order.
- Address the reader with the plain imperative/potential (「〜してください」、
  「〜できます」). No あなた as a pronoun, no 貴殿/皆様, no 敬語 beyond ですます
  (avoid いたします、ございます、〜させていただきます). Do not add 丁寧語
  softeners the English does not have.
- No 終助詞 (`ね`、`よ`、`かな`) and no exclamation marks the English does not
  have. Keep hedges as hedges: "has not been checked" → 「未検証です」,
  not 「対応していません」.
- Japanese is longer than English in characters but should track it in
  structure: do not merge two English sentences into one Japanese sentence, and
  do not split one into three. Sentence and paragraph counts should track the
  source closely.
- Prefer ひらがな for 補助動詞 and formal nouns, per JTF: こと (not 事)、
  もの (not 物)、ため (not 為)、とき (not 時, when it means "when")、
  できる (not 出来る)、ください (not 下さい)、および・または・ただし・
  なお・すでに・さらに. Keep 表外漢字 out of prose.
- Spell out what the English spells out; keep the same level of formality.

## What to translate

- Frontmatter: `title`, `seo_title`, `description`, `lead`, and `faq` (both
  `q` and `a`). Quote YAML strings that contain `:` or other special chars.
  A full-width `：` does not need quoting; a half-width one inside a code
  fragment does. Quote any value that begins with `「`, and always quote
  values containing `#`.
- Reader-visible prose values in `meta` and `verification` frontmatter blocks
  (labels like "Install" → 「インストール」, "Writes" → 「出力」, "Reloads with"
  → 「再読み込み」; prose values like "3.10 or newer" → 「3.10 以降」). Values
  marked `mono` or that are code/identifiers stay.
- Snippet and tab `label` values that are descriptive phrases ("Video and
  streams" → 「動画とストリーム」, "Check a dataset" → 「データセットを確認」,
  "Instance segmentation" → 「インスタンスセグメンテーション」, "Use the
  exported file" → 「エクスポートしたファイルを使う」). Labels that are proper
  names stay untouched: `Python`, `CLI`, `Bash`, library, tool and format names.
- `keywords`: **localize, don't translate literally.** Write the queries a
  Japanese-speaking developer would actually type, which is usually a Japanese
  phrase plus the English term they would really use, separated by a half-width
  space (e.g. `物体検出 python`、`yolo 自作データセット 学習`、
  `yolo onnx エクスポート`、`rtmdet mmdetection なし`). Keep model names as-is.
  Do not translate a keyword into a phrase nobody searches for: 「物体検出」 is
  what people type, 「オブジェクト検出」 is not.
- Body prose, headings, table cells, image alt text, admonitions, and
  `hero.caption` (the media `src`/`poster` paths stay).
- Comments **inside** code blocks and snippets (`# 初回実行時に自動ダウンロード`).

## What stays exactly as in English

- All code (statements, identifiers, strings, CLI commands, file names, URLs):
  only comments change. Never translate string literals passed to code. Never
  let a full-width character (`、。：（）「」％＝－`) into a code block, an
  inline `code span`, a path, a flag, or a version string.
- Frontmatter keys themselves, their order, and the structural fields:
  `families`, `last_verified`, `snippets` structure (keys, order, count),
  `language` values, hero/media paths.
- Internal and external link **targets** (translate link text, never hrefs).
- Model names (YOLOv9, RF-DETR, RTMDet...), product names, library names,
  format names (ONNX, TensorRT, CoreML), and heading anchors implied by them.
- Custom components (`<code-tabs />`, `<export-matrix />`,
  `<checkpoint-table />`, `<provenance-box>`, `<citation-block />`) and their
  attributes, including `name="predict"`, which is a key and not prose.
- Terms the English source itself treats as a proper noun: the `Results`
  object stays 「`Results` オブジェクト」, never 「結果オブジェクト」.

## Japanese-only rules: spacing, punctuation and script

These are hard rules. A reviewer should be able to fail a page on any one of
them without a discussion.

### Spacing between Japanese and Latin/numeric runs

- **Do not insert a space at a Japanese/Latin or Japanese/numeral boundary.**
  Write 「ONNXにエクスポートします」、「300エポック」、「HuggingFaceから
  ダウンロードします」、「mAP50-95が向上します」、「`pip install libreyolo`
  を実行します」. This follows the JTF 日本語標準スタイルガイド
  (和欧文間にスペースを入れない) and Microsoft's Japanese style, and it is the
  opposite of the rule in `STYLE-zh.md`; the two languages genuinely differ
  here, so do not copy spacing habits across from a `*.zh.md` twin. Rationale
  beyond convention: an inserted half-width space is a real break opportunity,
  so it lets the line wrap in places Japanese 禁則処理 would not, and it makes
  the same term look different depending on whether a particle follows.
- **Spacing *inside* a Latin run is preserved exactly as the English has it**:
  `640 px`、`Apple Silicon`、`Hugging Face`、`0.001 mAP`、`v1.5.0`. Do not
  compress `640 px` to `640px` and do not expand `mAP50-95`.
- The only half-width space you add is between keyword tokens in `keywords`
  (see above) and inside code, where the code decides.
- Never use a full-width space (U+3000) anywhere, including for indentation.
  Never use 半角カナ (`ﾃﾞｰﾀ`). Never use full-width Latin or full-width digits
  (`ＯＮＮＸ`、`６４０`); numbers are always half-width Arabic numerals.

### Full-width vs half-width punctuation

- **Prose takes full-width Japanese punctuation: `、。「」『』（）：；？！・…`.**
  Use 読点 `、` and 句点 `。`, never the academic `，．` pair and never ASCII
  `,` `.` in a Japanese sentence.
- **Half-width and ASCII punctuation appear only inside code**: code blocks,
  inline code spans, identifiers, paths, flags, file names, versions, URLs and
  YAML/JSON structure. `format="tflite"`, `metrics/mAP50-95(B)`,
  `libreyolo[coreml]`, `v1.5.0`, `weights/x.pt`. This is not a stylistic
  preference: a full-width character here breaks the validator's code
  comparison and breaks copy-paste for the reader.
- A sentence that ends on a code span still takes a full-width `。` outside the
  backticks: 「デフォルトは `0.25` です。」
- **Parentheses**: full-width `（）` in prose, including when the content
  inside is Latin: 「アンカーフリー（anchor-free）」. Half-width `()` only
  inside code and inside identifiers that contain them (`mAP50-95(B)`,
  `LibreYOLO()` when written as a code span). Never a space next to `（` or `）`.
- **Quotation marks: `「」`, and `『』` when nested.** Use them for scare
  quotes, quoted search queries and quoted titles: 「YOLOv8は商用利用できるか」
  で検索する. Never ASCII `"…"` and never `“”` in Japanese prose.
- **The middle dot `・`** joins coordinate items and multi-word transliterated
  names: 「学習・検証・エクスポート」、「クラス・アクティベーション・マップ」.
  It does not replace `、` in a list of clauses.
- **No em dashes.** `—` is banned across this website (see `AGENTS.md`), and
  the Chinese exception for `——` does not extend to Japanese. Do not
  substitute a Japanese dash (`―`、`――`、`ー`) for it either. Recast instead:
  split into two sentences, use 「：」, use 「、」, or use 「（）」. An English
  parenthetical break such as 「No external dependencies, no complex setup -
  just flags and function calls.」 becomes two Japanese sentences:
  「外部依存も複雑な設定も不要です。フラグと関数呼び出しだけで使えます。」
  The same applies to the hyphen and the en dash used as a break: recast, never
  transliterate the punctuation.
- Ranges in prose use `〜` (「5〜10エポック」); ranges inside identifiers and
  metric names keep the ASCII hyphen (`mAP50-95`).
- Ellipsis is `…` (doubled as `……` when it stands for an omission), never
  `...` in prose.
- Do not add `？` or `！` the English does not have. A Japanese question ending
  in 「〜か」 takes `。`, not `？`, unless the English had a question mark.
- Bullet lists follow the shape of the English: if the items are fragments,
  end them with `。` or with nothing, consistently within one list; if they are
  sentences, end every one with `。`.

### Particles attaching to Latin identifiers

- **The particle goes outside the code span, with no space**:
  「`model.export()`でONNXを書き出します」、「`conf`は信頼度のしきい値です」、
  「`LibreRTMDets.pt`を読み込みます」. Never put a particle inside backticks,
  and never let a code span swallow the `を`/`は`/`が` that follows it.
- **Do not inflect an English identifier as a Japanese verb.** Write
  「`train()` を呼び出します」, not 「`train()`する」 (and note that per the
  spacing rule above, a code span whose content is Latin is still set without
  a boundary space: 「`train()`を呼び出します」).
- **Do not pluralize.** Japanese has no plural agreement, so `boxes` renders as
  「`boxes`」 or 「検出ボックス」, never 「ボックスたち」/「ボックス達」.
- An identifier that ends in `.`, `)`, `_` or a digit is still followed
  directly by its particle; do not insert a `の` to smooth it over unless the
  identifier is genuinely a modifier: 「`Results`オブジェクト」 (compound, no
  `の`) but 「`snippets`の構造」 (possessive, `の` required).
- When an identifier is the subject of the sentence, prefer `は`/`が` over
  turning it into a topic phrase with 「という」. 「`pretrained`引数は読まれ
  ません」, not 「`pretrained`というものは…」.

### Katakana

- **Keep the trailing long vowel `ー` on loanwords of three or more mora**
  (JTF and Microsoft style, not the old JIS Z 8301 rule): ユーザー、サーバー、
  エンコーダー、デコーダー、コンピューター、フォルダー、レイヤー. Do not mix
  「エンコーダ」 and 「エンコーダー」 within the corpus.
  Fixed exceptions, because the shortened form is the dominant written form in
  Japanese ML writing and in search: **コンピュータビジョン**、**パラメータ**、
  **テンソル**、**ベクトル**、**リポジトリ**、**メモリ**.
- Do not katakanize a term that has a settled kanji rendering (see the
  glossary): 推論 not インファレンス, 重み not ウェイト, 学習率 not
  ラーニングレート.
- Do not katakanize a name that is normally written in Latin script: ONNX,
  TensorRT, PyTorch, Raspberry Pi, NVIDIA, Hugging Face all stay in Latin.

## Glossary (use these consistently)

The "Script" column is the ruling that matters: Japanese ML writing mixes
kanji, katakana and bare Latin, and picking the wrong one for a given term is
the most common way a Japanese translation reads as machine output.

| English | Japanese | Script ruling |
|---|---|---|
| library | ライブラリ | katakana |
| weights | 重み | kanji, never ウェイト |
| checkpoint | チェックポイント | katakana |
| dataset | データセット | katakana, never 「データ集合」 |
| training / to train | 学習 / 学習する | kanji; 訓練 only inside a quoted upstream term |
| validation / to validate | 検証 / 検証する | kanji |
| inference / to infer | 推論 / 推論する | kanji, never インファレンス |
| prediction / to predict | 推論 (verb 予測する) | kanji; `predict()` stays Latin |
| fine-tuning / to fine-tune | ファインチューニング / ファインチューニングする | katakana; 「微調整」 only where the English is generic |
| pretrained / pretraining | 学習済み / 事前学習 | kanji |
| object detection | 物体検出 | kanji, never オブジェクト検出 |
| bounding box | バウンディングボックス | katakana; 「検出ボックス」 acceptable when shortening in running prose |
| oriented bounding box | 回転バウンディングボックス | kanji + katakana |
| segmentation | セグメンテーション | katakana, never 領域分割 |
| instance segmentation | インスタンスセグメンテーション | katakana |
| semantic segmentation | セマンティックセグメンテーション | katakana |
| panoptic segmentation | パノプティックセグメンテーション | katakana |
| mask | マスク | katakana |
| pose estimation | 姿勢推定 | kanji |
| keypoint | キーポイント | katakana |
| depth estimation | 深度推定 | kanji |
| classification | 分類 | kanji |
| embedding | 埋め込みベクトル (verb 埋め込む) | kanji; 「エンベディング」 only in `keywords` |
| feature / features | 特徴量 | kanji |
| feature map | 特徴マップ | kanji + katakana, not フィーチャーマップ |
| backbone | バックボーン | katakana, never 骨格 |
| head (of a model) | ヘッド | katakana |
| neck | ネック | katakana |
| layer | 層 | kanji in running prose (第3層); レイヤー only when naming a UI/API concept |
| loss / loss function | 損失 / 損失関数 | kanji |
| threshold | しきい値 | mixed kana+kanji; do **not** write 閾値 or スレッショルド |
| confidence (score) | 信頼度（スコア） | kanji |
| accuracy | 精度 | kanji; reserved for accuracy, do not reuse for precision |
| precision / recall (metrics) | 適合率 / 再現率 | kanji; never 精度 for precision |
| batch / batch size | バッチ / バッチサイズ | katakana; `batch=16` stays as code |
| epoch | エポック | katakana |
| learning rate | 学習率 | kanji, never ラーニングレート |
| warmup | ウォームアップ | katakana |
| data augmentation | データ拡張 | katakana + kanji, not オーグメンテーション |
| mixed precision / half precision | 混合精度 / 半精度 | kanji |
| early stopping | 早期終了 | kanji; gloss once as 早期終了（early stopping） |
| quantization / to quantize | 量子化 / 量子化する | kanji |
| export / to export | エクスポート / エクスポートする | katakana; 「書き出し」 only for files on disk |
| deployment / to deploy | デプロイ / デプロイする | katakana, never 配備 |
| label (annotation) | ラベル / アノテーション | katakana |
| ground truth | 正解データ | kanji; gloss once as 正解データ（ground truth） |
| open-vocabulary detection | オープンボキャブラリ検出 | katakana + kanji |
| anchor-free | アンカーフリー | katakana; gloss once as アンカーフリー（anchor-free） |
| end-to-end | エンドツーエンド | katakana |
| edge device / edge | エッジデバイス / エッジ | katakana |
| on-device | オンデバイス | katakana |
| microcontroller | マイコン | katakana (the ordinary Japanese short form) |
| tensor | テンソル | katakana, no trailing ー |
| preprocessing / postprocessing | 前処理 / 後処理 | kanji |
| normalization | 正規化 | kanji |
| distillation / to distill | 蒸留 / 蒸留する | kanji |
| confusion matrix | 混同行列 | kanji |
| tiled / sliced inference | タイル分割推論 | katakana + kanji |
| latency | レイテンシ | katakana |
| frame rate / FPS | FPS | Latin, untranslated |
| real-time | リアルタイム | katakana |
| runtime | ランタイム | katakana |
| dependency | 依存関係 | kanji |
| repository / repo | リポジトリ | katakana, no trailing ー |
| upstream | アップストリーム | katakana; 「本家」 is too colloquial for docs |
| registry | レジストリ | katakana |
| config file | 設定ファイル | kanji + katakana |
| wheel (Python) | `wheel` | stays Latin |
| model family | モデルファミリー | katakana |
| variant / size | バリアント / サイズ | katakana |
| benchmark / baseline | ベンチマーク / ベースライン | katakana |
| license | ライセンス | katakana; 「使用許諾」 only in a formal license title |
| permissive (license) | 寛容なライセンス | kanji |
| copyleft | コピーレフト | katakana |
| commercial use | 商用利用 | kanji |
| open source | オープンソース | katakana |
| explainability / interpretability | 説明可能性 / 解釈可能性 | kanji |
| heatmap | ヒートマップ | katakana |
| attention mechanism | アテンション機構 | katakana + kanji |
| deprecated | 非推奨 | kanji |
| multi-GPU | マルチGPU | katakana + Latin, no boundary space |

`mAP`、`mAP50`、`mAP50-95`、`AP`、`IoU`、`NMS`、`ONNX`、`TensorRT`、`CoreML`、
`OpenVINO`、`NCNN`、`TFLite`、`GPU`、`CPU`、`NPU`、`CUDA`、`FP16`、`INT8`、`px`、
`FPS`、`SaaS`、`API`、`CLI` stay untouched.

## Formatting invariants (validated by script)

Run `node scripts/translation/validate.mjs ja [section/slug ...]` before
committing. It compares the twin against its English source and fails on drift.

- Same number of headings, at the same levels, in the same order.
- Same number of fenced code blocks; code identical except comment lines.
  A stray full-width `、`、`（` or `：` inside a code block is the most common
  way a Japanese twin fails this check.
- Same set of link targets.
- Same frontmatter keys as the English file, with no keys added or dropped;
  `families`, `last_verified` and `layout` byte-identical; `hero.src` and
  `hero.poster` byte-identical.
- Same snippet structure: same group keys, same order, same count, same
  `language` values. Only `label` and in-code comments change.
- No `—` anywhere in the file, in prose or in a code comment.
- No full-width space (U+3000), no 半角カナ, no full-width Latin or digits.
- Do not add translator notes, disclaimers, or a "translated from" line.
