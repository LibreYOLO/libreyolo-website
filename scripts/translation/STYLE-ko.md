# Korean translation style guide — LibreYOLO docs

How to produce `<slug>.ko.md` twins of the English docs in `content/docs/`.
The docs loader (`src/lib/docs.js`) picks up the twin automatically; the file
must be a complete, standalone copy of the English file with the rules below.

## Voice and register

- **합니다체 throughout.** Declarative sentences end in `-습니다 / -ㅂ니다`
  (`가중치는 처음 실행할 때 내려받아 로컬에 캐시됩니다`). This is the register
  Korean product and API documentation is written in, and it is the only one of
  the three that survives a page containing installation steps, hedged
  reliability notes, table cells and FAQ answers at once.
  - **Not 해요체** (`-해요 / -이에요`). It is conversational: it reads like a
    personal blog or an onboarding screen, it invites the exclamation marks and
    softeners the English source does not have, and it clashes with the flat,
    terse tone of these pages.
  - **Not 한다체** (`-한다 / -이다`). It is the register of papers, news and
    release notes. In user-facing docs it reads as curt, and it cannot host the
    polite instructions the Install and Train sections need without an abrupt
    register switch mid-page.
  - Never mix the three inside a page, a table, or a list. One page, one
    register.
- Prefer plain declaratives over imperatives: `conf로 신뢰도 임계값을 설정합니다`
  reads better than an order. When an instruction genuinely has to be an
  imperative, use `-하십시오`. Never `-해라 / -해 / -하자`. `-하세요` is allowed
  only in CTA-style button labels, not in body prose.
- **No second-person pronoun.** Korean docs do not say `당신` or `여러분`.
  English "you" almost always disappears (`your model` → `모델`,
  `you can resume` → `이어서 학습할 수 있습니다`). Use `사용자` only when the
  sentence really needs a third party.
- **Headings are noun phrases**, not verbs or sentences: `설치`, `예측`, `학습`,
  `검증`, `내보내기`, `변형`, `체크포인트`, `라이선스`, `인용`. Buttons and
  CTA-style labels take the `-하기` nominal (`시작하기`, `문서 보기`).
- Technical, direct, unadorned: mirror the tone of the English source. Do not
  add, remove, summarize or "improve" content. Every paragraph, list item,
  table row, admonition and FAQ entry in the English file appears in the Korean
  file, in the same order. Do not merge two English sentences into one Korean
  sentence, and do not split one into two.
- Keep hedges as hedges. "What has not been checked" is `검증되지 않은 항목`,
  not `지원하지 않는 기능`. "currently" is `현재`, and it stays.
- Ban translationese (번역투). No double passive `-되어집니다` (write `-됩니다`),
  no `-에 의해` where a plain subject works, no `가지고 있습니다` for "has"
  (write `있습니다`), no `~중의 하나입니다` for "one of the", no `~을 통하여`
  for "through". Prefer active voice; English passives usually become Korean
  `-됩니다` forms, not a chain of nominalizations.
- Spell out what the English spells out; keep the same level of formality.

## What to translate

- Frontmatter: `title`, `seo_title`, `description`, `lead`, and `faq` (both
  `q` and `a`). Quote YAML strings that contain `:` or other special chars.
  Korean sentences hit this often (`"설치: 필요한 것"`).
- Reader-visible prose values in `meta` and `verification` frontmatter blocks
  (labels like "Install" → `설치`, prose values like "3.10 or newer" →
  `3.10 이상`). Values marked `mono` or that are code/identifiers stay.
- Snippet and tab `label` values that are descriptive phrases ("Video and
  streams" → `비디오와 스트림`, "Check a dataset" → `데이터셋 확인`,
  "Instance segmentation" → `인스턴스 분할`, "Use the exported file" →
  `내보낸 파일 사용하기`). Labels that are proper names stay untouched:
  `Python`, `CLI`, `Bash`, library, tool and format names.
- `keywords`: **localize, don't translate literally.** Write the queries a
  Korean-speaking developer would actually type, which is usually a Korean
  phrase mixed with the English terms they really use, lowercase, no
  punctuation between (e.g. `객체 탐지 파이썬`, `yolo 커스텀 데이터셋 학습`,
  `yolo onnx 변환`, `rtmdet mmdetection 없이`). Keep model names as-is. Do not
  translate a keyword into a phrase nobody searches for (`경계 상자` is textbook
  Korean, `바운딩 박스` is what people type).
- Body prose, headings, table cells, image alt text, admonitions, and
  `hero.caption` (the media `src`/`poster` paths stay).
- Comments **inside** code blocks and snippets (`# 처음 실행할 때 자동으로 내려받습니다`).
  Code comments take ASCII punctuation only, and they drop the sentence-final
  period when the English comment has none.

## What stays exactly as in English

- All code (statements, identifiers, strings, CLI commands, file names, URLs):
  only comments change. Never translate string literals passed to code.
- Frontmatter keys themselves, their order, and the structural fields:
  `families`, `last_verified`, `snippets` structure (keys, order, count),
  `language` values, hero/media paths.
- Internal and external link **targets** (translate link text, never hrefs).
- Model names (YOLOv9, RF-DETR, RTMDet...), product names, library names,
  format names (ONNX, TensorRT, CoreML), and heading anchors implied by them.
- Custom components (`<code-tabs />`, `<export-matrix />`,
  `<checkpoint-table />`, `<provenance-box>`, `<citation-block />`) and their
  attributes, including `name="predict"`, which is a key and not prose.
- Do not invent a Korean name for something the English source itself treats as
  a proper noun: `Results` object → `Results` 객체, not `결과 객체`.

## Korean-only rules: spacing, particles and punctuation

### Hangul, transliteration and Latin script

Three buckets. Decide by this list and by the glossary, not by feel: the same
term must not appear in Hangul on one page and in Latin on the next.

- **Translated into Korean** (a settled Sino-Korean or native word exists):
  가중치, 추론, 학습, 검증, 예측, 분류, 분할, 탐지, 정확도, 정밀도, 재현율,
  임계값, 신뢰도, 손실 함수, 양자화, 증류, 배포, 저장소, 지연 시간, 처리량,
  전처리, 후처리, 혼동 행렬, 조기 종료, 학습률, 가중치 감쇠, 데이터 증강,
  깊이 추정, 자세 추정, 시선 추정, 표면 법선, 중심점, 실시간, 의존성,
  특징 맵, 사전 학습, 상업적 사용.
- **Transliterated into Hangul** (외래어 표기; the community says it this way,
  and a "proper" Korean coinage would read as a back-translation exercise):
  데이터셋, 체크포인트, 임베딩, 백본, 헤드, 넥, 레이어, 배치, 에폭, 파인튜닝,
  바운딩 박스, 인스턴스, 시맨틱, 판옵틱, 키포인트, 마스크, 클래스, 텐서,
  레이블, 어노테이션, 파이프라인, 런타임, 프레임워크, 라이브러리, 패키지,
  벤치마크, 베이스라인, 업스트림, 라이선스, 옵티마이저, 메시, 매팅, 워밍업,
  스트리밍, 메타데이터, 플래그, 프레임, 앵커 프리, 머신러닝, 컴퓨터 비전,
  오픈 소스, 엣지 디바이스, 온디바이스.
- **Kept in Latin script**: model and family names (YOLOv9, YOLOX, RF-DETR,
  RTMDet, RTMDet-Ins, SAM, MiDaS, LibreYOLO), format and runtime names (ONNX,
  TensorRT, CoreML, OpenVINO, TFLite, PyTorch, CUDA), tool and site names
  (Python, NumPy, OpenCV, PIL, Hugging Face, GitHub), license identifiers (MIT,
  AGPL-3.0, Apache-2.0), metric and algorithm abbreviations (mAP, mAP50,
  mAP50-95, AP, IoU, NMS, FPS), hardware abbreviations (GPU, CPU, TPU, NPU,
  RAM, VRAM), API identifiers, flags, file suffixes and version strings.
- Follow 외래어 표기법 for the transliterated bucket, with one deliberate
  exception. Write `라이선스` (never 라이센스), `메시` (never 메쉬), `레이블`
  (never 라벨), `배지` (never 뱃지), `컨트롤러` (never 콘트롤러). The exception
  is `엣지`: the prescriptive spelling is `에지`, but no Korean CV engineer
  writes or searches for it, so this project uses `엣지 디바이스`, `엣지 검출`.
- Gloss a term in parentheses on **first mention only**, and only when the
  Korean rendering is not yet self-evident: `앵커 프리(anchor-free)`,
  `정답(ground truth)`, `오픈 보캐뷸러리(open-vocabulary)`. Never gloss the same
  term twice on one page.

### Spacing between Hangul and Latin or numeric runs

Korean already spaces its words, so a Latin or numeric run is simply another
word. That gives two rules that are easy to state and easy to get wrong.

- **One ordinary space on each side of a Latin or numeric run when it is a
  separate word**: `ONNX 파일`, `COCO 데이터셋`, `Python 스크립트`, `CLI 명령`,
  `mAP50 값`, `640 픽셀`, `300 에폭`, `35달러 보드` (a currency amount closes up,
  a counted unit noun does not). Never close up the noun: `ONNX모델` is wrong.
- **No space before an attached particle (조사) or copula.** The particle binds
  to the token, in Latin script or not: `ONNX로`, `GPU에서`, `LibreYOLO는`,
  `640으로`, `t부터 x까지`. Never `ONNX 로`, never `GPU 에서`.
- The same holds after an inline code span: write `` `conf`는 ``, not
  `` `conf` 는 ``, and `` `-seg` 접미사 `` with a space because `접미사` is a
  separate word.
- Do not use a hyphen to join a Latin word to a Hangul one. Korean takes a
  space, not `ONNX-모델`.
- Ordinary spaces only. No non-breaking spaces, no thin spaces, no full-width
  space: they are invisible in review and noisy in diffs.
- Get the ordinary 띄어쓰기 right too, since it is the most common review
  finding: `할 수 있습니다`, `하지 않습니다`, `첫 번째`, `한 줄`, `그 외`,
  `모델 등`. Bound nouns (`것`, `수`, `때`, `뿐`) take a space before them.
- Compound loanwords are closed or open by convention, and the convention is
  fixed here: `데이터셋`, `머신러닝`, `백본`, `체크포인트` closed;
  `바운딩 박스`, `오픈 소스`, `컴퓨터 비전`, `엣지 디바이스`, `특징 맵` open.

### Particle selection after a Latin-script identifier

`은/는`, `이/가`, `을/를`, `과/와`, `으로/로` are chosen by the **final sound of
the preceding word as it is read aloud in Korean**, not by its spelling. A
consonant-final (받침 있는) reading takes `은 / 이 / 을 / 과 / 으로`; a
vowel-final reading takes `는 / 가 / 를 / 와 / 로`. `ㄹ`-final readings are the
one exception: they take `로`, not `으로`.

So read the identifier the way a Korean developer would say it, and use that.

| Token | Korean reading | Final sound | Particles |
|---|---|---|---|
| `ONNX` | 오엔엔엑스 | vowel (스) | ONNX**는**, ONNX**가**, ONNX**를**, ONNX**로** |
| `GPU` | 지피유 | vowel (유) | GPU**는**, GPU**가**, GPU**를**, GPU**로** |
| `TensorRT` | 텐서알티 | vowel (티) | TensorRT**는**, TensorRT**를**, TensorRT**로** |
| `LibreYOLO` | 리브레욜로 | vowel (로) | LibreYOLO**는**, LibreYOLO**가**, LibreYOLO**를** |
| `CoreML` | 코어엠엘 | consonant ㄹ | CoreML**은**, CoreML**이**, CoreML**을**, CoreML**로** |
| `Python` | 파이썬 | consonant ㄴ | Python**은**, Python**이**, Python**을**, Python**으로** |
| `RTMDet` | 알티엠뎃 | consonant ㅅ | RTMDet**은**, RTMDet**이**, RTMDet**을**, RTMDet**으로** |
| `YOLOv9` | 욜로 브이나인 | consonant ㄴ | YOLOv9**은**, YOLOv9**이**, YOLOv9**을** |
| `pip` | 핍 | consonant ㅂ | pip**은**, pip**으로** |
| `640` | 육백사십 | consonant ㅂ | 640**은**, 640**으로** |
| `300` | 삼백 | consonant ㄱ | 300**은**, 300**으로** |
| `1.5.0` | 일 점 오 점 영 | consonant ㅇ | 1.5.0**은**, 1.5.0**으로** |
| `0.5` | 영 점 오 | vowel (오) | 0.5**는**, 0.5**로** |

Two shortcuts that cover almost every case in these docs:

- **Letter-by-letter acronyms:** only a final `L` (엘), `M` (엠) or `N` (엔)
  gives a consonant-final reading. Every other letter name ends in a vowel:
  A 에이, B 비, C 시, D 디, E 이, F 에프, G 지, H 에이치, I 아이, J 제이,
  K 케이, O 오, P 피, Q 큐, S 에스, T 티, U 유, V 브이, W 더블유, X 엑스,
  Y 와이, Z 제트. So `NMS는`, `IoU는`, `AP는`, `FPS는`, `CPU는`, but `XML은`,
  `HTML은`, `CNN은`. `R` is the one genuinely ambiguous letter (`아르`
  prescriptively, `알` in speech): do not end a clause on a bare `R`-final
  acronym, put a noun after it instead.
- **Sino-Korean numerals:** 영, 일, 삼, 육, 칠, 팔, 십, 백, 천, 만 are
  consonant-final; 이, 사, 오, 구 are vowel-final. Read the whole number and use
  its last syllable.

Two more rules on top of that:

- **Never write the slashed hedge** `은(는)`, `이(가)`, `을(를)`, `(으)로` in
  published prose. Pick the correct one. It is a placeholder from form letters,
  not a style.
- **When in doubt, insert a generic noun and let it take the particle.** This is
  what good Korean technical writing does anyway: `RF-DETR 모델은`,
  `ONNX 형식으로`, `RTMDet 계열에서`, `CUDA를 지원하는 장치에서`. It removes the
  ambiguity and reads more naturally than a bare identifier carrying a particle.

### Punctuation

- **Korean prose uses ASCII punctuation**: `.` `,` `?` `!` `:` `;` `(` `)`.
  Never the full-width or CJK forms `．` `，` `。` `、` `（` `）` `：`. Those are
  Chinese and Japanese conventions; in Korean they break search, copy-paste and
  the validator's comparisons. The 가운뎃점 `·` is acceptable for coordinating
  nouns inside a phrase, but never inside code.
- **ASCII punctuation stays inside code, always.** Code blocks, inline code
  spans, paths, flags, identifiers, file names, URLs and version strings take
  half-width ASCII characters only: `format="tflite"`, `metrics/mAP50-95(B)`,
  `libreyolo[coreml]`, `v1.5.0`, `weights/x.pt`. A single full-width character
  here fails the validator's code comparison and, worse, breaks copy-paste for
  the reader. This applies to translated comments inside code blocks too.
- **Em dashes `—` are banned on this site.** Do not use one anywhere in a
  Korean page, including inside a translated comment. Korean has no grammatical
  need for it (unlike Chinese `——` or the Russian copula dash), so where the
  English source has an em dash, use a comma, a colon, a full stop, or
  parentheses, and split the sentence if that reads better. Do not substitute
  `~`, `―`, `--` or a spaced hyphen either.
- Keep the source's straight ASCII quotes `"..."` for scare quotes and quoted
  strings, matching the English file and the other locales. Do not switch to
  `「」` (Japanese) or full-width forms.
- Numbers, versions, units, ranges and decimal points stay exactly as in the
  source: `0.001 mAP`, `lr0=0.004`, `mAP50-95`, `300`, `t`, `x`. Never switch a
  decimal point to a comma, and never rewrite a range with `~`.
- No exclamation marks the English source does not have.

## Glossary (use these consistently)

| English | Korean |
|---|---|
| library | 라이브러리 |
| framework | 프레임워크 |
| package | 패키지 |
| repository / repo | 저장소 |
| open source | 오픈 소스 |
| computer vision | 컴퓨터 비전 |
| machine learning | 머신러닝 (one word) |
| model family | 모델 계열 |
| variant / size | 변형 / 크기 |
| weights | 가중치 |
| checkpoint | 체크포인트 |
| pretrained / pretraining | 사전 학습된 / 사전 학습 |
| dataset | 데이터셋 |
| training / to train | 학습 / 학습하다 (never 교육; 훈련 only in a quoted paper title) |
| fine-tuning / to fine-tune | 파인튜닝 / 파인튜닝하다 |
| inference / to infer | 추론 / 추론하다 |
| prediction / to predict | 예측 / 예측하다 |
| validation / to validate | 검증 / 검증하다 |
| export / to export | 내보내기 / 내보내다 |
| deployment / to deploy | 배포 / 배포하다 |
| object detection | 객체 탐지 |
| bounding box | 바운딩 박스 |
| oriented bounding box | 회전 바운딩 박스 |
| segmentation | 분할 |
| instance segmentation | 인스턴스 분할 |
| semantic segmentation | 시맨틱 분할 |
| panoptic segmentation | 판옵틱 분할 |
| classification | 분류 |
| pose estimation | 자세 추정 |
| keypoint | 키포인트 |
| depth estimation | 깊이 추정 |
| text recognition / OCR | 텍스트 인식 / OCR |
| embedding | 임베딩 |
| gaze estimation | 시선 추정 |
| edge detection (the CV task) | 엣지 검출 |
| surface normal | 표면 법선 |
| matting | 매팅 |
| restoration | 복원 |
| mesh | 메시 (never 메쉬) |
| centroid | 중심점 |
| backbone | 백본 |
| head (of a model) | 헤드 |
| neck | 넥 |
| layer | 레이어 |
| feature / feature map | 특징 / 특징 맵 |
| tensor | 텐서 |
| mask | 마스크 |
| class | 클래스 |
| label (annotation) | 레이블 (never 라벨) |
| annotation (the data) | 어노테이션 (주석 means a code comment, never annotation data) |
| ground truth | 정답, glossed once as 정답(ground truth) |
| loss / loss function | 손실 / 손실 함수 |
| threshold | 임계값 |
| confidence (score) | 신뢰도 (신뢰도 점수) |
| accuracy | 정확도 |
| precision / recall (metrics) | 정밀도 / 재현율 (do not reuse 정확도 for precision) |
| batch / batch size | 배치 / 배치 크기 (keep `batch=16` in code) |
| epoch | 에폭 |
| learning rate | 학습률 |
| warmup | 워밍업 |
| optimizer | 옵티마이저 |
| weight decay | 가중치 감쇠 |
| data augmentation | 데이터 증강 |
| mixed precision / half precision | 혼합 정밀도 / 반정밀도 |
| early stopping | 조기 종료 |
| resume (training) | 학습 재개 (`이어서 학습합니다` in prose) |
| multi-GPU | 다중 GPU |
| confusion matrix | 혼동 행렬 |
| preprocessing / postprocessing | 전처리 / 후처리 |
| quantization / to quantize | 양자화 / 양자화하다 |
| distillation / to distill | 증류 / 증류하다 (지식 증류 on first mention) |
| anchor-free | 앵커 프리(anchor-free) |
| open-vocabulary detection | 오픈 보캐뷸러리(open-vocabulary) 탐지 |
| self-supervised | 자기 지도 학습 |
| edge device / edge | 엣지 디바이스 / 엣지 |
| on-device | 온디바이스 |
| runtime | 런타임 |
| pipeline | 파이프라인 |
| latency | 지연 시간 |
| throughput | 처리량 |
| frame / FPS | 프레임 / FPS |
| real-time | 실시간 |
| tiled / sliced inference | 타일 추론 |
| streaming | 스트리밍 |
| metadata | 메타데이터 |
| flag | 플래그 |
| dependency | 의존성 |
| upstream | 업스트림 |
| benchmark | 벤치마크 |
| baseline | 베이스라인 |
| reproduce (a paper) | 재현 |
| deprecated | 지원 중단됨 |
| unmaintained | 유지보수가 중단된 |
| license | 라이선스 (never 라이센스) |
| permissive (license) | 허용적 라이선스 |
| copyleft | 카피레프트 |
| commercial use | 상업적 사용 |
| proprietary / closed-source | 독점 / 비공개 소스 |
| explainability / interpretability | 설명 가능성 / 해석 가능성 |
| codebase | 코드베이스 |

`mAP`, `mAP50`, `mAP50-95`, `AP`, `IoU`, `NMS`, `ONNX`, `TensorRT`, `CoreML`,
`OpenVINO`, `TFLite`, `GPU`, `CPU`, `NPU`, `CUDA`, `FP16`, `INT8`, `px`, `FPS`
stay untouched.

## Formatting invariants (validated by script)

Run `node scripts/translation/validate.mjs ko [section/slug ...]` before
committing. It compares the twin against its English source and fails on drift.

- Same number of headings, at the same levels, in the same order.
- Same number of fenced code blocks; code identical except comment lines.
  A stray full-width character inside a code block is the most common way a
  Korean twin fails this check.
- Same set of link targets.
- Same frontmatter keys as the English file; no keys added or dropped.
  `families`, `last_verified` and `layout` byte-identical; `hero.src` and
  `hero.poster` byte-identical.
- Same snippet structure: same group keys, same order, same count, same
  `language` values. Only `label` and in-code comments change.
- No em dash anywhere in the file.
- Do not add translator notes, disclaimers, or a "translated from" line.
