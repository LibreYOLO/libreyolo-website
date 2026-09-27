---
title: 임베딩
seo_title: LibreYOLO의 이미지 및 영역 임베딩
description: >-
  임베드 작업은 전체 이미지, 각 탐지된 영역 또는 텍스트에 대해 L2 정규화된 float32 벡터를 반환합니다. 갤러리를 등록하고, 코사인
  유사도로 매칭하며, Python 또는 CLI에서 검색하십시오.
lead: >-
  한 가지 작업이 LibreYOLO가 생성하는 모든 벡터를 포함합니다. embed는 행이 전체 이미지, 단일 탐지 얼굴 또는 텍스트 한 줄을
  설명하든 상관없이, 그 행의 내적이 유사도 점수가 되는 단위 길이 float32 행을 반환하며, 동일한 Gallery가 이들 모두와
  일치합니다.
keywords:
  - 이미지 임베딩 파이썬
  - L2 정규화 임베딩
  - 코사인 유사도 검색
  - libreyolo 임베드 작업
  - 이미지 검색
  - 갤러리 등록
  - 클립 임베딩
  - dinov2 임베딩
  - 리드 임베딩
last_verified: 1.5.0
verification: >-
  작업 키와 별칭은 libreyolo/tasks.py.에서 읽습니다. Embeddings 및 Identities 클래스에서
  libreyolo/utils/results.py. Gallery API의 결과 페이로드를 가져옵니다.
  libreyolo/utils/gallery.py.에서 embed와 _postprocess_embeddings를 가져옵니다. 지원되는 계열는
  SUPPORTED_TASKS에서 embed를 검색하여 libreyolo/models/**/model.py에서 찾습니다. CLI 인터페이스는
  libreyolo/cli/__init__.py, libreyolo/cli/commands/special.py 및
  libreyolo/cli/commands/predict.py.에서 가져옵니다. 설계 의도는
  docs/adr/0015-embed-generalization.md.에서 가져옵니다.
meta:
  - label: 작업 키
    value: embed
    mono: true
  - label: 별명
    value: 'face-recognition, reid, face'
    mono: true
  - label: 결과 페이로드
    value: 'Embeddings, Identities'
    mono: true
  - label: 행 데이터 유형
    value: 'float32, 단위 길이'
snippets:
  predict:
    - label: 전체 이미지
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # CLIP은 기본적으로 분류하도록 설정되어 있으므로, 벡터를 명시적으로 요청하십시오.
        model = LibreYOLO("LibreCLIPb32-cls.pt", task="embed")
        result = model(SAMPLE_IMAGE)

        print(result.embeddings.data.shape)  # (1, 512), 이미지당 한 행
        print(result.boxes)                  # 없음: 아무 것도 현지화되지 않음
    - label: 지역별
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("librefacerec-l.onnx")
        result = model(SAMPLE_IMAGE)

        # i번째 행은 i번째 상자에 있는 영역을 설명합니다.
        print(result.boxes.xyxy.shape)       # (N, 4)
        print(result.embeddings.data.shape)  # (N, 512)
    - label: 한 번에 많은 이미지
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="embed")

        # 모든 결과의 모든 행을 하나의 텐서로 연결한 것.
        vectors = model.embed(["a.jpg", "b.jpg", "c.jpg"])
        print(vectors.shape)  # (3, 384)
    - label: 텍스트
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreCLIPb32-cls.pt", task="embed")

        # 텍스트는 방법일 뿐, 예측의 출처는 아닙니다. 문자열이 전달된
        # model(...)은 여전히 경로나 URL입니다.
        text = model.embed_text(["a photo of a cat", "a photo of a dog"])
        print(text.shape)  # (2, 512)
  similarity:
    - label: 두 개의 행 집합을 비교하다
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreCLIPb32-cls.pt", task="embed")

        query = model.embed("query.jpg")          # (1, 512)
        pool = model.embed(["a.jpg", "b.jpg"])    # (2, 512)

        # 행(row)의 길이는 1이므로 코사인 유사도는 내적(dot product)입니다.
        scores = model("query.jpg").embeddings.similarity(pool)
        print(scores.shape)  # (1, 2)
    - label: 텍스트에 대한 이미지
      language: python
      code: |
        import torch

        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreCLIPb32-cls.pt", task="embed")

        image = model.embed("photo.jpg")                       # (1, 512)
        text = model.embed_text(["a cat", "a dog", "a car"])   # (3, 512)

        print(torch.matmul(image, text.T))
  gallery:
    - label: 등록하고 식별하다
      language: python
      code: |
        from libreyolo import Gallery, LibreYOLO

        model = LibreYOLO("librefacerec-l.onnx")

        gallery = Gallery(model)
        gallery.enroll("ada", ["people/ada/1.jpg", "people/ada/2.jpg"])
        gallery.enroll("grace", "people/grace/1.jpg")
        gallery.save("refs.npz")

        result = model("group.jpg", gallery=gallery, threshold=0.4)
        for name, score in result.identities.data:
            print(name, score)   # 이름이 기준치 아래입니다
    - label: 상위 k 검색
      language: python
      code: |
        from libreyolo import Gallery
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="embed")
        gallery = Gallery.load("refs.npz", model=model)

        result = model("query.jpg")
        matches = gallery.match(result.embeddings, top_k=5, threshold=0.4)
        print(matches[0])   # 첫 번째 행에 대한 [(이름, 점수), ...]
    - label: 이미 가지고 있는 벡터를 등록하십시오
      language: python
      code: |
        from libreyolo import Gallery

        gallery = Gallery()
        gallery.enroll_embedding("ada", vector)  # 들어오는 길에 정규화됨
        print(gallery.identities, gallery.dim, len(gallery))
  cli:
    - label: 폴더 트리 등록
      language: bash
      code: >
        # source/<identity>/*.jpg. 기존 갤러리가 제자리에서 확장됩니다.

        libreyolo enroll model=librefacerec-l.onnx source=people/
        gallery=refs.npz
    - label: 예측하는 동안 식별
      language: bash
      code: |
        libreyolo predict model=librefacerec-l.onnx source=group.jpg \
          gallery=refs.npz gallery_threshold=0.45
    - label: 두 이미지를 비교하다
      language: bash
      code: >
        libreyolo compare model=librefacerec-l.onnx \
          source=a.jpg source2=b.jpg threshold=0.4

        # verify는 다른 이름으로 된 동일한 명령입니다.

        libreyolo verify model=librefacerec-l.onnx source=a.jpg source2=b.jpg
        --json
source_hash: ffbaad5599035bc7
---

## 정의

`embed`는 이미지, 이미지의 일부 영역 또는 문자열을 길이가 1인 고정 폭의 float32 행으로 변환합니다. 모든 행이 단위 벡터이기 때문에 두 행을 비교하는 것은 내적이고, 두 세트를 비교하는 것은 단일 행렬 곱입니다. 작업의 다른 부분은 모델에 특화된 것이 없습니다: 검색, 중복 탐지, 재식별 및 얼굴 인식은 모두 서로 다른 행 위에서 동일한 연산입니다.

벡터가 출력입니다. 클래스 목록이 없으므로, 이름은 네트워크가 예측하도록 학습된 것이 아니라 사용자가 제공한 참조와 비교하여 나중에 붙여집니다.

### 세 가지 모양

| 모양 | `Results.embeddings` | `Results.boxes` | 제작 |
|---|---|---|---|
| 전체 이미지 | `(1, D)` | `None` | 전체 이미지 계열에 이미지를 전달하기 |
| 지역 | `(N, D)` | `(N, 4)`, 행 정렬 | 얼굴 인식과 같이 먼저 현지화하는 계열들 |
| 텍스트 | 전혀 `Results`가 아니다 | | `model.embed_text(texts)`, `(M, D)` 반환 |

전체 이미지 결과는 한 이미지의 경우에도 2차원으로 유지됩니다. `(D,)`는 허용된 반환 형식이 아니므로, 사용자는 단일 행 경우를 특별히 처리할 필요가 없습니다. Text는 `Results` 대신 일반 텐서를 반환합니다. 문자열은 이미지 소스가 아니기 때문입니다. 문자열을 `model(...)`에 전달해도 여전히 경로나 URL을 의미하며, 라이브러리는 문자열이 글이라는 것을 추측하지 않습니다.

정규 작업 키는 `embed`입니다. `embedding`, `embeddings`, `face-recognition`, `facial-recognition`, `recognition`, `face`, `faceid`, `reid` 모두 이 키로 정규화되므로 `task="reid"`와 `task="embed"`는 정확히 동일한 항목을 선택합니다.

## 모델들

네 가문이 그 임무를 수행하며, 그들은 먼저 무엇이든 현지화하는지 여부에 따라 깔끔하게 나뉩니다.

| 계열 | 모양 | 차원 | 또한 지원 |
|---|---|---|---|
| [LibreFaceRec](/docs/models/librefacerec) | 영역, 탐지된 얼굴마다 한 줄 | 512 | 없음; `embed`가 유일한 임무입니다 |
| [클립](/docs/models/clip) | 전체 이미지, 짝지어진 텍스트 타워와 함께 | `b32` 및 `b16`에는 512, `l14`에는 768 | `classify`, 기본값을 유지합니다 |
| [SigLIP 2](/docs/models/siglip2) | 전체 이미지, 짝지어진 텍스트 타워와 함께 | `b16`는 768, `so400m`는 1152 | `classify`, 기본값을 유지합니다 |
| [DINOv2](/docs/models/dinov2) | 전체 이미지, 이미지 만 | 384 | `semantic`, `classify` |

CLIP과 SigLIP 2는 `classify`를 기본 작업으로 유지하므로 `task="embed"`는 요청해야 합니다. 기존의 `-cls` 체크포인트는 공유된 투타워 아티팩트이며, 동일한 가중치에 대한 복제된 `-embed` 체크포인트는 공개되지 않습니다.

`embed_text`는 텍스트 타워가 있는 두 가지 계열인 CLIP과 SigLIP 2에서만 존재합니다. DINOv2에는 없습니다. DINOv2 임베딩은 의미 및 분류 헤드를 우회하며 224픽셀에서 최종 정규화된 CLS 토큰을 읽습니다; `n`, `s`, `m`, `l` 변형은 모두 DINOv2-S 인코더를 공유하므로 네 가지 모두 `D = 384`를 반환합니다.

이번 릴리스에 추가된 분류 전용 백본인 [ViT](/docs/models/vit), [Swin](/docs/models/swin) 및 [DeiT](/docs/models/deit)는 `classify`만 선언하며 이 작업을 수행하지 않습니다.

<code-tabs name="predict" />

`model.embed(source, **kwargs)`는 배치 단축키입니다: 이는 `predict`를 실행하고 모든 결과의 각 행을 하나의 `(N_total, D)` CPU float32 텐서로 연결하며, 행의 차원이 혼합되어 있으면 오류를 발생시킵니다. 지원되는 작업에 `embed`가 없는 계열는 `NotImplementedError`를 발생시킵니다.

## 결과 페이로드

`result.embeddings`는 `Embeddings` 페이로드입니다. 그 `data`는 항상 `(N, D)` float32이며, 추론 경로에 의해 이미 L2 정규화되어 있고, 2차원이 아닌 입력은 조용히 재형성되는 대신 오류를 발생시킵니다.

| 회원 | 의미 |
|---|---|
| `.data` | `(N, D)` 행렬 |
| `.dim` | `D` |
| `.normalized` | 같은 행들, 방어적으로 재정규화됨 |
| `.similarity(other)` | `(N, M)`를 다른 세트와 비교하거나, `(N,)`를 단일 `(D,)` 벡터와 비교 |
| `.verify(i, j, threshold=0.4)` | 행 `i`와 `j`가 같은 대상인지 여부 |

`result.identities`는 갤러리가 전달된 경우에만 존재하는 `Identities` 페이로드입니다. 이는 텐서가 아닌 일반 컨테이너이므로, `Results`를 장치 간에 이동해도 그대로 유지됩니다.

| 회원 | 의미 |
|---|---|
| `.name` | 이름 목록, `None`에서 아무 것도 기준을 통과하지 못함 |
| `.score` | `(N,)` float32 최고 코사인 점수, 이름이 `None`일 때도 유지됨 |
| `.data` | `(name, score)` 튜플 목록 |

<code-tabs name="similarity" />

벡터는 기본적으로 `summary()`와 `to_json()`에서 제외됩니다. 한 행이 512-float이기 때문에 피험자당 약 2킬로바이트이기 때문입니다. 대신 각 행은 `embedding_dim`를 보고하며, 갤러리가 사용된 경우 `identity`와 `identity_score`도 포함됩니다. 숫자를 포함하려면 `summary(embeddings=True)`를 전달하십시오.

## 갤러리

`Gallery`는 이름이 지정된 참조 행 집합입니다. 각 참조를 개별적으로 저장하므로 이름은 단일 최적 일치 참조로 점수가 매겨지며, 나쁜 사진을 추가해도 동일성의 중심이 이동하지 않습니다.

<code-tabs name="gallery" />

`Gallery(model)`는 자신의 벡터를 생성할 가중치에 결합됩니다. `enroll(name, sources, select="best")`는 각 소스에서 예측을 수행하고 결과별로 가장 신뢰도가 높은 행을 유지합니다; `select="all"`는 대신 모든 행을 유지하는데, 참조 이미지에 여러 피사체가 실제로 포함된 경우 원하는 동작입니다. `enroll_embedding(name, vector)`는 추론을 건너뛰고 벡터를 직접 가져와 정규화하며, 모든 값이 0인 행은 거부합니다.

`FaceGallery`는 동일한 클래스의 영구 별칭이며, 이전 얼굴 전용 릴리스에서 작성된 아카이브도 여전히 로드됩니다.

### 매칭 및 임계값

매칭은 모든 저장된 참조에 대해 조밀한 행렬 곱셈을 수행하며, 각 이름에 대해 최대값을 취해 하나의 점수로 축소됩니다. 근사 인덱스가 없기 때문에 숫자가 정확하게 유지되며 갤러리 크기에 실질적인 한계를 둡니다.

두 개의 입력 지점은 임계값 이하에서 수행하는 작업이 다릅니다. `match()`는 임계값 이하의 모든 것을 제거하고 행별로 `[(name, score), ...]`를 반환하므로, 일치하는 항목이 없는 행은 빈 목록이 됩니다. `identify()`는 항상 최고 점수를 유지하고 임계값 이하일 때 이름을 `None`로 설정하는 `Identities` 페이로드를 반환합니다. 둘 다 임계값 이하의 가장 가까운 이름을 대체하지는 않습니다.

기본 임계값은 전반적으로 `0.4`입니다. 이것은 확률이 아니라 코사인 값이며, 올바른 작동 지점은 데이터와 잘못된 일치에 대한 허용치의 속성이므로 기본값을 그대로 사용하지 말고 레이블이 지정된 쌍에서 조정하십시오. `libreyolo enroll`와 `gallery=` 예측 인자는 동일한 숫자를 사용합니다.

### 끈기

`save(path)`는 벡터, 이름 및 형식 버전, 임베딩 차원, 행을 생성한 가중치의 지문을 포함하는 메타데이터 블록을 보유한 압축된 `.npz`를 작성합니다. `Gallery.load(path, model=...)`는 비교하기 전에 둘 다를 확인하므로, 갤러리를 다른 모델을 가리키도록 지정하면 두 개의 관련 없는 공간의 벡터를 서로 점수매기는 대신 오류가 발생합니다. 빈 갤러리를 저장하는 것은 거부됩니다.

## 명령 줄

| 명령 | 목적 |
|---|---|
| `libreyolo enroll` | 폴더별 아이덴티티 트리를 탐색하고 `.npz` 갤러리를 작성하거나 확장합니다 |
| `libreyolo compare` | 기본 주제를 두 이미지에 삽입하고 코사인 유사도를 보고하십시오 |
| `libreyolo verify` | 두 번째 이름으로 같은 명령 |
| `libreyolo predict gallery=...` | 일반 예측 실행에 식별자를 연결합니다 |

<code-tabs name="cli" />

모든 LibreYOLO 명령어는 `key=value`와 `--key value`를 모두 허용하므로 `gallery=refs.npz`와 `--gallery refs.npz`는 동일한 인수입니다.

`enroll`는 `model`, `source` 및 `gallery`를 사용하며, 선택적으로 `face-detector`, `device`, `--json` 및 `--quiet`도 사용할 수 있습니다. 각 아이덴티티마다 하나의 폴더를 읽으며, 폴더 이름이 아이덴티티가 되고 그 안의 모든 이미지는 참조로 기여합니다:

```text
people/
  ada/
    1.jpg
    2.jpg
  grace/
    1.jpg
```

아무 것도 생성하지 않는 이미지는 실행을 중단하지 않고 stderr에 한 줄로 건너뛰며, 요약에는 각 이름마다 몇 개의 참조가 저장되었는지 보고합니다. 기존 갤러리 파일은 제자리에서 확장되므로 시간이 지나면서 신원을 추가할 수 있습니다.

`compare`와 `verify`는 동일한 기능이 두 번 등록된 것입니다. 이들은 `model`, `source`, `source2`와 선택적인 `threshold`를 받아 코사인 유사도, 동일 또는 다른 판정, 그리고 이를 산출한 임계값을 출력합니다. `--json`는 동일한 세 가지 필드를 객체로 출력합니다.

`predict`에서, `gallery`는 저장된 `.npz`를 가리키고 `gallery_threshold`는 `0.4` 기본값을 재정의합니다. 작업이 `embed`이 아닌 모델에 갤러리를 전달하는 것은 무시되는 동작이 아니라 오류이며, 갤러리 파일이 없으면 이를 생성할 `libreyolo enroll` 명령을 나타냅니다.

## 얼굴들

얼굴 인식은 이 작업의 영역 형태이며, 그 형태의 유일한 제공 구현입니다. 임베딩 헤드 앞에 탐지 및 정렬 단계를 추가하고, `verify()` 방법, 사용자가 직접 박스를 제공하는 인수, 공개된 정확도 수치 및 임계값에 대한 보정 지침을 포함합니다. 이 모든 것은 [얼굴 인식](/docs/tasks/face-recognition)에서 제공되며, 피사체가 얼굴일 때 따라야 할 안내입니다. 이 페이지의 모든 내용은 변경 없이 적용됩니다.

## 학습, 검증 및 내보내기

이 작업에서는 LibreYOLO 내부에서 학습되는 것은 없습니다. 얼굴 임베딩 헤드는 ONNX 산물로, `train()`, `val()`, `export()`가 모두 오류를 발생시킵니다; 헤드를 업스트림에서 학습시키고 경로를 통해 파일을 불러오십시오. CLIP, SigLIP 2, DINOv2는 `embed`를 통해서가 아니라 분류 및 세분화 작업을 통해 학습하고 내보냅니다.

검색 검증기가 없습니다. 레이블이 지정된 쌍에서 `threshold`를 스윕하여 검증 정확도를 측정하고, 갤러리를 등록하고 `identities.name`와 `identities.score`를 보류된 이미지에서 읽어 `None` 이름을 거부로 계산하여 식별 정확도를 측정합니다.
