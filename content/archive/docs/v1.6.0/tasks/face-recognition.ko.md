---
title: 얼굴 인식
seo_title: LibreYOLO에서 얼굴 인식
description: >-
  LibreYOLO에서 얼굴을 탐지, 임베딩 및 식별합니다. 갤러리를 등록하고, 두 이미지를 비교하며, Python 또는 CLI에서 코사인
  유사도로 매칭합니다.
lead: >-
  얼굴 인식은 얼굴에 적용되는 내장 작업입니다. 검출기는 각 얼굴을 찾고 정렬하며, 인식 헤드는 각 얼굴에 대해 L2 정규화된 벡터를
  반환하고, 신원은 고정된 클래스 목록이 아니라 등록된 참조와의 코사인 유사도에 따라 결정됩니다.
keywords:
  - 얼굴 인식 파이썬
  - 얼굴 임베딩
  - 얼굴 인증
  - 얼굴 갤러리
  - 아크페이스 온NX
  - libreyolo 임베드 작업
  - 코사인 유사도 얼굴
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # librefacerec-* 이름은 어쨌든 얼굴 임베딩 계열로 경로를 지정합니다
        # 파일 접미사 및 LibreYOLO Hugging Face 조직에서 다운로드
        # 기본 얼굴 탐지기와 함께 처음 사용
        model = LibreYOLO("librefacerec-l.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)             # (N, 4) 얼굴 상자
        print(result.embeddings.data.shape)  # (N, D), 얼굴당 한 행
        print(result.embeddings.dim)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=librefacerec-l.onnx source=photo.jpg
    - label: 두 이미지를 비교하다
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("librefacerec-l.onnx")

        # 두 이미지에서 탐지 및 임베딩을 실행하고 이를 비교합니다
        # 가장 자신 있는 얼굴. 코사인 유사도는 [-1, 1] 범위에 있습니다.
        outcome = model.verify("person_a.jpg", "person_b.jpg", threshold=0.4)
        print(outcome["similarity"], outcome["same_person"])
    - label: 갤러리를 등록하고 식별하십시오
      language: python
      code: |
        from libreyolo import Gallery, LibreYOLO

        model = LibreYOLO("librefacerec-l.onnx")

        gallery = Gallery(model)
        gallery.enroll("ada", ["people/ada/1.jpg", "people/ada/2.jpg"])
        gallery.enroll("grace", "people/grace/1.jpg")
        gallery.save("faces.npz")

        result = model("group_photo.jpg", gallery=gallery, threshold=0.4)
        for name, score in result.identities.data:
            print(name, score)   # 이름이 기준치 아래입니다
    - label: CLI에서 등록하고 식별하기
      language: bash
      code: >
        libreyolo enroll model=librefacerec-l.onnx source=people/
        gallery=faces.npz

        libreyolo predict model=librefacerec-l.onnx source=group_photo.jpg
        gallery=faces.npz
    - label: 자신의 얼굴 상자를 가져오십시오
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("librefacerec-l.onnx")

        # face_boxes는 탐지를 완전히 건너뜁니다; face_detector는 수락합니다
        # 호출 가능한 객체, LibreYOLO 탐지 모델, 또는 FaceDetector 인스턴스.
        result = model(SAMPLE_IMAGE, face_boxes=[[34, 12, 90, 80]])
        print(result.embeddings.data.shape)
source_hash: d7dfcb6f812ebb2d
---

## 정의

얼굴 인식은 레이블이 아니라 얼굴당 벡터를 반환합니다. 예측은 두 단계로 이루어집니다: 얼굴 탐지기가 각 얼굴과 다섯 개의 랜드마크를 찾고, 자른 이미지는 정규 112x112 정렬로 변형되며, 인식 헤드는 L2 정규화된 임베딩을 출력합니다.

`result.embeddings`는 `Embeddings` 모양의 `(N, D)` 페이로드로, `result.boxes`와 행 정렬되어 있어, 행 `i`는 `i`의 박스 안에 있는 얼굴을 설명합니다. 행들이 단위 벡터이기 때문에, 코사인 유사도는 내적이며, `embeddings.similarity()`는 다른 `Embeddings`에 대해 또는 전체 행렬에 대해 한 번에 계산합니다.

얼굴에 이름을 붙이는 것은 별도의 단계입니다. `Gallery`는 이름이 지정된 참조 벡터를 보관합니다; `gallery=`를 `predict()`에 전달하면 `result.identities`가 첨부되며, 임베딩과 행이 정렬되고 얼굴마다 이름과 최상의 코사인 점수를 포함합니다. 일치 임계값 이하의 얼굴은 이름으로 `None`를 유지하며, 임계값 이하의 가장 가까운 이름은 절대 대체되지 않습니다.

라이브러리의 표준 작업 키는 `embed`입니다. `face-recognition`, `facial-recognition`, `reid` 및 `face`는 모두 이것으로 정규화되므로 `task="face-recognition"`와 `task="embed"`도 동일한 것을 선택합니다. Faces는 그 더 넓은 작업의 영역 모양이며; [embeddings](/docs/tasks/embeddings)는 전체 이미지 및 텍스트 모양, 공유된 `Embeddings`, `Identities` 및 `Gallery` API, 그리고 어떤 것도 탐지하지 않고 벡터를 생성하는 모델들을 다룹니다.

## 모델들

[LibreFaceRec](/docs/models/librefacerec)은 이 작업을 위한 계열입니다. 이것은 두 개의 ONNX 아티팩트를 한 번의 호출로 연결한 것입니다: `librefacerec-l.onnx`, 512-d 임베딩을 생성하는 iResNet100 인식 헤드, 그리고 `librefacerec-det.onnx`, OpenCV 마스코트에서 가져온 기본 얼굴 탐지기(5개 랜드마크 포함)입니다. 둘 다 첫 사용 시 LibreYOLO Hugging Face 조직에서 다운로드됩니다. 다른 ArcFace 규약 ONNX 파일(정렬된 112x112 입력, `(N, D)` 출력)은 `librefacerec-*` 이름 대신 해당 경로를 전달하여 인식 헤드를 교체할 수 있습니다.

`embed` 작업 키는 얼굴보다 넓습니다. [CLIP](/docs/models/clip), [SigLIP2](/docs/models/siglip2) 및 [DINOv2](/docs/models/dinov2)도 `task="embed"`를 지원하며 전체 이미지 벡터 하나를 반환하는데, 이는 얼굴 식별보다는 이미지 검색입니다. 이들은 `Gallery` 및 `Embeddings` API를 공유하므로 아래의 등록 및 매칭 워크플로가 동일하게 적용되지만, 얼굴을 탐지하거나 정렬하지는 않습니다.

인식 헤드는 `onnxruntime`를 통해 실행되며, 기본 설치에는 포함되어 있지 않습니다:

```bash
pip install "libreyolo[onnx]"
```

## 예측

<code-tabs name="predict" />

혼자 두면 `predict()`가 기본 탐지기를 다운로드하고 연결합니다. `face_detector`는 이를 호출 가능한 객체, LibreYOLO 탐지 모델 또는 `FaceDetector` 인스턴스로 대체할 수 있으며, 생성자에서 설정하거나 호출 시마다 설정할 수 있습니다. `face_boxes`는 이미 가지고 있는 박스로 탐지를 우회합니다. CLI에서 `face_detector=`는 얼굴 탐지기 `.onnx` 경로나 LibreYOLO 탐지기 이름을 받을 수 있습니다.

`model.verify(image_a, image_b)`는 두 이미지 단축키입니다: 각각의 이미지에서 가장 확신 있는 얼굴을 삽입하고 `{"similarity", "same_person", "threshold"}`를 반환합니다. `model.embed(sources)`는 하나 이상의 이미지에 걸쳐 모든 얼굴 행을 단일 `(N_total, D)` 텐서로 쌓아 반환합니다. 소스, 스트리밍 및 결과 처리는 [prediction](/docs/predict)를 참조하십시오.

## 데이터셋 형식

등록은 각 개별 사용자의 아이덴티티마다 폴더를 읽습니다. 폴더 이름이 아이덴티티가 되며, 그 안의 모든 이미지는 해당 이름에 대한 참조로 사용됩니다:

```text
people/
  ada/
    1.jpg
    2.jpg
  grace/
    1.jpg
```

`libreyolo enroll`는 그 나무를 걷고 `.npz` 갤러리를 작성합니다. 기존 갤러리 파일은 교체되는 대신 제자리에서 확장되므로 시간이 지남에 따라 신원을 추가할 수 있습니다. 갤러리는 임베딩 차원과 파일 지문으로 생성된 가중치에 바인딩되며, 다른 모델과의 매칭은 호환되지 않는 벡터 공간을 비교하는 대신 오류를 발생시킵니다.

기본적으로 각 소스 이미지는 가장 자신 있는 얼굴 한 개를 참조 행으로 기여하므로, 구경꾼이 포함된 초상화는 오직 그 사람만 등록됩니다. 반환된 모든 행을 저장하려면 `select="all"`에서 `Gallery.enroll`로 전달하십시오.

## 학습

이 작업에서는 어떤 계열도 LibreYOLO 안에서 학습하지 않습니다. `LibreFaceEmbedder.train()`는 다음과 같이 제안합니다: 인식 헤드를 업스트림에서 학습시키고, ArcFace 규약으로 ONNX로 내보내고, 경로를 통해 파일을 로드합니다.

## 검증

이 작업에는 데이터셋 검증기가 없으며, `val()`는 그렇지 않은 척하지 않고 직접 오류를 발생시킵니다. 검증 정확도는 `model.verify()`가 붙은 이미지 쌍에서 측정되며, 원하는 작동점을 선택하기 위해 `threshold`를 조정합니다. 식별 정확도는 갤러리를 등록하고, 보류된 이미지에서 `result.identities.name`와 `result.identities.score`를 읽은 후, `None` 이름을 거부로 계산하여 측정됩니다.

## 내보내기

인식 헤드는 이미 ONNX 그래프이므로 변환할 것이 없습니다: `LibreFaceEmbedder.export()`가 표시됩니다. `.onnx` 파일을 직접 배포하거나 LibreYOLO를 해당 파일로 지정하고 계열가 탐지, 정렬 및 정규화를 처리하도록 하십시오.
