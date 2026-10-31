---
title: LibreFaceRec
families:
  - facerec
seo_title: 'LibreFaceRec: 얼굴 인식 및 검증'
description: >-
  LibreYOLO에서 얼굴 탐지, 임베딩, 검증에 LibreFaceRec을 사용합니다. 설치하고 예측할 수 있으며 임베딩 가중치에는
  Apache-2.0이 적용됩니다.
lead: >-
  LibreFaceRec은 LibreYOLO의 얼굴 임베딩 작업입니다. 얼굴 탐지기가 얼굴을 찾고 정렬하며, 인식 헤드는 검증 또는 검색을
  위해 L2 정규화된 신원 임베딩을 생성합니다.
keywords:
  - LibreFaceRec 사용법
  - 얼굴 인식
  - 얼굴 임베딩
  - 얼굴 검증
  - ArcFace
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # librefacerec-* 이름은 파일 접미사와 관계없이 이 계열로 라우팅하며
        # 처음 사용할 때 기본 얼굴 탐지기와 함께 LibreYOLO Hugging Face
        # 조직에서 내려받습니다.
        model = LibreYOLO("librefacerec-l.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.embeddings.data.shape)   # (N, D), L2 정규화됨
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=librefacerec-l.onnx source=face.jpg
    - label: 검증
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("librefacerec-l.onnx")

        # L2 정규화된 임베딩의 코사인 유사도로 각 이미지에서 가장
        # 두드러진 얼굴을 비교합니다.
        result = model.verify("person_a.jpg", "person_b.jpg", threshold=0.4)
        print(result["similarity"], result["same_person"])
    - label: 갤러리 검색
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("librefacerec-l.onnx")

        query = model("query.jpg").embeddings          # 이 이미지의 얼굴
        gallery = model.embed(["a.jpg", "b.jpg", "c.jpg"])   # (N_total, D)

        # (query_faces, N_total) 코사인 유사도입니다.
        scores = query.similarity(gallery)
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("librefacerec-l.onnx")
        model.export(format="onnx")
source_hash: f1a345bb96e32f12
---

## 설치

LibreFaceRec의 인식 헤드는 기본 설치에 포함되지 않는 `onnxruntime`을 사용해
실행합니다.

```bash
pip install "libreyolo[onnx]"
```

## 예측

<code-tabs name="predict" />

한 번의 호출 뒤에서 탐지와 인식이라는 두 개의 별도 ONNX 그래프가 작동합니다.
얼굴 탐지기가 각 얼굴을 찾아 표준 크롭으로 정렬하고, 인식 헤드는 얼굴마다 L2
정규화된 임베딩을 반환합니다. 그대로 사용하면 `predict()`가 번들된 기본 탐지기를
자동으로 내려받아 연결합니다. `face_detector`에는 호출 가능 객체, LibreYOLO 탐지
모델 또는 `FaceDetector` 인스턴스를 사용할 수 있으며, `face_boxes`는 이미 보유한
바운딩 박스로 탐지를 완전히 건너뜁니다. `result.embeddings`에는 탐지된 얼굴마다
`result.boxes`와 정렬된 행 하나가 들어 있습니다. `.similarity()` 메서드는 다른
임베딩 또는 전체 갤러리와의 코사인 유사도를 한 번의 호출로 계산합니다. 이미 계산된
두 임베딩 대신 두 이미지를 직접 비교하려면 `model.verify(image_a, image_b)`가 두
이미지에서 탐지와 임베딩을 실행하고 가장 신뢰도가 높은 얼굴을 비교합니다. 다른
ArcFace 규약 ONNX 인식 모델도 사용할 수 있습니다. 정렬된 크롭을 입력받고 `(N, D)`
임베딩을 출력하는 모델의 파일 경로를 `librefacerec-*` 이름 대신 전달합니다. 소스,
스트리밍, 결과 처리는 [예측](/docs/predict)을 참조합니다.

## 내보내기

<export-matrix />

LibreFaceRec은 이미 내보낸 ONNX 그래프를 래핑하므로 다른 형식으로 다시 내보내기는
구현되지 않았습니다.

## 라이선스

<provenance-box>

번들된 기본 얼굴 탐지기는 별도의 라이선스가 적용되는 두 번째 아티팩트입니다.
OpenCV Zoo의 YuNet이며 MIT, copyright Shiqi Yu입니다. 어느 프로젝트에서도
아키텍처 코드를 포팅하지 않았습니다. 두 그래프 모두 `onnxruntime`을 통해 불투명하게
사용하므로 LibreYOLO 자체 래퍼에는 타사 코드가 없으며 전체에 MIT가 적용됩니다.

</provenance-box>

## 인용

<citation-block />
