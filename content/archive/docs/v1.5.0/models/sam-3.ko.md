---
title: SAM 3
families:
  - sam3
seo_title: 'SAM 3: LibreYOLO의 프롬프트 가능 분할과 개념 분할'
description: >-
  LibreYOLO에서 SAM 3으로 점, 박스, 텍스트 개념 분할을 수행합니다. Meta의 SAM License로 제한되는 large
  체크포인트를 설치하고 예측합니다.
lead: >-
  SAM 3는 일반적인 점과 박스 프롬프트에 텍스트 개념 프롬프트를 더해 SAM을 확장합니다. 따라서 "yellow school bus" 같은
  문구로 일치하는 모든 인스턴스를 반환할 수 있습니다. LibreYOLO는 `LibreYOLO()` 탐지기 팩토리와 분리된 전용
  LibreSAM 팩토리로 이미지 경로를 지원합니다.
keywords:
  - SAM 3 사용법
  - Segment Anything
  - 프롬프트 가능 분할
  - 개념 분할
  - 텍스트 프롬프트
  - 점 프롬프트
  - 박스 프롬프트
  - Meta AI
last_verified: 1.5.0
snippets:
  predict:
    - label: 점과 박스 프롬프트
      language: python
      code: |
        from libreyolo import LibreSAM, SAMPLE_IMAGE

        # "sam3"는 유일한 크기인 "large"입니다. 별칭은 "sam3", "sam-3", "sam3-large"입니다.
        model = LibreSAM("sam3")

        # 점 프롬프트: 픽셀 좌표 [x, y], 레이블 1은 전경입니다.
        result = model.predict(SAMPLE_IMAGE, points=[640, 420], labels=[1])
        print(result.masks.xy)      # 마스크별 폴리곤
        print(result.boxes.xyxy)    # 마스크에서 파생된 꼭 맞는 박스

        # 점 대신 박스 프롬프트를 사용합니다.
        result = model.predict(SAMPLE_IMAGE, bboxes=[300, 200, 900, 700])
    - label: 텍스트(개념) 프롬프트
      language: python
      code: |
        from libreyolo import LibreSAM3, SAMPLE_IMAGE

        model = LibreSAM3("large")

        # 객체 하나만이 아니라 문구와 일치하는 모든 인스턴스를 찾습니다.
        # text=는 points, bboxes, labels, masks와 함께 사용할 수 없습니다.
        result = model.predict(SAMPLE_IMAGE, text="a person")
        print(result.names)         # {0: "a person"}
        print(result.boxes.conf)    # 인스턴스별 PCS 탐지 점수
    - label: 한 번 인코딩하고 여러 번 프롬프팅
      language: python
      code: |
        from libreyolo import LibreSAM3, SAMPLE_IMAGE

        model = LibreSAM3("large")

        # 이미지 인코더는 비용이 많이 드는 부분입니다. set_image()는 한 번 실행하고
        # 이후 모든 predict() 호출은 캐시된 임베딩을 재사용합니다. tracker와
        # 개념 분할 인코더는 캐시를 공유하지 않으므로 text= 호출은 내부에서 다시 인코딩합니다.
        model.set_image(SAMPLE_IMAGE)
        a = model.predict(points=[640, 420], labels=[1])
        b = model.predict(bboxes=[300, 200, 900, 700])
        model.reset_image()
source_hash: c4fb6d5a622f99ff
---

## 설치

SAM 3에는 `transformers`와 `timm`을 설치하는 `sam` extra가 필요합니다.

```bash
pip install "libreyolo[sam]"
```

가중치에는 접근 제한이 있습니다. [huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3)을 방문해 Meta의 SAM License에 동의한 다음 첫 다운로드 전에 `hf auth login`을 실행하거나 `HF_TOKEN`을 설정합니다. LibreYOLO는 이 계열을 처음 다운로드할 때 라이선스 고지를 기록합니다.

## 예측

`LibreSAM(...)` 또는 계열별 `LibreSAM3(...)`은 `LibreYOLO(...)`와 별도의 진입점입니다. 여기서는 프롬프트 없는 순전파에 의미가 없으므로 탐지기 대신 프롬프트 가능 분할기를 반환합니다. 이 계열에는 `libreyolo predict` CLI 명령이 없으므로 Python API를 사용합니다. 이미지 추론만 지원하며 SAM 3의 동영상 모델은 범위에 포함되지 않습니다.

<code-tabs name="predict" />

점과 박스 경로는 다른 SAM 계열과 같습니다. 점 프롬프트는 객체 하나에 `[x, y]`를, 여러 객체에 `[[x, y], ...]`를 받습니다. `labels`는 각 점을 `1`(전경) 또는 `0`(배경)으로 표시하며, 박스 프롬프트는 `[x1, y1, x2, y2]` 또는 박스 목록을 받습니다. 이 경로의 `conf`는 탐지 신뢰도가 아니라 예측된 마스크 품질(IoU)을 기준으로 필터링합니다.

`text=` 경로는 SAM 3에 추가된 기능입니다. 개념 문자열을 사용하면 Promptable Concept Segmentation으로 이미지에서 일치하는 모든 인스턴스를 반환하며 점, 박스, 레이블 또는 마스크와 함께 사용할 수 없습니다. 이 경로에서 `conf`는 마스크 IoU가 아니라 PCS 탐지 점수입니다. 기본값을 유지하면 모델 자체의 0.3 임곗값이 적용되고 `conf=0.0`은 모든 후보를 유지합니다. 프롬프트 가능 마스크에는 달리 고정된 클래스 집합이 없으므로 반환되는 `names`는 클래스 ID `0`을 요청한 개념 문자열에 매핑합니다. `device=`는 모델을 이동하며 `set_image()` 세션이 활성화된 경우 캐시된 임베딩도 이동합니다. 이 계열에서 `train()`, `val()`, `export()`, `track()`은 모두 `NotImplementedError`를 발생시킵니다. SAM 3는 LibreYOLO에서 예측 전용이며 동영상 추적은 범위에 포함되지 않습니다. 소스 유형은 [예측](/docs/predict)을 참조합니다.

## 변형

크기는 하나이며 고정된 1008 px 입력을 사용하는 large입니다. SAM 3.1은 지원하지 않습니다. 구현에 이 MIT 저장소로 벤더링할 수 없는 사용자 정의 라이선스가 적용되며, LibreYOLO가 의존하는 Transformers 버전도 아직 해당 체크포인트 형식을 불러오지 못합니다.

## 라이선스

<provenance-box>

LibreYOLO는 SAM 3 가중치 사본을 자체적으로 호스팅하거나 재배포하지 않습니다. `LibreSAM("sam3")`는 Hugging Face에서 Meta의 접근 제한 저장소인 `facebook/sam3`의 파일을 직접 다운로드합니다. 첫 다운로드 전에 Meta의 SAM License에 동의하고 인증해야 합니다.

</provenance-box>

## 인용

<citation-block />
