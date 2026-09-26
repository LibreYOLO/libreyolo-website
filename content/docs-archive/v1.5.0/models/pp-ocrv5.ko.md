---
title: PP-OCRv5
families:
  - ppocr
seo_title: 'PP-OCRv5: LibreYOLO의 텍스트 탐지 및 인식'
description: >-
  LibreYOLO에서 다국어 장면 텍스트 OCR에 PP-OCRv5를 사용합니다. Apache-2.0이 적용된 t 및 l 체크포인트를 설치하고
  예측, 검증합니다.
lead: >-
  PP-OCRv5는 PaddleOCR의 텍스트 탐지 및 인식 파이프라인입니다. 미분 가능한 이진화 탐지기가 텍스트 사각형을 찾고
  SVTR/CTC 인식기가 읽습니다. LibreYOLO는 두 계층을 PyTorch로 포팅합니다.
keywords:
  - PP-OCRv5 사용법
  - PaddleOCR
  - OCR
  - 텍스트 탐지
  - 텍스트 인식
  - 장면 텍스트
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibrePPOCRl-ocr.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for text, conf in zip(result.ocr.texts, result.ocr.conf):
            print(text, float(conf))
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibrePPOCRl-ocr.pt source=receipt.jpg save=True
    - label: 사각형
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibrePPOCRl-ocr.pt")
        result = model(SAMPLE_IMAGE)

        # 읽기 순서의 (N, 4, 2) 폴리곤: 왼쪽 위, 오른쪽 위,
        # 오른쪽 아래, 왼쪽 아래입니다. 탐지 사각형은 실제 폴리곤
        # (회전 텍스트)이므로 result.boxes가 아닌 result.ocr에 채워집니다.
        print(result.ocr.data.shape)
        print(result.ocr.det_conf)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePPOCRl-ocr.pt")
        metrics = model.val(data="my-dataset")

        print(metrics["metrics/det_hmean"])
        print(metrics["metrics/e2e_f1"])       # 주요 메트릭
        print(metrics["metrics/rec_1-NED"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibrePPOCRl-ocr.pt data=my-dataset
source_hash: 9835057f8bd95bc1
---

## 설치

PP-OCRv5에는 기본 패키지 외에 추가 항목이 필요하지 않습니다.

```bash
pip install libreyolo
```

## 예측

가중치는 처음 사용할 때 Hugging Face에서 내려받아 로컬에 캐시됩니다.

<code-tabs name="predict" />

각 체크포인트는 탐지와 인식 두 단계를 하나의 `.pt` 파일에 번들하며, 인식 문자
집합과 파이프라인 기본값은 체크포인트 메타데이터에 담깁니다. 인식기는 하나의
딕셔너리로 중국어 간체 및 번체, 영어, 일본어, 병음을 읽습니다. `result.ocr`은
`OCRRegions` 페이로드입니다. `.data`에는 4점 폴리곤, `.texts`에는 인식문,
`.conf`에는 영역별 인식 점수, `.det_conf`에는 탐지 점수가 들어 있습니다. 다중
이미지 소스는 순차적으로 실행합니다. 2단계 파이프라인은 이미지 간 배치를 수행하지
않습니다. 소스, 스트리밍, 결과 처리는 [예측](/docs/predict)을 참조합니다.

## 변형

계층은 두 가지입니다. `t`는 CPU 사용을 위해 더 가벼운
PP-LCNetV3/PP-OCRv5_mobile 백본으로 구성되며, `l`은 더 높은 정확도를 위해
PP-HGNetV2 서버 백본으로 구성됩니다. 두 계층 모두 고정된 긴 변 제한으로 탐지를
실행하고 크롭을 배치로 인식합니다. `rec_batch`는 순전파마다 인식기를 통과하는 크롭
수를 제어합니다.

## 검증

`val()`은 이미지 디렉터리와 `labels/<split>.jsonl` 파일 또는 동등한 데이터셋
YAML을 기준으로 파이프라인을 측정합니다. 각 레이블에는 이미지별 텍스트 영역 폴리곤과
인식문이 나열됩니다. 탐지 hmean(IoU로 일치시킨 정밀도, 재현율, F1), 엔드투엔드
F1(hmean과 정규화 후 정확한 인식문 일치이며 체크포인트의 적합도 메트릭), 일치한 쌍의
평균 정규화 편집 거리인 1-NED를 보고합니다.

<code-tabs name="val" />

## 내보내기

<export-matrix />

PP-OCRv5는 하나의 추적 가능한 그래프가 아니라 탐지와 인식이 함께 이동하는 2개
네트워크 파이프라인이므로 내보내기가 구현되지 않았습니다. 아직 지원하는 형식이
없습니다. 이 형식 외의 체크포인트가 필요하면 Apache-2.0 업스트림 학습 코드를 직접
파인튜닝하고 `weights/convert_ppocr_weights.py`로 결과를 변환합니다.

## 체크포인트

이 계열에서 공개된 모든 가중치 파일입니다.

<checkpoint-table />

## 라이선스

<provenance-box></provenance-box>

## 인용

<citation-block />
