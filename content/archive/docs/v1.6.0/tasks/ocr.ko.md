---
title: 광학 문자 인식
seo_title: 'OCR: LibreYOLO에서의 텍스트 탐지 및 인식'
description: >-
  LibreYOLO로 이미지에서 텍스트를 찾고 읽으십시오. 쿼드와 전사(transcripts)를 예측하고, JSONL 데이터셋에 레이블을
  지정하며, hmean, end-to-end F1, 1-NED로 검증하십시오.
lead: >-
  OCR은 이미지에서 텍스트를 찾아 읽습니다. LibreYOLO는 이를 ocr 작업으로 노출하며, 텍스트 영역마다 읽기 순서대로 한 개의 4점
  폴리곤과 한 개의 문자열을 반환합니다.
keywords:
  - OCR 파이썬 라이브러리
  - 장면 텍스트 인식
  - 텍스트 탐지 쿼드
  - PP-OCRv5 파이썬
  - 끝에서 끝까지 텍스트 탐지
last_verified: 1.5.0
snippets:
  predict:
    - label: 이미지의 텍스트를 읽으십시오
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # T 등급은 두 가지 중 더 가벼운 것으로, CPU용으로 제작되었습니다. SAMPLE_IMAGE
        # 이것을 실행 가능하게 유지합니다; 자신의 텍스트가 있는 이미지로 가리키십시오.
        model = LibreYOLO("LibrePPOCRt-ocr.pt")
        result = model(SAMPLE_IMAGE)

        regions = result.ocr
        print(len(regions), "regions")
        for text, score in zip(regions.texts, regions.conf):
            print(repr(text), float(score))
    - label: 사두근을 읽으십시오
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibrePPOCRt-ocr.pt")
        result = model(SAMPLE_IMAGE)

        regions = result.ocr
        print(regions.data.shape)   # (N, 4, 2) 다각형, TL TR BR BL
        print(regions.xyxy)         # 그 다각형들의 축 정렬 껍질
        print(regions.det_conf)     # 탐지 점수, .conf와 별도
    - label: 인식 신뢰도로 필터링
      language: python
      code: |
        import numpy as np
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibrePPOCRt-ocr.pt")
        result = model(SAMPLE_IMAGE)

        # 위치로 인덱싱하며, 불리언 마스크는 아님: 슬라이싱이 이어진다
        # 기록과 두 점수 배열, 그리고 기하 정보를 함께.
        regions = result.ocr.numpy()
        keep = regions[np.flatnonzero(regions.conf >= 0.9)]
        print(keep.texts)
  val:
    - label: 메트릭 키를 검증하고 읽습니다
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePPOCRt-ocr.pt")
        metrics = model.val(data="my-ocr-dataset")

        print(metrics["metrics/det_precision"], metrics["metrics/det_recall"])
        print(metrics["metrics/det_hmean"])
        print(metrics["metrics/e2e_f1"])       # 체력
        print(metrics["metrics/rec_1-NED"])
source_hash: 58ad5305c9dd458c
---

## 정의

`ocr` 작업은 한 번의 호출로 두 가지 일을 수행합니다: 이미지 내 모든 텍스트 영역을 찾고 이를 전사합니다. 영역은 축에 맞춘 사각형이 아니라 네 점 폴리곤으로 반환되는데, 이는 장면 텍스트가 자주 회전되어 있으며, 읽기 순서가 위에서 아래로, 그 다음 왼쪽에서 오른쪽이기 때문입니다.

예측은 `result.ocr`, `OCRRegions` 페이로드를 채웁니다. `.data`는 원본 이미지 픽셀의 다각형으로 이루어진 `(N, 4, 2)` 부동 소수 배열이며, 순서는 좌상단, 우상단, 우하단, 좌하단입니다; `.texts`는 N개의 대본 목록입니다; `.conf`는 영역별 인식 점수이고 `.det_conf`는 탐지 점수입니다; `.xyxy`는 각 다각형의 축에 맞춰진 외형을 제공합니다. 쿼드는 진짜 다각형이기 때문에 `result.boxes`를 채우지 않습니다. `OCRRegions`를 슬라이스하면 대본과 두 점수 배열, 그리고 기하 정보가 함께 전달됩니다.

## 모델들

두 계열이 `ocr`를 제공합니다.

[PP-OCRv5](/docs/models/pp-ocrv5)는 전용 파이프라인입니다: 미분 가능 이진화 탐지기가 텍스트 사각형을 찾고 SVTR/CTC 인식기가 이를 읽으며, 두 단계 모두 인식 문자 집합과 함께 하나의 `.pt` 파일로 번들로 제공됩니다. 이 파이프라인은 두 가지 버전으로 제공되며, 하나는 CPU용 가벼운 버전이고, 다른 하나는 더 높은 정확도를 위한 서버용 버전입니다. 하나의 사전에는 간체 및 번체 중국어, 영어, 일본어, 병음이 포함되어 있습니다.

[SenseNova-Vision](/docs/models/sensenova-vision)은 여섯 가지 다른 작업을 수행하는 동일한 7B 체크포인트에서 태그된 텍스트로 단어를 생성함으로써 OCR에 도달하며, `LibreVLM("sensenova-vision", task="ocr")`로 로드됩니다. `sensenova` 추가가 필요하며, 그 가중치는 비상업적 사용으로 제한되며 라이선스는 해당 페이지에 있습니다.

## 예측

가중치는 처음 사용할 때 Hugging Face에서 다운로드되며 로컬에 캐시됩니다.

<code-tabs name="predict" />

PP-OCRv5는 고정된 긴 변 한계에서 검출을 수행한 후, 잘라낸 영역을 배치 단위로 인식하며, `rec_batch`는 한 포워드 패스당 얼마나 많은 크롭이 인식기로 들어가는지를 제어합니다. 다중 이미지 소스는 순차적으로 실행되는데, 두 단계 파이프라인은 이미지 간에 배치를 처리하지 않기 때문입니다. 소스, 스트리밍 및 결과 처리에 대해서는 [prediction](/docs/predict)을 참조하십시오.

## 데이터셋 형식

OCR 레이블은 스플릿마다 하나의 JSONL 파일로 구성되며, 이미지 자체 외에 이미지마다 하나의 JSON 객체가 있습니다.

```text
my-ocr-dataset/
  images/
    val/receipt.jpg
  labels/
    val.jsonl
```

각 줄은 이미지를 이름 짓고 그 영역을 나열합니다:

```json
{"image": "receipt.jpg", "regions": [{"polygon": [[10, 12], [118, 14], [117, 40], [9, 38]], "text": "TOTAL 12.50"}]}
```

`polygon`는 절대 픽셀 좌표에서 네 점 쿼드로, 순서는 좌상, 우상, 우하, 좌하입니다. 텍스트를 읽을 수 없는 영역은 `"text": "###"`로 표시되며, 이는 ICDAR의 관대 규칙으로: 인식 점수에서 제외되며, 그 위에 겹치는 예측은 잘못된 긍정으로 계산되지 않고 무시됩니다.

루트 디렉토리를 `data=`로 지정하는 것만으로 충분합니다. 데이터셋 YAML이 대안이며, 이 경우 `path`와 선택적으로 `images` 및 `labels` 디렉토리 이름, 그리고 `nc: 1`와 `names: {0: text}`를 스키마 자리 표시자로 사용합니다. 이는 OCR 모델이 객체 탐지 대신 `Results.ocr`를 반환하기 때문입니다. 전체 계약 내용은 [데이터셋 형식](/docs/reference/dataset-formats)을 참조하십시오.

## 학습

어느 OCR 계열도 학습 구현을 가지고 있지 않습니다: `train()`는 둘 다에서 `NotImplementedError`를 발생시키며, OCR 지원은 예측과 검증만 다룹니다. PP-OCRv5의 페이지는 Apache-2.0 업스트림 학습 코드와 파인튜닝된 체크포인트를 다시 LibreYOLO로 가져오는 변환 스크립트를 명시하고 있습니다.

## 검증

`val()`는 전체 파이프라인, 탐지 및 인식을 함께 평가하며, 예측된 다각형과 실제 다각형을 IoU가 0.5 이상일 때 일대일로 매칭합니다.

<code-tabs name="val" />

`metrics/det_precision`, `metrics/det_recall` 및 `metrics/det_hmean`는 위치 정보만 평가합니다: 매칭은 다각형 겹침만 필요하며, 전사 내용은 상관없습니다. `metrics/e2e_precision`, `metrics/e2e_recall` 및 `metrics/e2e_f1`는 읽기를 추가합니다: 매칭은 동일한 다각형 겹침과 NFKC 정규화 및 공백 제거 후 정확한 전사 일치가 필요하며, 비교는 대소문자를 구분합니다. `metrics/e2e_f1`는 또한 `fitness`이기도 하며, 숫자 최적 체크포인트 선택을 의미합니다.

`metrics/rec_1-NED`는 이미 매치된 쌍 탐지 위에서 스스로 인식기를 평가합니다: 정규화된 편집 거리에서 1을 뺀 값으로, 한 글자만 다른 전사는 점수가 1에 가까운 반면, end-to-end F1은 0으로 점수를 매깁니다.

## 내보내기

이 작업에는 사용할 수 있는 내보내기 형식이 없습니다. PP-OCRv5는 추적 가능한 하나의 그래프가 아닌 두 개의 네트워크가 함께 움직이는 구조이며, `export()`는 두 계열의 모든 형식에 대해 발생합니다. LibreYOLO 외부에 배포하려면, 업스트림을 파인튜닝하고 업스트림 배포 경로를 사용하십시오.
