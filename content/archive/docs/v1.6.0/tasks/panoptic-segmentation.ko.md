---
title: 판옵틱 분할
seo_title: LibreYOLO에서의 전방위 분할
description: >-
  LibreYOLO에서 모든 픽셀에 하나의 세그먼트를 할당합니다: 작업을 수행하는 계열, COCO-팬옵틱 데이터셋 형식, 그리고
  predict와 validate 호출.
lead: >-
  전방위 분할(panoptic segmentation)은 모든 픽셀을 정확히 하나의 겹치지 않는 분할에 할당하며, 셀 수 있는 객체 인스턴스와
  형태가 없는 배경 영역을 통합합니다. 이 작업의 핵심은 전방위(panoptic)입니다.
keywords:
  - 파노프틱 세그멘테이션 파이썬
  - 전면적 품질
  - 사물 및 물체 분할
  - COCO 전방위 형식
  - 세그먼트 ID 맵
  - PQ 지표
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 파일 이름의 -panoptic 접미사는 작업을 선택하므로, 작업이 없습니다
        # 주장이 필요합니다.
        model = LibreYOLO("LibreEoMTl-panoptic.pt")
        result = model(SAMPLE_IMAGE, save=True)

        pan = result.panoptic
        print(pan.data.shape)       # (H, W) 세그먼트 아이디
        print(pan.segments_info)    # [{"id": ..., "category_id": ...}, ...]
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreEoMTl-panoptic.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: 한 번에 하나의 구간
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        result = LibreYOLO("LibreEoMTl-panoptic.pt")(SAMPLE_IMAGE)
        pan = result.panoptic

        for segment in pan.segments_info:
            pixels = pan.segment_mask(segment["id"])   # 부울 (H, W)
            print(result.names[segment["category_id"]], int(pixels.sum()))
    - label: 더 작은 검문소
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreEoMTs-panoptic.pt")
        result = model(SAMPLE_IMAGE)

        print(len(result.panoptic.segment_ids))
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEoMTl-panoptic.pt")

        # val()은 객체가 아니라 일반 딕셔너리를 반환합니다.
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/PQ"])
        print(metrics["metrics/SQ"], metrics["metrics/RQ"])
        print(metrics["metrics/PQ_things"], metrics["metrics/PQ_stuff"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreEoMTl-panoptic.pt data=my-dataset.yaml
source_hash: b8adc9ccde7a4e6c
---

## 정의

파노프틱 세그멘테이션은 다른 두 개의 세그멘테이션 작업의 합입니다. 모든 픽셀은 정확히 하나의 세그먼트를 가지며, 세그먼트는 겹치지 않고, 세그먼트는 'thing', 즉 셀 수 있는 객체 인스턴스이거나, 'stuff', 즉 하늘이나 도로와 같은 무정형 영역 중 하나입니다. 이는 배경 픽셀을 할당하지 않고 마스크가 겹칠 수 있는 [인스턴스 분할](/docs/tasks/instance-segmentation)보다 엄격하며, 모든 픽셀에 레이블을 부여하지만 하나의 클래스에 속하는 인스턴스를 병합하는 [시맨틱 분할](/docs/tasks/semantic-segmentation)보다도 엄격합니다.

`panoptic`는 표준 작업 키이며, 체크포인트 파일 이름의 `-panoptic` 접미사가 이를 선택하므로, 공개된 가중치를 로드할 때 `task=`는 필요하지 않습니다.

`predict()`는 `result.panoptic`를 채웁니다. `.data`는 원본 이미지 캔버스에 대한 `(H, W)` 정수 세그먼트-ID 맵입니다. `.segments_info`는 세그먼트당 하나씩의 dict 목록으로, 최소한 `{"id", "category_id"}`를 가지며, 여기서 `id`는 맵의 값과 일치하고 `category_id`는 `result.names`를 인덱싱합니다. `.segment_ids`는 정렬된 순서로 존재하는 ID를 나열하고 `.segment_mask(id)`는 한 세그먼트에 대한 불리언 `(H, W)` 선택을 반환합니다. 세그먼트 ID `0`는 공란 값입니다: 레이블이 없는 픽셀로, 지표에서 제외되며 `.segment_ids`에서 제외됩니다.

Thing과 stuff는 개별 세그먼트의 속성이 아니라 범주의 속성입니다. 이것은 레이블 세트의 범주 메타데이터에 담겨 있으며, 예측 페이로드가 편의를 위해 각 세그먼트에 `"isthing"`로 복사할 수 있지만, 범주 메타데이터가 여전히 권위적입니다.

## 모델들

[EoMT](/docs/models/eomt)는 `LibreYOLO()`를 통해 이 작업을 수행하는 계열입니다. 기본 패키지에서 실행되며, COCO에서 학습된 세 가지 크기(s, b, l)의 파노프틱 체크포인트를 제공합니다.

[SenseNova-Vision](/docs/models/sensenova-vision) 또한 파노옵틱 맵을 생성합니다. 이 모델은 자체 팩토리 `LibreVLM`와 자체 추가 기능을 갖춘 프롬프트 기반 생성 모델이며, 어휘 집합이 없을 경우 조정된 COCO 파노옵틱 카테고리로 대체됩니다. 모델 가중치는 비상업적입니다. 이미지당 지연 시간은 목적에 맞게 제작된 세그먼트 모델보다 훨씬 높습니다. 이는 모든 예측이 디퓨전 디코딩을 거치기 때문입니다.

## 예측

가중치는 처음 사용할 때 Hugging Face에서 다운로드되며 로컬에 캐시됩니다.

<code-tabs name="predict" />

`conf`는 쿼리 선택을 필터링합니다. 소스, 스트리밍 및 결과 처리에 대해서는 [예측](/docs/predict)을 참조하십시오.

## 데이터셋 형식

LibreYOLO는 Kirillov 등, CVPR 2019에서 제시한 COCO-파노라마 형식을 그대로 채택합니다. LibreYOLO 전용 파노라마 레이아웃은 없습니다.

```text
dataset/
  data.yaml
  images/
    val/000000000139.jpg
  annotations/
    panoptic_val.json
    panoptic_val/000000000139.png
```

각 이미지는 동일한 해상도의 하나의 RGB PNG와 쌍을 이루며, 각 픽셀의 색상은 그것이 속한 세그먼트의 ID를 인코딩합니다:

```text
segment_id = R + 256 * G + 256 * 256 * B
```

세그먼트 ID `0`, RGB 블랙은 빈 공간입니다: 예측에 대해 보상도 부과하지도 않는 레이블 없는 픽셀입니다. 나머지 모든 픽셀은 정확히 하나의 세그먼트에 속합니다.

JSON은 이미지별로 세그먼트 ID PNG와 그 안의 세그먼트를 나열합니다:

```json
{
  "images":      [{"id": 139, "file_name": "000000000139.jpg"}],
  "annotations": [{"image_id": 139, "file_name": "000000000139.png",
                   "segments_info": [
                     {"id": 3226956, "category_id": 1, "area": 2840,
                      "bbox": [413, 158, 53, 138], "iscrowd": 0}]}],
  "categories":  [{"id": 1, "name": "person", "isthing": 1}]
}
```

`annotations[].file_name`는 panoptic 디렉토리 안의 PNG를 이름 짓고, `segments_info[].id`는 그 PNG 안의 값을 일치시킵니다. `iscrowd`는 그룹 영역을 표시합니다: 이들은 절대로 거짓 부정으로 계산되지 않으며, 주로 하나를 덮는 예측은 거짓 양성으로 간주되지 않습니다. `isthing`는 `categories` 위에 존재하며 개별 세그먼트 위에는 존재하지 않습니다.

YAML은 둘 다를 가리킵니다:

```yaml
path: dataset
val: images/val
annotations:
  val: annotations/panoptic_val.json
panoptic_dir:
  val: annotations/panoptic_val
names:
  0: person
  1: bicycle
```

`annotations`와 `panoptic_dir`는 각각 단일 경로 또는 분할별 매핑을 허용합니다. 원시 COCO 카테고리 ID는 일반적으로 연속적이지 않지만, 모델은 연속적인 `0..nc-1`를 예측하므로 ID는 카테고리 이름을 통해 `names`로 다시 매핑됩니다. `names`에서 누락된 JSON 카테고리는 조용히 제거되는 것이 아니라 오류로 간주되며, 제거하면 영구적인 false negative로 채점되기 때문입니다.

정식 로더는 `libreyolo.data.PanopticDataset`입니다.

## 학습

오늘 LibreYOLO에서는 어떤 계열도 파노프틱 세분화를 학습하지 않습니다: EoMT의 `train()`가 `NotImplementedError`를 상승시키므로, 파노프틱 체크포인트는 발표된 대로 사용됩니다.

## 검증

`val()`는 데이터셋 YAML에서 `val`로 이름 붙여진 분할에 대해 실제 해상도로 계산된 `metrics/` 키의 일반 딕셔너리를 반환합니다. 동일한 범주의 예측된 세그먼트와 실제 세그먼트는 IoU가 0.5를 초과할 때 일치하며, 그 일치는 고유합니다.

<code-tabs name="val" />

`metrics/PQ`는 전체 품질(Panoptic Quality)로, 주요 수치입니다. 한 카테고리 내에서는 두 요소의 곱으로 계산됩니다. 분할 품질(Segmentation quality)은 매칭된 세그먼트의 평균 IoU이며, 매칭된 형태가 얼마나 잘 맞는지를 나타냅니다. 인식 품질(Recognition quality)은 `TP / (TP + 0.5 FP + 0.5 FN)`로, 매칭 자체의 F1 점수이며 전체 세그먼트 중 얼마나 많이 발견되었는지를 나타냅니다. 세 가지 수치 모두 나타난 카테고리에 대해 평균을 내어 `metrics/PQ`, `metrics/SQ` 및 `metrics/RQ`로 보고되며, 따라서 보고된 PQ는 각 카테고리 곱의 평균이지 보고된 두 평균의 곱이 아닙니다.

`metrics/PQ_things`와 `metrics/PQ_stuff`는 thing 카테고리와 stuff 카테고리에서 각각의 카테고리별 PQ를 동일하게 평균하고, `metrics/categories`는 나타난 카테고리를 세어 평균에 포함된 카테고리 수를 기록합니다. 이 사전에는 PQ 값의 복사본인 `fitness`도 포함되어 있습니다.

## 내보내기

파노프틱 체크포인트는 내보낼 수 없습니다. `export()`는 이 작업에 대해 `NotImplementedError`를 제기합니다. 왜냐하면 쿼리-마스크 출력에는 아직 런타임 내보내기 계약이 없기 때문입니다. EoMT의 의미론적 작업은 내보낼 수 있습니다; [시맨틱 분할](/docs/tasks/semantic-segmentation)과 [내보내기 및 배포](/docs/export)를 참조하십시오.
