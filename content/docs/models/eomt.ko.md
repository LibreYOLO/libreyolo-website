---
title: EoMT
families:
  - eomt
seo_title: 'EoMT: 의미, 인스턴스 및 파놉틱 분할 예측'
description: >-
  LibreYOLO에서 디코더가 필요 없는 plain DINOv2 vision transformer 기반 EoMT로 의미, 인스턴스, 파놉틱
  분할을 수행합니다. MIT 라이선스입니다.
lead: >-
  전용 픽셀 디코더 없이 plain vision transformer에 구축된 분할 네트워크입니다. 인코더 자체에 추가된 학습 쿼리가 마스크를
  예측합니다. LibreYOLO는 의미, 인스턴스, 파놉틱 분할을 지원합니다.
keywords:
  - EoMT 사용법
  - encoder-only mask transformer
  - DINOv2 분할
  - 파놉틱 분할
  - 인스턴스 분할
  - 의미 분할
last_verified: 1.5.0
snippets:
  predict:
    - label: 의미 분할
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreEoMTl-sem.pt")
        result = model(SAMPLE_IMAGE, save=True)

        mask = result.semantic_mask
        print(mask.data.shape)   # (H, W) 클래스 ID
        print(mask.classes)      # 이미지에 있는 클래스 ID 정렬 목록
    - label: 인스턴스 분할
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 파일명의 -seg 접미사가 인스턴스 작업을 선택하므로 여기서는
        # task 인수가 필요하지 않습니다.
        model = LibreYOLO("LibreEoMTl-seg.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.boxes.xyxy)
        print(result.masks.data.shape)
    - label: 파놉틱 분할
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreEoMTl-panoptic.pt")
        result = model(SAMPLE_IMAGE, save=True)

        pan = result.panoptic
        print(pan.data.shape)       # (H, W) 세그먼트 ID
        print(pan.segments_info)    # [{"id": ..., "category_id": ...}, ...]
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreEoMTl-sem.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: 의미 분할
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEoMTl-sem.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: 인스턴스 분할
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEoMTl-seg.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95(M)"])   # 마스크
        print(metrics["metrics/mAP50-95(B)"])   # 박스
    - label: 파놉틱 분할
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEoMTl-panoptic.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/PQ"])
        print(metrics["metrics/SQ"], metrics["metrics/RQ"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreEoMTl-sem.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEoMTl-sem.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreEoMTl-sem.pt format=onnx
        libreyolo export model=LibreEoMTl-sem.pt format=tensorrt half=True
    - label: 내보낸 파일 사용
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 팩토리는 파일 접미사에 따라 라우팅하므로 내보낸 아티팩트도
        # 다른 체크포인트처럼 불러와 동일한 Results 객체를 반환합니다.
        model = LibreYOLO("LibreEoMTl-sem.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.semantic_mask.data.shape)
source_hash: 64b2da642999f150
---

## 설치

EoMT에는 선택적 extra가 필요하지 않습니다. 가져오는 모든 항목이 기본 설치에 포함됩니다.

```bash
pip install libreyolo
```

## 예측

처음 사용할 때 Hugging Face에서 가중치를 다운로드해 로컬에 캐시합니다. 파일명의 작업 접미사(`-sem`, `-seg`, `-panoptic`)가 작업을 선택하며 `LibreYOLO()`가 파일명에서 이를 추론하므로 `task=` 인수가 필요하지 않습니다.

<code-tabs name="predict" />

의미 분할은 `result.semantic_mask`를 채우며 `.data`에는 클래스 ID의 `(H, W)` 배열이 들어 있습니다. 인스턴스 분할은 다른 분할 계열과 같은 형태로 `result.boxes`와 `result.masks`를 채웁니다. 파놉틱 분할은 `result.panoptic`을 채웁니다. `.data`에는 `(H, W)` 세그먼트 ID 맵이, `.segments_info`에는 세그먼트마다 하나씩 `{"id", "category_id"}` 사전 목록이 들어 있습니다. `conf`는 쿼리 선택을 필터링합니다. 의미 작업은 NMS 단계 없이 픽셀별 argmax를 사용하므로 `iou`는 효과가 없습니다. 소스, 스트리밍, 결과 처리는 [예측](/docs/predict)을 참조합니다.

## 변형

DINOv2 기반 인코더 크기는 s/b/l 세 가지입니다. 의미 체크포인트는 512 px의 ADE20K에서 학습했고 인스턴스 및 파놉틱 체크포인트는 640 px의 COCO에서 학습했으며, 인스턴스 체크포인트 하나는 1280 px에서 추가로 학습했습니다. 업스트림은 DINOv2 인스턴스 분할 가중치를 l 크기로만 제공합니다. s와 b는 의미 및 파놉틱 작업용으로만 공개됩니다. DINOv3 기반 EoMT 변형도 업스트림에 있지만 접근 제한이 있는 비상업적 DINOv3 가중치에 의존하므로 여기서는 제공하지 않습니다.

LibreYOLO는 EoMT를 학습하지 않습니다. 이 계열에서 `train()`은 `NotImplementedError`를 발생시키며 위의 [지원 티어](/docs/models)에서는 이를 추론 전용으로 표시합니다.

## 검증

`val()`은 작업에 따라 디스패치합니다. 의미 분할은 `metrics/mIoU`와 `metrics/pixel_accuracy`를 반환합니다. 인스턴스 분할은 다른 분할 계열과 같은 마스크 및 박스 mAP 키를 반환합니다. 파놉틱 분할은 `metrics/PQ`로 Panoptic Quality를 반환하고 이를 `metrics/SQ`(분할 품질)와 `metrics/RQ`(인식 품질)로 나누며 `metrics/PQ_things`와 `metrics/PQ_stuff`도 반환합니다.

<code-tabs name="val" />

## 내보내기

<export-matrix />

현재는 의미 작업만 내보낼 수 있습니다. 인스턴스와 파놉틱 분할에서 `export()`를 호출하면 `NotImplementedError`가 발생합니다. 쿼리 마스크 출력의 런타임 내보내기 계약이 아직 없기 때문입니다. 내보낸 의미 아티팩트는 파일 접미사에 따라 `LibreYOLO()`로 다시 불러오므로 `.onnx` 또는 `.engine` 파일은 체크포인트처럼 동작하며 동일한 `Results`를 반환합니다.

<code-tabs name="export" />

## 체크포인트

이 계열에 공개된 모든 가중치 파일입니다.

<checkpoint-table />

## 라이선스

<provenance-box></provenance-box>

## 인용

<citation-block />
