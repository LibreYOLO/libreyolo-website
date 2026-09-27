---
title: Mask R-CNN
families:
  - mask_rcnn
seo_title: 'LibreYOLO의 Mask R-CNN: 예측, 검증 및 내보내기'
description: >-
  LibreYOLO에서 Mask R-CNN으로 객체 탐지와 인스턴스 분할을 수행합니다. BSD-3-Clause torchvision 이식
  모델을 설치하고 예측, 검증, 내보내기합니다.
lead: >-
  Mask R-CNN은 Faster R-CNN에 영역별 마스크 브랜치를 추가하여 탐지한 각 박스와 함께 분할 마스크를 예측합니다.
  LibreYOLO는 객체 탐지와 인스턴스 분할을 위해 torchvision 구현을 이식합니다.
keywords:
  - Mask R-CNN 사용법
  - 인스턴스 분할
  - 객체 탐지
  - Faster R-CNN
  - torchvision
  - 2단계 탐지기
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreMaskRCNNr50.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.masks.data.shape)
        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreMaskRCNNr50.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: 박스만 반환
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # task="detect"는 마스크 헤드를 건너뛰고 같은 체크포인트에서
        # 박스를 반환하며 결과에는 마스크가 없습니다.
        model = LibreYOLO("LibreMaskRCNNr50.pt", task="detect")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMaskRCNNr50.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])      # 마스크
        print(metrics["metrics/mAP50-95(B)"])   # 박스
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreMaskRCNNr50.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMaskRCNNr50.pt")
        model.export(format="onnx", imgsz=800)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreMaskRCNNr50.pt format=onnx imgsz=800
    - label: 내보낸 파일 사용
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 팩토리는 파일 접미사에 따라 라우팅하므로 내보낸 아티팩트도
        # 다른 체크포인트처럼 불러와 동일한 Results 객체를 반환합니다.
        model = LibreYOLO("LibreMaskRCNNr50.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.masks.data.shape)
source_hash: 9608459b801aa6d5
---

## 설치

Mask R-CNN에는 선택적 extra가 필요하지 않습니다. 가져오는 모든 항목이 기본 설치에 포함됩니다.

```bash
pip install libreyolo
```

## 예측

처음 사용할 때 Hugging Face에서 가중치를 다운로드해 로컬에 캐시합니다.

<code-tabs name="predict" />

반환되는 `Results` 객체는 모든 계열이 반환하는 것과 같으므로 탐지기를 바꾸려면 한 줄만 변경하면 됩니다. `task` 인수 없이 체크포인트를 불러오면 인스턴스 마스크를 반환합니다. 분할이 이 계열의 기본 작업이므로 `result.masks`에 박스와 함께 마스크가 담깁니다. `task="detect"`를 전달하면 마스크 헤드 없이 같은 가중치를 불러와 박스만 반환합니다. `conf`와 `iou`는 신뢰도 및 NMS 임곗값을 설정합니다. 쿼리 기반 탐지기와 달리 Mask R-CNN은 업스트림 NMS 단계를 유지합니다. 소스, 스트리밍, 결과 처리는 [예측](/docs/predict)을 참조합니다.

## 변형

백본은 torchvision의 v2 Mask R-CNN 빌더를 사용하는 특징 피라미드 ResNet-50 하나입니다. 공개된 체크포인트에는 BSD-3-Clause 라이선스가 적용되며 이 계열의 두 작업에 모두 사용됩니다. 따라서 선택할 크기가 없습니다.

## 검증

`val()`은 `metrics/` 키 사전을 반환합니다. 이 체크포인트의 기본 분할 작업을 기준으로 일반 `metrics/mAP50-95` 키에는 마스크 점수가 들어가며 같은 실행에서 `(B)` 접미사 키로 박스도 보고하므로 한 번의 패스로 둘 다 얻을 수 있습니다.

<code-tabs name="val" />

## 내보내기

<export-matrix />

Mask R-CNN은 배치 크기 1로 ONNX에만 내보낼 수 있습니다. 내보낸 그래프는 업스트림 크기 조정과 마스크 붙여넣기 단계를 내부에 유지합니다. 정사각형이 아닌 소스에도 그래프가 유효하도록 LibreYOLO는 전달값과 관계없이 `dynamic=True`를 강제합니다. 내보낸 `.onnx` 파일은 파일 접미사에 따라 `LibreYOLO()`로 다시 불러오며 동일한 `Results`를 반환합니다.

<code-tabs name="export" />

## 체크포인트

이 계열에 공개된 모든 가중치 파일입니다. 아래 체크포인트 하나는 detect 아래에 표시되지만 같은 파일을 분할에도 불러옵니다. `task` 인수를 생략하면 기본으로 마스크를 반환합니다.

<checkpoint-table />

## 라이선스

<provenance-box>

Mask R-CNN은 LibreYOLO의 Faster R-CNN 래퍼를 상속한 하위 클래스로 구축됩니다. 같은 torchvision 소스와 BSD-3-Clause 라이선스를 공유하며 동일하게 이식된 커밋의 마스크 예측기와 마스크 RoI 헤드를 추가합니다.

</provenance-box>
