---
title: YOLOv1
families:
  - yolo1
seo_title: 'LibreYOLO의 YOLOv1: 예측, 검증, 내보내기'
description: >-
  LibreYOLO에서 원본 YOLOv1 탐지기를 실행합니다. 고정된 추론 전용 박물관 계열이며 퍼블릭 도메인 라이선스로 예측, 검증,
  내보내기를 지원합니다.
lead: >-
  YOLOv1은 YOLO 계열에 이름을 붙인 2016년 원본 탐지기입니다. 앵커 박스 없이 완전 연결 헤드가 있는 하나의 컨볼루션 네트워크가
  한 번의 순전파로 모든 바운딩 박스와 클래스 점수를 예측합니다. LibreYOLO는 이를 고정된 추론 전용 전시물로 제공합니다.
keywords:
  - YOLOv1 사용법
  - YOLO v1
  - Darknet
  - 객체 탐지
  - Pascal VOC
  - 이전 yolo 모델
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO1b.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreYOLO1b.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO1b.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLO1b.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO1b.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreYOLO1b.pt format=onnx
        libreyolo export model=LibreYOLO1b.pt format=tensorrt half=True
    - label: 내보낸 파일 사용하기
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 팩토리는 파일 접미사로 라우팅하므로 내보낸 아티팩트도
        # 체크포인트처럼 불러오며 동일한 Results 객체를 반환합니다.
        model = LibreYOLO("LibreYOLO1b.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: a786372dba86f2f8
---

## 설치

YOLOv1에는 기본 패키지 외에 추가 항목이 필요하지 않습니다.

```bash
pip install libreyolo
```

## 예측

이 계열은 추론 전용입니다. `train()`이 `NotImplementedError`를 일으키므로 이
페이지에는 학습 절이 없습니다. 예측, 검증, 내보내기는 모두 지원합니다. 가중치는
처음 사용할 때 Hugging Face에서 내려받아 로컬에 캐시됩니다.

<code-tabs name="predict" />

반환되는 `Results` 객체는 모든 계열에서 동일하므로 다른 탐지기로 바꾸려면 한 줄만
수정하면 됩니다. 이 계열에는 두 가지 고유한 점이 있습니다. 공개된 체크포인트는
COCO가 아니라 Pascal VOC(2007+2012)로 학습되었으므로 `box.cls`는 COCO 클래스
80개가 아니라 VOC 카테고리 20개(aeroplane, bicycle, bird, boat, bottle, bus,
car, cat, chair, cow, diningtable, dog, horse, motorbike, person, pottedplant,
sheep, sofa, train, tvmonitor)를 인덱싱합니다. 또한 완전 연결 탐지 헤드는 한 번에
이미지 하나만 받으므로 소스 목록은 실제 배치로 실행되지 않고 반복 처리됩니다.
소스, 스트리밍, 결과 처리는 [예측](/docs/predict)을 참조합니다.

## 검증

`val()`은 체크포인트 학습에 사용한 것과 같은 VOC 방식 레이블 공간의 데이터셋을
대상으로 측정한 정밀도, 재현율, mAP 50, mAP 50-95를 포함하는 `metrics/` 키
딕셔너리를 반환합니다.

<code-tabs name="val" />

## 내보내기

<export-matrix />

내보낸 아티팩트는 파일 접미사에 따라 `LibreYOLO()`로 다시 불러오므로 `.onnx` 또는
`.engine` 파일이 체크포인트처럼 동작하며 동일한 `Results`를 반환합니다.
LibreYOLO를 설치하지 않고 런타임에서 그래프를 직접 실행할 수도 있지만 이 경우
전처리와 후처리를 직접 작성해야 합니다.

<code-tabs name="export" />

## 체크포인트

이 계열에서 공개된 모든 가중치 파일입니다.

<checkpoint-table />

## 라이선스

<provenance-box></provenance-box>

