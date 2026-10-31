---
title: Faster R-CNN
families:
  - faster_rcnn
seo_title: 'LibreYOLO의 Faster R-CNN: 예측, 검증, 내보내기'
description: >-
  LibreYOLO에서 백본 4개로 객체 탐지에 Faster R-CNN을 사용합니다. BSD-3-Clause torchvision 포트를
  설치하고 예측, 검증, 내보내기합니다.
lead: >-
  Faster R-CNN은 영역 제안 네트워크가 2단계 분류기에 입력되는 방식으로 객체를 탐지하며, 별도 단계였던 영역 제안을 같은 학습
  네트워크에 포함시킨 아키텍처입니다. LibreYOLO는 탐지용 torchvision 구현을 포팅합니다.
keywords:
  - Faster R-CNN 사용법
  - 객체 탐지
  - 영역 제안 네트워크
  - 2단계 탐지기
  - torchvision
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreFasterRCNNl.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreFasterRCNNl.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFasterRCNNl.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreFasterRCNNl.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFasterRCNNl.pt")
        model.export(format="onnx", imgsz=800)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreFasterRCNNl.pt format=onnx imgsz=800
    - label: 내보낸 파일 사용하기
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 팩토리는 파일 접미사로 라우팅하므로 내보낸 아티팩트도
        # 체크포인트처럼 불러오며 동일한 Results 객체를 반환합니다.
        model = LibreYOLO("LibreFasterRCNNl.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 3fd82eb835399560
---

## 설치

Faster R-CNN에는 선택적 extra가 필요하지 않습니다. 가져오는 모든 항목이 기본
설치에 포함됩니다.

```bash
pip install libreyolo
```

## 예측

가중치는 처음 사용할 때 Hugging Face에서 내려받아 로컬에 캐시됩니다.

<code-tabs name="predict" />

반환되는 `Results` 객체는 모든 계열에서 동일하므로 다른 탐지기로 바꾸려면 한 줄만
수정하면 됩니다. `conf`와 `iou`는 신뢰도 및 NMS 임계값을 설정합니다. 쿼리 기반
탐지기와 달리 Faster R-CNN은 업스트림 NMS 단계를 유지합니다. 소스, 스트리밍,
결과 처리는 [예측](/docs/predict)을 참조합니다.

## 변형

크기는 4가지이며 같은 모델의 비례 변형이 아니라 각각 다른 torchvision 구성입니다.
`n`은 320 px 입력의 MobileNetV3-Large, `s`는 800 px 입력의 같은 백본, `m`은
특징 피라미드가 있는 ResNet-50, `l`은 더 깊은 영역 제안 헤드와 `m`의 헤드 대신
4개 컨볼루션 바운딩 박스 헤드를 사용하는 v2 개정판입니다. `n`과 `s`는 더 가벼운
백본을 위해 정확도를 절충합니다.

## 검증

`val()`은 학습에 사용한 형식의 모든 데이터셋을 대상으로 측정한 정밀도, 재현율,
mAP 50, mAP 50-95를 포함하는 `metrics/` 키 딕셔너리를 반환합니다.

<code-tabs name="val" />

## 내보내기

<export-matrix />

Faster R-CNN은 배치 크기 1의 ONNX로만 내보냅니다. 내보낸 그래프는 업스트림 크기
조정 단계를 내부에 유지하므로 LibreYOLO는 전달된 값과 관계없이 `dynamic=True`를
강제하여 정사각형이 아닌 소스에도 그래프가 유효하도록 합니다. 내보낸 `.onnx`
파일은 파일 접미사에 따라 `LibreYOLO()`로 다시 불러오며 동일한 `Results`를
반환합니다.

<code-tabs name="export" />

## 체크포인트

이 계열에서 공개된 모든 가중치 파일입니다.

<checkpoint-table />

## 라이선스

<provenance-box></provenance-box>

## 인용

<citation-block />
