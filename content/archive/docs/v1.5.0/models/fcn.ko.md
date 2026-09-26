---
title: FCN
families:
  - fcn
seo_title: 'FCN: BSD-3-Clause ResNet FCN 예측 및 내보내기'
description: >-
  LibreYOLO에서 FCN으로 의미 분할을 수행합니다. torchvision의 dilated-ResNet FCN 체크포인트를 설치하고
  예측, 검증, 내보내기합니다.
lead: >-
  탐지기의 완전 연결 계층을 합성곱으로 대체하는 조밀한 픽셀별 분류기입니다. 박스 대신 전체 해상도 클래스 맵을 출력합니다.
  LibreYOLO는 의미 분할 전용으로 FCN을 제공합니다.
keywords:
  - FCN 사용법
  - 완전 합성곱 네트워크
  - 의미 분할
  - 조밀 예측
  - ResNet 분할
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreFCNr50.pt")
        result = model(SAMPLE_IMAGE, save=True)

        mask = result.semantic_mask
        print(mask.data.shape)   # (H, W) 클래스 ID
        print(mask.classes)      # 이미지에 있는 클래스 ID 정렬 목록
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreFCNr50.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFCNr50.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreFCNr50.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFCNr50.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreFCNr50.pt format=onnx
        libreyolo export model=LibreFCNr50.pt format=tensorrt half=True
    - label: 내보낸 파일 사용
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 팩토리는 파일 접미사에 따라 라우팅하므로 내보낸 아티팩트도
        # 다른 체크포인트처럼 불러와 동일한 Results 객체를 반환합니다.
        model = LibreYOLO("LibreFCNr50.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.semantic_mask.data.shape)
source_hash: 7776b0fc85a208fb
---

## 설치

FCN에는 선택적 extra가 필요하지 않습니다. 가져오는 모든 항목이 기본 설치에 포함됩니다.

```bash
pip install libreyolo
```

## 예측

처음 사용할 때 Hugging Face에서 가중치를 다운로드해 로컬에 캐시합니다.

<code-tabs name="predict" />

의미 분할은 박스가 아니라 픽셀마다 클래스 ID 하나를 반환합니다. 따라서 `result.semantic_mask`의 `.data`에는 `(H, W)` 배열이, `.classes`에는 이미지에 있는 클래스 ID 목록이 들어 있습니다. `conf`, `iou`, `max_det`은 API 일관성을 위해 허용되지만 아무 효과가 없습니다. 모델은 신뢰도 임곗값이나 NMS 단계 없이 argmax로 모든 픽셀에 클래스를 할당합니다. 소스, 스트리밍, 결과 처리는 [예측](/docs/predict)을 참조합니다.

## 변형

ResNet 깊이는 두 가지이며 모두 고정된 520 px 입력을 사용합니다. 라이브러리의 추론 그래프는 torchvision의 dilated-ResNet FCN입니다. 스킵 연결을 사용하는 원 논문의 VGG 기반 FCN-8s 네트워크가 아닙니다.

LibreYOLO는 FCN을 학습하지 않습니다. 이 계열에서 `train()`은 `NotImplementedError`를 발생시키며 위의 [지원 티어](/docs/models)에서는 이를 추론 전용으로 표시합니다. 공개된 두 체크포인트는 torchvision 자체의 COCO 학습 가중치를 LibreYOLO 로더용으로 변환한 것입니다.

## 검증

`val()`은 학습에 사용한 형식의 데이터셋을 대상으로 측정한 `metrics/mIoU`와 `metrics/pixel_accuracy`를 반환합니다.

<code-tabs name="val" />

## 내보내기

<export-matrix />

내보낸 아티팩트는 파일 접미사에 따라 `LibreYOLO()`로 다시 불러옵니다. 따라서 `.onnx` 또는 `.engine` 파일은 체크포인트처럼 동작하며 동일한 `Results`를 반환합니다. 각 형식이 받는 인수는 [내보내기](/docs/export)에 나와 있습니다.

<code-tabs name="export" />

## 체크포인트

이 계열에 공개된 모든 가중치 파일입니다.

<checkpoint-table />

## 라이선스

<provenance-box></provenance-box>
