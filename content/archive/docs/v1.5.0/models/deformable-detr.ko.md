---
title: Deformable DETR
families:
  - deformable_detr
seo_title: 'Deformable DETR: Apache-2.0 기반 예측 및 내보내기'
description: >-
  LibreYOLO에서 Deformable DETR로 객체 탐지를 수행합니다. 모두 Apache-2.0 라이선스인 다섯 가지 희소 어텐션
  크기를 설치하고 예측, 검증, 내보내기합니다.
lead: >-
  Deformable DETR은 DETR의 조밀 교차 어텐션을 각 참조점 주변의 희소 다중 스케일 샘플링으로 대체하여 transformer
  탐지기를 실용적으로 학습할 수 있게 했습니다. LibreYOLO는 탐지용으로 다섯 가지 크기를 제공하며 추론만 지원합니다.
keywords:
  - Deformable DETR 사용법
  - detection transformer
  - 희소 어텐션
  - 다중 스케일 어텐션
  - 객체 탐지
  - SenseTime
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDeformableDETRr50.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreDeformableDETRr50.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDeformableDETRr50.pt")

        # val()은 객체가 아닌 일반 dict를 반환합니다.
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDeformableDETRr50.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDeformableDETRr50.pt")
        model.export(format="onnx", imgsz=800)
        model.export(format="tensorrt", imgsz=800, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreDeformableDETRr50.pt format=onnx imgsz=800

        libreyolo export model=LibreDeformableDETRr50.pt format=tensorrt
        imgsz=800 half=True
    - label: 내보낸 파일 사용
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 팩토리는 파일 접미사에 따라 라우팅하므로 내보낸 아티팩트도
        # 다른 체크포인트처럼 불러와 동일한 Results 객체를 반환합니다.
        model = LibreYOLO("LibreDeformableDETRr50.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 35225efc54b5ef91
---

## 설치

Deformable DETR에는 선택적 extra가 필요하지 않습니다. 순수 PyTorch 다중 스케일 deformable attention 코어를 사용하며 가져오는 모든 항목이 기본 설치에 포함됩니다.

```bash
pip install libreyolo
```

`libreyolo[hub-kernels]` 설치는 선택 사항입니다. `kernels` 패키지가 있으면 LibreYOLO는 런타임에 Hugging Face Hub에서 컴파일된 다중 스케일 deformable attention 커널을 가져와 순수 PyTorch 코어 대신 사용합니다. `LIBREYOLO_HUB_KERNELS=0`으로 다시 비활성화할 수 있습니다.

## 예측

처음 사용할 때 Hugging Face에서 가중치를 다운로드해 로컬에 캐시합니다.

<code-tabs name="predict" />

반환되는 `Results` 객체는 모든 계열이 반환하는 것과 같으므로 탐지기를 바꾸려면 한 줄만 변경하면 됩니다. `conf`와 `max_det`은 쿼리 선택을 필터링합니다. 디코더가 NMS 단계 없는 집합 예측기이므로 `iou`는 API 일관성을 위해 허용되지만 아무 효과가 없습니다. 소스, 스트리밍, 결과 처리는 [예측](/docs/predict)을 참조합니다.

LibreYOLO에서 Deformable DETR은 추론 전용입니다. 업스트림은 헝가리안 매칭과 focal 분류 손실로 학습하지만 해당 레시피는 여기 구현되지 않았으므로 `train()`은 `NotImplementedError`를 발생시킵니다.

## 변형

공개된 구성을 포괄하는 체크포인트는 다섯 가지이며 모두 같은 입력 해상도를 사용합니다. `r50ss`는 어텐션을 단일 특징 스케일로 제한하고 `r50ssdc5`는 여기에 팽창 C5 백본 단계를 추가합니다. `r50`은 네 특징 맵 레벨에서 샘플링하는 기본 다중 스케일 구성입니다. `r50refine`은 디코더 계층 전반에 반복 바운딩 박스 정제를 추가하고 `r50twostage`는 학습된 쿼리 대신 인코더 출력에서 초기 영역 제안을 생성합니다.

## 검증

`val()`은 학습에 사용한 형식의 데이터셋을 대상으로 측정한 정밀도, 재현율, mAP 50, mAP 50-95를 포함하는 `metrics/` 키 사전을 반환합니다.

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

## 인용

<citation-block />
