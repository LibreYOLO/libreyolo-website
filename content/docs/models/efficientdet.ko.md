---
title: EfficientDet
families:
  - efficientdet
seo_title: 'EfficientDet: LibreYOLO의 객체 탐지'
description: >-
  LibreYOLO에서 EfficientDet D0-D4를 실행합니다. Apache-2.0의 BiFPN 탐지기로 예측, 검증하고 ONNX,
  TensorRT, OpenVINO로 내보낼 수 있습니다.
lead: >-
  EfficientDet은 EfficientNet 백본과 반복 양방향 특징 피라미드 네트워크(BiFPN)를 결합하고 다섯 가지 크기에서 깊이,
  너비, 해상도를 함께 조정합니다. LibreYOLO는 이를 추론 전용 탐지기로 제공합니다.
keywords:
  - EfficientDet 사용법
  - BiFPN 객체 탐지
  - EfficientNet 탐지기
  - 객체 탐지 모델
  - 복합 스케일링
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreEfficientDetd0.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreEfficientDetd0.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEfficientDetd0.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreEfficientDetd0.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEfficientDetd0.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreEfficientDetd0.pt format=onnx
        libreyolo export model=LibreEfficientDetd0.pt format=tensorrt half=True
    - label: 내보낸 파일 사용
      language: python
      code: |
        from libreyolo import LibreYOLO

        # 팩토리는 파일 접미사에 따라 라우팅하므로 내보낸 아티팩트도
        # 다른 체크포인트처럼 불러와 동일한 Results 객체를 반환합니다.
        model = LibreYOLO("LibreEfficientDetd0.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 12c61fb0035437ce
---

## 설치

EfficientDet에는 선택적 extra가 필요하지 않습니다. 가져오는 모든 항목이 기본 설치에 포함됩니다.

```bash
pip install libreyolo
```

## 예측

처음 사용할 때 Hugging Face에서 가중치를 다운로드해 로컬에 캐시합니다.

<code-tabs name="predict" />

반환되는 `Results` 객체는 모든 계열이 반환하는 것과 같으므로 탐지기를 바꾸려면 한 줄만 변경하면 됩니다. EfficientDet은 앵커 기반 후보를 디코딩한 다음 클래스별 비최대 억제를 실행하므로 여기서는 `conf`, `iou`, `max_det`이 모두 실제로 영향을 줍니다. 소스, 스트리밍, 결과 처리는 [예측](/docs/predict)을 참조합니다.

## 변형

D0부터 D4까지 다섯 가지 크기가 있습니다. 한 단계 커질 때마다 더 큰 EfficientNet 백본이 더 깊고 넓은 BiFPN 및 더 깊은 예측 헤드와 결합됩니다. 따라서 논문의 복합 스케일링 규칙에 따라 매개변수 수와 연산량이 함께 증가합니다.

## 검증

`val()`은 학습에 사용한 형식의 데이터셋을 대상으로 측정한 정밀도, 재현율, mAP 50, mAP 50-95를 포함하는 `metrics/` 키 사전을 반환합니다.

<code-tabs name="val" />

## 내보내기

<export-matrix />

내보낸 아티팩트는 파일 접미사에 따라 `LibreYOLO()`로 다시 불러옵니다. 따라서 `.onnx` 또는 `.engine` 파일은 체크포인트처럼 동작하며 동일한 `Results`를 반환합니다.

<code-tabs name="export" />

## 체크포인트

이 계열에 공개된 모든 가중치 파일입니다.

<checkpoint-table />

## 라이선스

<provenance-box>

LibreYOLO의 D0-D4 체크포인트는 Apache-2.0 라이선스의 rwightman/efficientdet-pytorch 프로젝트를 통해 변환됩니다. 이 프로젝트 자체는 학습된 텐서를 변경하지 않고 google/automl의 공식 TensorFlow 학습 가중치를 미러링합니다. LGPL 라이선스의 zylo117/Yet-Another-EfficientDet-Pytorch 프로젝트 소스는 참조하거나 사용하지 않았습니다.

</provenance-box>
