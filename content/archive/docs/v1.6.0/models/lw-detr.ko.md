---
title: LW-DETR
families:
  - lwdetr
seo_title: 'LW-DETR: Apache-2.0 기반 예측 및 내보내기'
description: >-
  LibreYOLO에서 LW-DETR로 실시간 객체 탐지를 수행합니다. 모두 Apache-2.0 라이선스인 다섯 가지 ViT 기반 크기를
  설치하고 예측, 검증, 내보내기합니다.
lead: >-
  Baidu가 YOLO 탐지기의 실시간 대안으로 제시한 plain-ViT detection transformer입니다. LibreYOLO는
  탐지용으로 다섯 가지 크기를 제공하며 추론만 지원합니다.
keywords:
  - LW-DETR 사용법
  - detection transformer
  - 실시간 객체 탐지
  - plain ViT
  - DETR
  - Baidu
  - Atten4Vis
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreLWDETRt.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreLWDETRt.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreLWDETRt.pt")

        # val()은 객체가 아닌 일반 dict를 반환합니다.
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreLWDETRt.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreLWDETRt.pt")
        model.export(format="onnx", imgsz=640)
        model.export(format="tensorrt", imgsz=640, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreLWDETRt.pt format=onnx imgsz=640

        libreyolo export model=LibreLWDETRt.pt format=tensorrt imgsz=640
        half=True
    - label: 내보낸 파일 사용
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 팩토리는 파일 접미사에 따라 라우팅하므로 내보낸 아티팩트도
        # 다른 체크포인트처럼 불러와 동일한 Results 객체를 반환합니다.
        model = LibreYOLO("LibreLWDETRt.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: badd1d8255df5bbd
---

## 설치

LW-DETR에는 선택적 extra가 필요하지 않습니다. 가져오는 모든 항목이 기본 설치에 포함됩니다.

```bash
pip install libreyolo
```

## 예측

처음 사용할 때 Hugging Face에서 가중치를 다운로드해 로컬에 캐시합니다.

<code-tabs name="predict" />

반환되는 `Results` 객체는 모든 계열이 반환하는 것과 같으므로 탐지기를 바꾸려면 한 줄만 변경하면 됩니다. `conf`와 `max_det`은 쿼리 선택을 필터링합니다. 디코더가 NMS 단계 없는 집합 예측기이므로 `iou`는 API 일관성을 위해 허용되지만 아무 효과가 없습니다. 소스, 스트리밍, 결과 처리는 [예측](/docs/predict)을 참조합니다.

LibreYOLO에서 LW-DETR은 추론 전용입니다. 업스트림은 여러 쿼리 그룹에 걸친 Group-DETR 일대다 지도와 IoU 인식 분류 손실로 학습하지만 해당 레시피는 여기 연결되어 있지 않으므로 `train()`은 `NotImplementedError`를 발생시킵니다.

## 변형

다섯 가지 크기는 모두 plain-ViT 인코더, 다중 스케일 프로젝터, deformable DETR 디코더를 공유하며 같은 입력 해상도에서 실행됩니다. 가장 작은 두 모델은 같은 인코더 너비를 공유하고 블록 깊이로 구분됩니다. 다음 두 모델은 더 넓은 인코더를 공유하고 디코더에 공급하는 프로젝터 레벨 수로 구분됩니다. 가장 큰 모델은 가장 넓은 인코더를 사용합니다.

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
