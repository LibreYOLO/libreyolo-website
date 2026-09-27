---
title: YOLOv2
families:
  - yolo2
seo_title: 'LibreYOLO의 YOLOv2: 예측, 검증, 내보내기'
description: >-
  LibreYOLO에서 YOLOv2(YOLO9000)를 실행합니다. 고정된 추론 전용 박물관 계열이며 퍼블릭 도메인 라이선스로 예측, 검증,
  내보내기를 지원합니다.
lead: >-
  YOLO9000으로도 발표된 YOLOv2는 YOLO 계열에 앵커 박스와 패스스루 레이어를 도입한 Darknet-19 탐지기입니다.
  LibreYOLO는 이를 고정된 추론 전용 전시물로 제공합니다.
keywords:
  - YOLOv2 사용법
  - YOLO9000
  - Darknet
  - Darknet-19
  - 객체 탐지
  - 앵커 박스
  - 이전 yolo 모델
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO2b.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreYOLO2b.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO2b.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLO2b.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO2b.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreYOLO2b.pt format=onnx
        libreyolo export model=LibreYOLO2b.pt format=tensorrt half=True
    - label: 내보낸 파일 사용하기
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 팩토리는 파일 접미사로 라우팅하므로 내보낸 아티팩트도
        # 체크포인트처럼 불러오며 동일한 Results 객체를 반환합니다.
        model = LibreYOLO("LibreYOLO2b.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: ba2884a2f6e1b0da
---

## 설치

YOLOv2에는 기본 패키지 외에 추가 항목이 필요하지 않습니다.

```bash
pip install libreyolo
```

## 예측

이 계열은 추론 전용입니다. `train()`이 `NotImplementedError`를 일으키므로 이
페이지에는 학습 절이 없습니다. 예측, 검증, 내보내기는 모두 지원합니다. 가중치는
처음 사용할 때 Hugging Face에서 내려받아 로컬에 캐시됩니다.

<code-tabs name="predict" />

반환되는 `Results` 객체는 모든 계열에서 동일하므로 다른 탐지기로 바꾸려면 한 줄만
수정하면 됩니다. `conf`는 신뢰도 임계값을, `iou`는 NMS 임계값을 필터링하며
`region` 헤드의 앵커 기반 예측에 적용됩니다. 소스, 스트리밍, 결과 처리는
[예측](/docs/predict)을 참조합니다.

## 검증

`val()`은 검증하는 형식의 모든 데이터셋을 대상으로 측정한 정밀도, 재현율, mAP 50,
mAP 50-95를 포함하는 `metrics/` 키 딕셔너리를 반환합니다.

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

