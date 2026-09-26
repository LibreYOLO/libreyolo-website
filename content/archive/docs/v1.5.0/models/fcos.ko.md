---
title: FCOS
families:
  - fcos
seo_title: 'LibreYOLO의 FCOS: 예측, 검증, 내보내기'
description: >-
  LibreYOLO에서 앵커 프리 객체 탐지에 FCOS를 사용합니다. BSD-3-Clause torchvision 포트인
  ResNet-50/FPN을 설치하고 예측, 검증, 내보내기합니다.
lead: >-
  FCOS는 미리 정의한 앵커 박스 집합에 의존하지 않고 픽셀별로 객체를 탐지하며, 특징 맵의 각 위치에서 바운딩 박스와 중심성 점수를
  예측합니다. LibreYOLO는 탐지용 torchvision 구현을 포팅합니다.
keywords:
  - FCOS 사용법
  - 앵커 프리 객체 탐지
  - 객체 탐지
  - 단일 단계 탐지기
  - torchvision
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreFCOSr50.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreFCOSr50.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFCOSr50.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreFCOSr50.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFCOSr50.pt")
        model.export(format="onnx", imgsz=800)
        model.export(format="torchscript", imgsz=800)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreFCOSr50.pt format=onnx imgsz=800
    - label: 내보낸 파일 사용하기
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 팩토리는 파일 접미사로 라우팅하므로 내보낸 아티팩트도
        # 체크포인트처럼 불러오며 동일한 Results 객체를 반환합니다.
        model = LibreYOLO("LibreFCOSr50.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 60bd7b8dfd903a8c
---

## 설치

FCOS에는 선택적 extra가 필요하지 않습니다. 가져오는 모든 항목이 기본 설치에
포함됩니다.

```bash
pip install libreyolo
```

## 예측

가중치는 처음 사용할 때 Hugging Face에서 내려받아 로컬에 캐시됩니다.

<code-tabs name="predict" />

반환되는 `Results` 객체는 모든 계열에서 동일하므로 다른 탐지기로 바꾸려면 한 줄만
수정하면 됩니다. 임계값 인수 없이 모델을 호출하면 FCOS 자체의 공개된 기본값인
`conf=0.2`, `iou=0.6`, `max_det=100`을 적용합니다. 세 값 중 하나라도 전달하면
재정의할 수 있습니다. FCOS는 픽셀별 예측에 최종 NMS 단계를 유지합니다. 소스,
스트리밍, 결과 처리는 [예측](/docs/predict)을 참조합니다.

## 변형

크기는 특징 피라미드가 있는 ResNet-50 하나이며, 이 계열이 인식하는 유일한
변형입니다.

## 검증

`val()`은 학습에 사용한 형식의 모든 데이터셋을 대상으로 측정한 정밀도, 재현율,
mAP 50, mAP 50-95를 포함하는 `metrics/` 키 딕셔너리를 반환합니다.

<code-tabs name="val" />

## 내보내기

<export-matrix />

FCOS는 ONNX, TorchScript, OpenVINO로 내보냅니다. FCOS는 그래프 실행 전에 소스
종횡비를 유지하므로 LibreYOLO는 전달된 값과 관계없이 ONNX 및 OpenVINO 경로에서
`dynamic=True`를 강제하여 패딩된 입력 형상에 그래프가 유효하도록 합니다. 내보낸
`.onnx` 파일은 파일 접미사에 따라 `LibreYOLO()`로 다시 불러오며 동일한 `Results`를
반환합니다.

<code-tabs name="export" />

## 체크포인트

이 계열에서 공개된 모든 가중치 파일입니다.

<checkpoint-table />

## 라이선스

<provenance-box></provenance-box>

## 인용

<citation-block />
