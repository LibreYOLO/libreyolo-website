---
title: RetinaNet
families:
  - retinanet
seo_title: 'LibreYOLO의 RetinaNet: 예측, 검증, 내보내기'
description: >-
  LibreYOLO에서 초점 손실을 사용하는 단일 단계 객체 탐지에 RetinaNet을 사용합니다. BSD-3-Clause
  torchvision 포트를 설치하고 예측, 검증, 내보내기합니다.
lead: >-
  RetinaNet은 쉬운 음성 샘플의 가중치를 낮추는 초점 손실로 학습된 단일 단계 탐지기이므로 밀집 앵커 그리드가 정확도를 유지하기 위해
  별도의 제안 단계가 필요하지 않습니다. LibreYOLO는 탐지용 torchvision 구현을 포팅합니다.
keywords:
  - RetinaNet 사용법
  - 초점 손실
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

        model = LibreYOLO("LibreRetinaNetr50v2.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreRetinaNetr50v2.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRetinaNetr50v2.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreRetinaNetr50v2.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRetinaNetr50v2.pt")
        model.export(format="onnx", imgsz=800)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreRetinaNetr50v2.pt format=onnx imgsz=800
    - label: 내보낸 파일 사용하기
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 팩토리는 파일 접미사로 라우팅하므로 내보낸 아티팩트도
        # 체크포인트처럼 불러오며 동일한 Results 객체를 반환합니다.
        model = LibreYOLO("LibreRetinaNetr50v2.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 1cc7ceb6de290bdb
---

## 설치

RetinaNet에는 선택적 extra가 필요하지 않습니다. 가져오는 모든 항목이 기본 설치에
포함됩니다.

```bash
pip install libreyolo
```

## 예측

가중치는 처음 사용할 때 Hugging Face에서 내려받아 로컬에 캐시됩니다.

<code-tabs name="predict" />

반환되는 `Results` 객체는 모든 계열에서 동일하므로 다른 탐지기로 바꾸려면 한 줄만
수정하면 됩니다. `conf`와 `iou`는 신뢰도 및 NMS 임계값을 설정합니다. RetinaNet은
밀집 앵커 그리드에서 업스트림 NMS 단계를 유지합니다. 소스, 스트리밍, 결과 처리는
[예측](/docs/predict)을 참조합니다.

## 변형

크기는 두 가지이며 모두 특징 피라미드가 있는 ResNet-50을 사용합니다. `r50`은 원본
헤드이고 `r50v2`는 이를 GroupNorm 헤드와 FPN 출력 대신 백본의 마지막 단계에서
입력받는 더 넓은 P6 블록으로 대체합니다.

## 검증

`val()`은 학습에 사용한 형식의 모든 데이터셋을 대상으로 측정한 정밀도, 재현율,
mAP 50, mAP 50-95를 포함하는 `metrics/` 키 딕셔너리를 반환합니다.

<code-tabs name="val" />

## 내보내기

<export-matrix />

RetinaNet은 배치 크기 1의 ONNX로만 내보냅니다. RetinaNet은 가변 크기의 종횡비
유지 입력으로 크기를 조정하므로 LibreYOLO는 전달된 값과 관계없이 `dynamic=True`를
강제하여 서로 다른 형상의 소스에 그래프가 유효하도록 유지합니다. 내보낸 `.onnx`
파일은 파일 접미사에 따라 `LibreYOLO()`로 다시 불러오며 동일한 `Results`를
반환합니다.

<code-tabs name="export" />

## 체크포인트

이 계열에서 공개된 모든 가중치 파일입니다.

<checkpoint-table />

## 라이선스

<provenance-box></provenance-box>
