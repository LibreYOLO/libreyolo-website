---
title: LingBot-Vision
families:
  - lingbotvision
seo_title: 'LingBot-Vision: LibreYOLO의 의미 분할'
description: >-
  LibreYOLO에서 Apache-2.0 ViT 백본 기반 LingBot-Vision으로 의미 분할을 수행합니다. s/b/l 크기를 설치하고
  예측, 학습, 검증, 내보내기합니다.
lead: >-
  LingBot-Vision은 조밀한 공간 인식을 위해 경계 중심 마스킹 모델링으로 자기 지도 학습한 vision transformer 백본
  계열으로 Robbyant가 공개했습니다. LibreYOLO는 백본에 조밀 헤드를 결합하여 의미 분할 작업 하나를 지원합니다.
keywords:
  - LingBot-Vision 사용법
  - 의미 분할
  - vision transformer
  - 자기 지도 사전 학습
  - 경계 모델링
  - Robbyant
  - 조밀 예측
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreLingBotVisions-sem.pt")
        result = model(SAMPLE_IMAGE, save=True)

        mask = result.semantic_mask
        print(mask.data.shape, mask.classes)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreLingBotVisions-sem.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  train:
    - label: Python(선형 프로브)
      language: python
      code: |
        from libreyolo import LibreYOLO

        # 업스트림 평가 프로토콜과 일치하도록 기본적으로 백본을 동결합니다.
        # 1x1 조밀 헤드만 학습합니다.
        model = LibreYOLO("LibreLingBotVisions-sem.pt")
        model.train(data="my-dataset.yaml", epochs=20, imgsz=512, batch=16)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreLingBotVisions-sem.pt data=my-dataset.yaml \
          epochs=20 imgsz=512 batch=16
    - label: 전체 파인튜닝
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreLingBotVisions-sem.pt")
        model.train(
            data="my-dataset.yaml", epochs=20, imgsz=512, batch=16,
            freeze_backbone=False,
        )
    - label: 다중 GPU
      language: bash
      code: |
        libreyolo train model=LibreLingBotVisions-sem.pt data=my-dataset.yaml \
          epochs=20 device=0,1 batch=32
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreLingBotVisions-sem.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreLingBotVisions-sem.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreLingBotVisions-sem.pt")
        model.export(format="onnx", imgsz=512)
        model.export(format="coreai", imgsz=512)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreLingBotVisions-sem.pt format=onnx imgsz=512
    - label: 내보낸 파일 사용
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 팩토리는 파일 접미사에 따라 라우팅하므로 내보낸 아티팩트도
        # 다른 체크포인트처럼 불러와 동일한 Results 객체를 반환합니다.
        model = LibreYOLO("LibreLingBotVisions-sem.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.semantic_mask.data.shape)
source_hash: c47b33fdc6fa1139
---

## 설치

LingBot-Vision에는 선택적 extra가 필요하지 않습니다. 가져오는 모든 항목이 기본 설치에 포함됩니다.

```bash
pip install libreyolo
```

## 예측

처음 사용할 때 Hugging Face에서 가중치를 다운로드해 로컬에 캐시합니다.

<code-tabs name="predict" />

`result.semantic_mask`는 조밀한 클래스 맵을 담습니다. `.data`는 원본 이미지 크기의 클래스 ID `(H, W)` 텐서이고 `.classes`는 실제로 존재하는 클래스 ID를 나열합니다. 인스턴스별 탐지가 없으므로 `result.boxes`는 `None`입니다. 모델이 필터링할 탐지 대신 픽셀마다 클래스 하나를 반환하므로 `conf`와 `iou`는 API 일관성을 위해 허용되지만 출력을 바꾸지 않습니다. 소스, 스트리밍, 결과 처리는 [예측](/docs/predict)을 참조합니다.

## 변형

공개된 s, b, l 세 크기는 매개변수 11억 개의 ViT-g/16 teacher에서 증류했습니다. `g` 크기 teacher 자체도 LibreYOLO에서 불러오고 파인튜닝할 수 있지만 LibreYOLO는 자체 `g` 체크포인트를 호스팅하지 않습니다.

<checkpoint-table />

## 학습

`train()`은 공개된 체크포인트를 파인튜닝합니다. 기본 레시피는 업스트림 보고서의 선형 프로브입니다. ViT 백본을 동결하고 1x1 조밀 헤드만 학습하며 위 LibreYOLO 호스팅 가중치가 생성된 방식과 일치합니다. 전체 네트워크를 파인튜닝하려면 `freeze_backbone=False`를 전달하고 그에 맞춰 `lr0`를 낮춥니다.

<code-tabs name="train" />

데이터셋, 증강, 다중 GPU, 로거는 [학습](/docs/train)을 참조합니다.

## 검증

`val()`은 학습에 사용한 형식의 데이터셋에서 측정한 mIoU와 픽셀 정확도의 `metrics/` 키 사전을 반환합니다.

<code-tabs name="val" />

## 내보내기

<export-matrix />

내보낸 아티팩트는 파일 접미사에 따라 `LibreYOLO()`로 다시 불러오므로 `.onnx` 또는 `.engine` 파일은 체크포인트처럼 동작하며 동일한 `Results`를 반환합니다. 각 형식이 받는 인수는 [내보내기](/docs/export)에 나와 있습니다.

<code-tabs name="export" />

## 체크포인트

이 계열에 공개된 모든 가중치 파일입니다.

<checkpoint-table />

## 라이선스

<provenance-box>

업스트림 릴리스는 해당 ViT가 Meta AI에서 공개한 DINOv2/DINOv3 구조를 기반으로 구축되었다고 설명합니다. Robbyant는 구현을 Apache-2.0으로 배포하며 이 LibreYOLO 이식은 Meta의 DINOv2 또는 DINOv3 코드가 아니라 Robbyant 저장소만 사용해 만들었습니다.

</provenance-box>

## 인용

<citation-block />
