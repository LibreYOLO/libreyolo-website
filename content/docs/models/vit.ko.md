---
title: ViT
families:
  - vit
seo_title: 'ViT: LibreYOLO에서 고전 Vision Transformer 분류기 실행'
description: >-
  LibreYOLO로 ViT 분류기를 예측, 검증, 내보내기합니다. Apache-2.0 AugReg 가중치를 사용하며 파인튜닝은 아직 지원하지
  않습니다.
lead: >-
  고전 Vision Transformer는 학습된 클래스 토큰과 고정 크기 이미지 패치에 적용되는 순수 트랜스포머이며 컨볼루션을 사용하지
  않습니다. LibreYOLO는 이미지 분류용 AugReg 사전 학습 크기 4개를 제공합니다.
keywords:
  - ViT 사용법
  - Vision Transformer
  - AugReg
  - 이미지 분류
  - 트랜스포머 분류기
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreViTti-cls.pt")
        result = model(SAMPLE_IMAGE, save=True)

        probs = result.probs
        print(probs.top1, probs.top1conf)
        print(probs.top5, probs.top5conf)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreViTti-cls.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreViTti-cls.pt")

        # data는 데이터셋 YAML이 아니라 train/ 및 val/ 클래스 폴더 분할이 있는
        # 디렉터리 루트(ImageFolder 레이아웃)입니다.
        metrics = model.val(data="imagenet-1k/")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreViTti-cls.pt data=imagenet-1k/
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreViTti-cls.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreViTti-cls.pt format=onnx
        libreyolo export model=LibreViTti-cls.pt format=tensorrt half=True
    - label: 내보낸 파일 사용하기
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 팩토리는 파일 접미사로 라우팅하므로 내보낸 아티팩트도
        # 체크포인트처럼 불러오며 동일한 Results 객체를 반환합니다.
        model = LibreYOLO("LibreViTti-cls.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.probs.top1)
source_hash: f63e98454913765a
---

## 설치

ViT에는 선택적 extra가 필요하지 않습니다. 가져오는 모든 항목이 기본 설치에
포함됩니다.

```bash
pip install libreyolo
```

## 예측

가중치는 처음 사용할 때 Hugging Face에서 내려받아 로컬에 캐시됩니다.

<code-tabs name="predict" />

분류기는 `result.boxes` 대신 `result.probs`를 반환합니다. `top1`과 `top5`는
클래스 인덱스를, `top1conf`와 `top5conf`는 해당 신뢰도를 제공합니다. 전처리는
timm의 AugReg 평가 레시피에 따라 0.9 크롭 비율의 bicubic 보간을 사용하여 고정
224px 입력으로 크기를 조정하고 중앙 크롭합니다. 소스, 스트리밍, 결과 처리는
[예측](/docs/predict)을 참조합니다.

## 변형

tiny부터 large까지 크기 4개가 있으며 모두 고정 224px patch-16 그래프를 공유하고
임베딩 너비와 트랜스포머 깊이가 다릅니다. LibreYOLO는 이 계열을 추론 전용으로
제공합니다. 예측, ImageNet 방식 top-1/top-5 검증, 내보내기를 지원하며 AugReg
파인튜닝 레시피는 구현되지 않았습니다.

## 검증

`val()`은 ImageFolder 방식 분할(`train/` 및 `val/` 하위 폴더와 클래스별 폴더가
있는 디렉터리)을 대상으로 실행하고 top-1 및 top-5 정확도를 반환합니다.

<code-tabs name="val" />

## 내보내기

<export-matrix />

내보낸 아티팩트는 파일 접미사에 따라 `LibreYOLO()`로 다시 불러오므로 `.onnx` 또는
`.engine` 파일이 체크포인트처럼 동작하며 동일한 `Results`를 반환합니다.
[내보내기](/docs/export)에는 모든 형식이 허용하는 인수와 일부 형식이 추가하는 extra가
나열되어 있습니다.

<code-tabs name="export" />

## 체크포인트

이 계열에서 공개된 모든 가중치 파일입니다.

<checkpoint-table />

## 라이선스

<provenance-box></provenance-box>

## 인용

<citation-block />
