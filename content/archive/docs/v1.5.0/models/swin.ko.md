---
title: Swin Transformer
families:
  - swin
seo_title: 'Swin Transformer: LibreYOLO의 LibreSwin으로 이미지 분류'
description: >-
  LibreYOLO로 Swin Transformer 분류기를 예측, 검증, 내보내기합니다. MIT 가중치를 사용하며 파인튜닝은 아직 지원하지
  않습니다.
lead: >-
  Swin Transformer V1은 전체 이미지가 아니라 이동된 로컬 윈도우 안에서 어텐션을 계산하는 계층적 비전 트랜스포머입니다.
  LibreYOLO는 이미지 분류용 크기 4개를 제공합니다.
keywords:
  - Swin Transformer 사용법
  - 계층적 비전 트랜스포머
  - 이동 윈도우 어텐션
  - 이미지 분류
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreSwint-cls.pt")
        result = model(SAMPLE_IMAGE, save=True)

        probs = result.probs
        print(probs.top1, probs.top1conf)
        print(probs.top5, probs.top5conf)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreSwint-cls.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSwint-cls.pt")

        # data는 데이터셋 YAML이 아니라 train/ 및 val/ 클래스 폴더 분할이 있는
        # 디렉터리 루트(ImageFolder 레이아웃)입니다.
        metrics = model.val(data="imagenet-1k/")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreSwint-cls.pt data=imagenet-1k/
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSwint-cls.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreSwint-cls.pt format=onnx
        libreyolo export model=LibreSwint-cls.pt format=tensorrt half=True
    - label: 내보낸 파일 사용하기
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 팩토리는 파일 접미사로 라우팅하므로 내보낸 아티팩트도
        # 체크포인트처럼 불러오며 동일한 Results 객체를 반환합니다.
        model = LibreYOLO("LibreSwint-cls.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.probs.top1)
source_hash: faa6bbacae62d88e
---

## 설치

Swin에는 선택적 extra가 필요하지 않습니다. 가져오는 모든 항목이 기본 설치에
포함됩니다.

```bash
pip install libreyolo
```

## 예측

가중치는 처음 사용할 때 Hugging Face에서 내려받아 로컬에 캐시됩니다.

<code-tabs name="predict" />

분류기는 `result.boxes` 대신 `result.probs`를 반환합니다. `top1`과 `top5`는
클래스 인덱스를, `top1conf`와 `top5conf`는 해당 신뢰도를 제공합니다. 마지막
어텐션 단계가 해당 해상도용으로 만들어졌으므로 모든 크기는 224px 입력으로
고정됩니다. 다른 `imgsz`를 전달하면 예측, 검증, 내보내기에서 모두 오류가
발생합니다. 소스, 스트리밍, 결과 처리는 [예측](/docs/predict)을 참조합니다.

## 변형

tiny부터 large까지 크기 4개가 있으며 같은 이동 윈도우 타워로 구성되고 임베딩
너비와 단계 깊이가 다릅니다. large는 ImageNet-22k로 사전 학습하고 ImageNet-1k로
파인튜닝했으며, 나머지 3개는 ImageNet-1k에서 직접 학습했습니다. LibreYOLO는 이
계열을 추론 전용으로 제공합니다. 예측, ImageNet 방식 top-1/top-5 검증,
내보내기를 지원하며 업스트림 ImageNet 학습 레시피는 구현되지 않았습니다.

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
