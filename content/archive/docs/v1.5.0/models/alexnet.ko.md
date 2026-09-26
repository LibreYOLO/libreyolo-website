---
title: AlexNet
families:
  - alexnet
seo_title: 'AlexNet: LibreYOLO에서 고전 ImageNet 분류기 실행'
description: >-
  LibreYOLO로 AlexNet을 예측, 검증, 내보내기합니다. BSD-3-Clause torchvision 가중치를 사용하며 파인튜닝은
  아직 지원하지 않습니다.
lead: >-
  AlexNet은 ILSVRC 2012에서 우승하고 컴퓨터 비전의 딥러닝 시대를 여는 데 기여한 컨볼루션 네트워크입니다. LibreYOLO는
  이미지 분류를 위해 아키텍처의 단일 타워 후기 개정판을 제공합니다.
keywords:
  - AlexNet 사용법
  - ImageNet 분류
  - 컨볼루션 신경망
  - 이미지 분류
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreAlexNetb-cls.pt")
        result = model(SAMPLE_IMAGE, save=True)

        probs = result.probs
        print(probs.top1, probs.top1conf)
        print(probs.top5, probs.top5conf)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreAlexNetb-cls.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreAlexNetb-cls.pt")

        # data는 데이터셋 YAML이 아니라 train/ 및 val/ 클래스 폴더 분할이 있는
        # 디렉터리 루트(ImageFolder 레이아웃)입니다.
        metrics = model.val(data="imagenet-1k/")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreAlexNetb-cls.pt data=imagenet-1k/
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreAlexNetb-cls.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreAlexNetb-cls.pt format=onnx
        libreyolo export model=LibreAlexNetb-cls.pt format=tensorrt half=True
    - label: 내보낸 파일 사용하기
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 팩토리는 파일 접미사로 라우팅하므로 내보낸 아티팩트도
        # 체크포인트처럼 불러오며 동일한 Results 객체를 반환합니다.
        model = LibreYOLO("LibreAlexNetb-cls.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.probs.top1)
source_hash: 68c09f080c74bb87
---

## 설치

AlexNet에는 선택적 extra가 필요하지 않습니다. 가져오는 모든 항목이 기본 설치에
포함됩니다.

```bash
pip install libreyolo
```

## 예측

가중치는 처음 사용할 때 Hugging Face에서 내려받아 로컬에 캐시됩니다.

<code-tabs name="predict" />

분류기는 `result.boxes` 대신 `result.probs`를 반환합니다. `top1`과 `top5`는
클래스 인덱스를, `top1conf`와 `top5conf`는 해당 신뢰도를 제공합니다. 소스,
스트리밍, 결과 처리는 [예측](/docs/predict)을 참조합니다.

## 변형

크기는 하나입니다. 제공되는 그래프는 torchvision이 출시한 후기 단일 타워
개정판으로, 첫 레이어 필터가 64개이며 로컬 응답 정규화가 없습니다. 원본 2012년
2-GPU 아키텍처가 아닙니다. LibreYOLO는 이 계열을 추론 전용으로 제공합니다. 예측,
ImageNet 방식 top-1/top-5 검증, 내보내기를 지원하며 파인튜닝은 구현되지 않았습니다.

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

