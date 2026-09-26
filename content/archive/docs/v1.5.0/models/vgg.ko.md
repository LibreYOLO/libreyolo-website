---
title: VGG
families:
  - vgg
seo_title: 'VGG: LibreYOLO에서 VGG-16/19 이미지 분류기 실행'
description: >-
  LibreYOLO로 VGG 분류기를 예측, 검증, 내보내기합니다. BSD-3-Clause torchvision 가중치를 사용하며 파인튜닝은
  아직 지원하지 않습니다.
lead: >-
  VGG는 큰 필터 대신 작은 3x3 컨볼루션의 균일한 스택으로 구성된 컨볼루션 이미지 분류기입니다. LibreYOLO는 이미지 분류용 16
  및 19 레이어 크기를 BatchNorm 적용 여부에 따라 제공합니다.
keywords:
  - VGG 사용법
  - VGG-16
  - VGG-19
  - 컨볼루션 신경망
  - 이미지 분류
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreVGG16-cls.pt")
        result = model(SAMPLE_IMAGE, save=True)

        probs = result.probs
        print(probs.top1, probs.top1conf)
        print(probs.top5, probs.top5conf)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreVGG16-cls.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreVGG16-cls.pt")

        # data는 데이터셋 YAML이 아니라 train/ 및 val/ 클래스 폴더 분할이 있는
        # 디렉터리 루트(ImageFolder 레이아웃)입니다.
        metrics = model.val(data="imagenet-1k/")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreVGG16-cls.pt data=imagenet-1k/
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreVGG16-cls.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreVGG16-cls.pt format=onnx
        libreyolo export model=LibreVGG16-cls.pt format=tensorrt half=True
    - label: 내보낸 파일 사용하기
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 팩토리는 파일 접미사로 라우팅하므로 내보낸 아티팩트도
        # 체크포인트처럼 불러오며 동일한 Results 객체를 반환합니다.
        model = LibreYOLO("LibreVGG16-cls.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.probs.top1)
source_hash: 26eb6ff5811533fd
---

## 설치

VGG에는 선택적 extra가 필요하지 않습니다. 가져오는 모든 항목이 기본 설치에
포함됩니다.

```bash
pip install libreyolo
```

## 예측

가중치는 처음 사용할 때 Hugging Face에서 내려받아 로컬에 캐시됩니다.

<code-tabs name="predict" />

분류기는 `result.boxes` 대신 `result.probs`를 반환합니다. `top1`과 `top5`는
클래스 인덱스를, `top1conf`와 `top5conf`는 해당 신뢰도를 제공합니다. 예측은 고정
224px 입력으로 실행되며 다른 `imgsz`를 전달하면 오류가 발생합니다. 소스, 스트리밍,
결과 처리는 [예측](/docs/predict)을 참조합니다.

## 변형

크기는 4가지입니다. 컨볼루션 레이어가 16개 또는 19개이며 각각 기본 변형과
BatchNorm 변형이 있습니다. 제공되는 가중치는 Oxford 그룹의 원본 2014 Caffe
릴리스를 변환한 것이 아니라 torchvision이 나중에 처음부터 ImageNet으로 학습한
가중치입니다. LibreYOLO는 이 계열을 추론 전용으로 제공합니다. 예측, ImageNet 방식
top-1/top-5 검증, 내보내기를 지원하며 파인튜닝은 구현되지 않았습니다.

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
