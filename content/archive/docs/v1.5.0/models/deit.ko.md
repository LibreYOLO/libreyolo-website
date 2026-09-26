---
title: DeiT
families:
  - deit
seo_title: 'DeiT 이미지 분류기: 예측, 검증, 내보내기'
description: >-
  LibreYOLO에서 DeiT 이미지 분류기를 실행합니다. Apache-2.0이 적용된 tiny, small, base 크기의 고정된 추론
  전용 박물관 계열입니다.
lead: >-
  DeiT(Data-efficient image Transformer)는 추가 사전 학습 데이터 없이 ImageNet-1k만으로 학습한 일반
  Vision Transformer 분류기입니다. LibreYOLO는 tiny, small, base patch-16 크기를 고정된 추론 전용
  전시물로 제공합니다.
keywords:
  - DeiT 사용법
  - Vision Transformer
  - ViT
  - 이미지 분류
  - ImageNet
  - 데이터 효율 학습
  - 이전 이미지 분류 모델
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDeiTb-cls.pt")
        result = model(SAMPLE_IMAGE)

        print(result.probs.top1, result.probs.top1conf)
        print(result.probs.top5)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreDeiTb-cls.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDeiTb-cls.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDeiTb-cls.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDeiTb-cls.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreDeiTb-cls.pt format=onnx
        libreyolo export model=LibreDeiTb-cls.pt format=tensorrt half=True
    - label: 내보낸 파일 사용하기
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 팩토리는 파일 접미사로 라우팅하므로 내보낸 아티팩트도
        # 체크포인트처럼 불러오며 동일한 Results 객체를 반환합니다.
        model = LibreYOLO("LibreDeiTb-cls.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.probs.top1)
source_hash: 9c67c8554b2af5c6
---

## 설치

DeiT에는 기본 패키지 외에 추가 항목이 필요하지 않습니다.

```bash
pip install libreyolo
```

## 예측

이 계열은 추론 전용입니다. `train()`이 `NotImplementedError`를 일으키므로 이
페이지에는 학습 절이 없습니다. 예측, 검증, 내보내기는 모두 지원합니다. 가중치는
처음 사용할 때 Hugging Face에서 내려받아 로컬에 캐시됩니다. 파일 이름의 `-cls`
접미사는 필수이며 분류 작업을 선택합니다.

<code-tabs name="predict" />

반환되는 `Results` 객체에는 `boxes` 대신 `probs` 텐서가 들어 있습니다. `top1`과
`top5`는 ImageNet-1k 클래스 1,000개를 인덱싱하며 `top1conf`는 최상위 예측의
softmax 점수입니다. 각 크기는 위치 임베딩에 따른 고정 입력 해상도를 사용합니다.
전처리는 해당 크기로 조정한 뒤 중앙 크롭하며, 다른 `imgsz`를 전달하면 조용히 다시
샘플링하지 않고 오류가 발생합니다. 소스, 스트리밍, 결과 처리는
[예측](/docs/predict)을 참조합니다.

## 검증

`val()`은 일반적인 `train/<class>/` 및 `val/<class>/` 폴더 구조로 배치된
데이터셋을 대상으로 측정한 top-1 및 top-5 정확도 딕셔너리를 반환합니다.

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

## 인용

<citation-block />

