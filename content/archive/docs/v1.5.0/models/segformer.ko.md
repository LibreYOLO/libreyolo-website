---
title: SegFormer
families:
  - segformer
seo_title: 'SegFormer: LibreYOLO의 의미 분할'
description: >-
  LibreYOLO에서 b0-b5 크기의 SegFormer로 ADE20K 의미 분할을 수행합니다. 설치, 예측, 학습, 내보내기를 지원하며
  사전 학습 가중치는 비상업적입니다.
lead: >-
  SegFormer는 계층형 Mix Transformer(MiT) 인코더와 가벼운 all-MLP 디코딩 헤드를 결합한 의미 분할
  transformer입니다. 이전 분할 transformer에 필요했던 무거운 디코더와 고정 위치 인코딩을 피합니다. LibreYOLO는
  여섯 가지 크기의 의미 분할 작업 하나를 지원합니다.
keywords:
  - SegFormer 사용법
  - 의미 분할
  - Mix Transformer
  - MiT
  - transformer 분할
  - ADE20K
  - 조밀 예측
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreSegformerb0-sem.pt")
        result = model(SAMPLE_IMAGE, save=True)

        mask = result.semantic_mask
        print(mask.data.shape, mask.classes)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreSegformerb0-sem.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  train:
    - label: Python(파인튜닝)
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSegformerb0-sem.pt")
        model.train(data="my-dataset.yaml", epochs=160, imgsz=512, batch=8)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreSegformerb0-sem.pt data=my-dataset.yaml \
          epochs=160 imgsz=512 batch=8
    - label: 처음부터 학습
      language: python
      code: |
        from libreyolo.models.segformer.model import LibreSegformer

        # model_path가 없으면 무작위 초기화하며 아무것도 다운로드하지 않습니다.
        # 사전 학습 체크포인트의 비상업적 약관이 없는 가중치를 얻는 유일한 경로입니다.
        model = LibreSegformer(size="b0", nb_classes=150)
        model.train(data="my-dataset.yaml", epochs=160, imgsz=512, batch=8)
    - label: 다중 GPU
      language: bash
      code: |
        libreyolo train model=LibreSegformerb0-sem.pt data=my-dataset.yaml \
          epochs=160 device=0,1 batch=16
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSegformerb0-sem.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreSegformerb0-sem.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSegformerb0-sem.pt")
        model.export(format="onnx", imgsz=512)
        model.export(format="tensorrt", imgsz=512, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreSegformerb0-sem.pt format=onnx imgsz=512

        libreyolo export model=LibreSegformerb0-sem.pt format=tensorrt imgsz=512
        half=True
    - label: 내보낸 파일 사용
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 팩토리는 파일 접미사에 따라 라우팅하므로 내보낸 아티팩트도
        # 다른 체크포인트처럼 불러와 동일한 Results 객체를 반환합니다.
        model = LibreYOLO("LibreSegformerb0-sem.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.semantic_mask.data.shape)
source_hash: c236895b991beabf
---

## 설치

SegFormer에는 선택적 extra가 필요하지 않습니다. 가져오는 모든 항목이 기본 설치에 포함됩니다.

```bash
pip install libreyolo
```

## 예측

처음 사용할 때 Hugging Face에서 가중치를 다운로드해 로컬에 캐시합니다.

<code-tabs name="predict" />

`result.semantic_mask`는 조밀한 클래스 맵을 담습니다. `.data`는 원본 이미지 크기의 클래스 ID `(H, W)` 텐서이고 `.classes`는 실제로 존재하는 클래스 ID를 나열합니다. 인스턴스별 탐지가 없으므로 `result.boxes`는 `None`입니다. 모델이 필터링하거나 중복 제거할 인스턴스별 탐지 대신 픽셀마다 클래스 하나를 반환하므로 `conf`와 `iou`는 API 일관성을 위해 허용되지만 출력을 바꾸지 않습니다. 소스, 스트리밍, 결과 처리는 [예측](/docs/predict)을 참조합니다.

## 변형

b0부터 b5까지 여섯 가지 크기가 있습니다. 각 단계에서 Mix Transformer 인코더를 넓고 깊게 만들면서 같은 all-MLP 디코딩 헤드 설계를 유지합니다.

<checkpoint-table />

## 학습

기본적으로 `train()`은 공개된 체크포인트를 파인튜닝합니다. 대신 `model_path` 없이 `LibreSegformer(...)`를 호출하면 무작위로 초기화된 인코더와 헤드를 구축해 처음부터 학습합니다. 이 방법만 사전 학습 체크포인트의 비상업적 제한이 전혀 없는 가중치를 만듭니다([라이선스](#licensing) 참조).

<code-tabs name="train" />

기본 설정에서는 SegFormer 논문의 ADE20K 레시피를 따릅니다. AdamW를 백본 기본 학습률로 사용하고 디코딩 헤드는 그 10배 학습률로 학습합니다. LayerNorm과 Mix-FFN 위치 합성곱을 제외한 모든 곳에 weight decay를 적용하며 워밍업이 있는 선형 감쇠 일정을 사용합니다. b3부터 b5까지 큰 크기의 수렴은 처음부터 끝까지 검증하지 않았습니다.

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

LibreSegformer의 인코더와 디코딩 헤드는 NVlabs/SegFormer가 아니라 Hugging Face Transformers의 Apache-2.0 SegFormer 구현을 PyTorch로 이식한 것입니다. NVIDIA의 원래 저장소는 읽거나 복사하지 않았으며 논문 저자 표기를 위해서만 여기 언급합니다. 위의 사전 학습 체크포인트에만 NVIDIA의 비상업적 제한이 적용되고 구조와 LibreYOLO 자체 코드는 계속 MIT입니다.

</provenance-box>

## 인용

<citation-block />
