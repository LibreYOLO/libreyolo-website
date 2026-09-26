---
title: ConvNeXt
families:
  - convnext
seo_title: 'ConvNeXt: Apache-2.0으로 학습, 검증, 내보내기'
description: >-
  LibreYOLO에서 이미지 분류에 ConvNeXt를 사용합니다. LibreConvNeXt tiny/small/base를 설치하고 예측,
  LoRA 파인튜닝, 검증, 내보내기합니다.
lead: >-
  ConvNeXt는 표준 컨볼루션으로만 구성된 이미지 분류기로, ResNet에서 비전 트랜스포머의 설계 선택을 향해 블록별로 현대화했습니다.
  LibreYOLO는 분류 한 작업에서 이를 지원합니다.
keywords:
  - ConvNeXt 사용법
  - ConvNeXt tiny
  - 이미지 분류
  - 순수 컨볼루션 네트워크
  - ImageNet 분류기
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreConvNeXtt-cls.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.probs.top1, result.probs.top1conf)
        print(result.probs.top5)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreConvNeXtt-cls.pt source=cat.jpg save=True
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreConvNeXtt-cls.pt")
        model.train(data="imagenette160", epochs=5)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreConvNeXtt-cls.pt data=imagenette160 epochs=5
    - label: LoRA
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreConvNeXtt-cls.pt")
        model.train(data="imagenette160", epochs=5, lora=True)
    - label: 다중 GPU
      language: bash
      code: |
        libreyolo train model=LibreConvNeXtt-cls.pt data=imagenette160 \
          epochs=50 device=0,1 batch=-1
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreConvNeXtt-cls.pt")
        metrics = model.val(data="imagenette160")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreConvNeXtt-cls.pt data=imagenette160
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreConvNeXtt-cls.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreConvNeXtt-cls.pt format=onnx
        libreyolo export model=LibreConvNeXtt-cls.pt format=tensorrt half=True
    - label: 내보낸 파일 사용하기
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 팩토리는 파일 접미사로 라우팅하므로 내보낸 아티팩트도
        # 체크포인트처럼 불러오며 동일한 Results 객체를 반환합니다.
        model = LibreYOLO("LibreConvNeXtt-cls.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.probs.top1)
source_hash: 1682cc69cf2925e6
---

## 설치

ConvNeXt에는 선택적 extra가 필요하지 않습니다. 가져오는 모든 항목이 기본 설치에
포함됩니다.

```bash
pip install libreyolo
```

`lora=True`를 사용하는 어댑터 파인튜닝은 예외이며 `lora` extra가 필요합니다.

```bash
pip install "libreyolo[lora]"
```

## 예측

가중치는 처음 사용할 때 Hugging Face에서 내려받아 로컬에 캐시됩니다.

<code-tabs name="predict" />

반환되는 `Results` 객체는 모든 계열에서 동일하므로 다른 모델로 바꾸려면 한 줄만
수정하면 됩니다. 분류기에는 바운딩 박스나 마스크가 없습니다. `result.probs`에는
전체 이미지 예측이 들어 있으며 `top1`, `top5`, `top1conf`, `top5conf`를
제공합니다. 단일 확률 벡터에는 임계값을 적용하거나 억제할 항목이 없으므로 API
동등성을 위해 `conf`, `iou`, `max_det`를 허용하지만 효과가 없습니다. 소스,
스트리밍, 결과 처리는 [예측](/docs/predict)을 참조합니다.

## 변형

tiny, small, base 크기 3개가 있으며 모두 같은 방식으로 학습하고 평가하므로 크기
선택은 매개변수 수와 정확도의 직접적인 절충입니다. 작업은 고정되어 모든 크기가
분류만 지원합니다. 모든 크기의 가중치 파일 이름은 `-cls.pt`로 끝나며 팩토리는 이
접미사를 읽어 해당 계열로 라우팅합니다. `task=` 인수는 필요하지 않습니다.

## 학습

파인튜닝은 공개된 ImageNet 백본에서 시작하며 최종 분류기 레이어를 대상 데이터셋의
클래스 수에 맞게 자동으로 다시 빌드합니다.

<code-tabs name="train" />

다른 값을 지정하지 않으면 트레이너는 AdamW, 배치 64, `lr0=1e-3`으로 100 에폭을
실행하며 50 에폭 동안 개선이 없으면 조기 종료합니다. `data`에는 데이터셋
루트(`train/` 및 `val/`, 클래스별 폴더), `imagenette160` 같은 알려진 짧은 이름
또는 `.zip` URL을 사용할 수 있습니다. ConvNeXt 블록에는 LoRA에 필요한
`nn.Linear` MLP가 있으므로 여기서는 `lora=True`를 지원하며, 전체 백본을
파인튜닝하는 대신 블록 MLP에 어댑터를 삽입합니다.

데이터셋, 증강, 다중 GPU, 로거는 [학습](/docs/train)을 참조합니다.

## 검증

`val()`은 `metrics/` 키 딕셔너리를 반환합니다. 분류에서는 검증 분할의 top-1 및
top-5 정확도입니다.

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

<provenance-box>

이 계열에는 ConvNeXt V1만 제공됩니다. ConvNeXt-V2의 작은 사전 학습 체크포인트에는
CC-BY-NC 4.0이 적용되며, 비상업용 가중치를 MIT 및 상업용 라이브러리 안에서
재배포할 수 없으므로 의도적으로 제외했습니다.

</provenance-box>

## 인용

<citation-block />
