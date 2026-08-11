---
title: LoRA 파인튜닝
seo_title: LibreYOLO에서 LoRA 미세조정
description: >-
  로라(True) 옵션으로 낮은 VRAM에서 트랜스포머 탐지기를 파인튜닝하십시오. 어떤 아홉 개의 계열가 이를 지원하는지, 계열별 어댑터
  레시피, 체크포인트가 어떻게 동작하는지.
lead: >-
  LoRA는 모델의 사전 학습된 무거운 부분을 고정시키고 그 옆에 작은 저랭크 어댑터와 반드시 밀집 상태를 유지해야 하는 층들을 학습합니다.
  LibreYOLO에서는 전체 공개 인터페이스가 하나의 불리언 값입니다.
keywords:
  - 로라 파인튜닝
  - 파라미터 효율적 파인튜닝
  - 페프트
  - 도라
  - 저 VRAM 학습
  - rf-detr 로라
  - 디-파인 로라
  - 어댑터 병합
last_verified: 1.5.0
snippets:
  install:
    - label: 파이프
      language: bash
      code: |
        pip install "libreyolo[lora]"
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs.pt")
        model.train(data="my-dataset.yaml", epochs=50, lora=True)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreRFDETRs.pt data=my-dataset.yaml \
          epochs=50 lora=true
  merge:
    - label: 내보내기는 어댑터를 병합합니다
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("runs/train/exp/weights/best.pt")
        model.export(format="onnx")
    - label: 제자리에서 병합
      language: python
      code: |
        from libreyolo import LibreYOLO
        from libreyolo.training.lora import merge_lora_adapters

        model = LibreYOLO("runs/train/exp/weights/best.pt")
        merged = merge_lora_adapters(model.model)

        print(f"{merged} adapter layers folded into dense weights")
source_hash: 603fdddf5ec0c316
---

## 설치

LoRA는 선택적 `peft` 종속성 위에서 작동합니다.

<code-tabs name="install" />

그것 없이는, `lora=True`는 실수로 전체 세부 조정을 학습하기보다는 그 명령의 이름을 `ImportError`로 올립니다.

## 사용해라

<code-tabs name="train" />

`lora=True`는 전체 인터페이스입니다. 순위(rank), 알파(alpha), 드롭아웃(dropout) 및 타깃(target) 모듈은 각 계열별로 업스트림 참조와 일치하도록 고정되어 있으며, 사용자가 조정할 수 있는 설정이 아닙니다.

LoRA를 지원하지 않는 계열는 플래그를 무시하는 대신 설정 시에 오류를 발생시킵니다:

```text
LoRA fine-tuning (lora=True) is not supported for yolo9. LoRA targets
transformer components with nn.Linear layers (e.g. RF-DETR, D-FINE, DEIM).
```

CLI는 모델이 구축되기 전에, 동일한 아홉 개 계열의 자체 허용 목록을 사용하여 이를 더 일찍 거부합니다.

## 어떤 계열들

RF-DETR, D-FINE, DEIM, DEIMv2, RT-DETR v1, v2 및 v4, EC 및 ConvNeXt. 게이트는 각 계열의 트레이너 클래스에 있는 `supports_lora` 속성이며, CLI는 일치하는 허용 목록을 가지고 있습니다.

작업 범위는 계열 범위보다 좁습니다. D-FINE과 EC는 탐지만 지원하며, 그들의 세그먼트 및 포즈 경로가 상승합니다. RF-DETR의 의미 경로가 상승합니다. ConvNeXt는 분류입니다.

다른 모든 것은 상승합니다. 부분 모드나 무음 모드는 없습니다.

## 각 레시피가 하는 일

아키텍처가 다르기 때문에 레시피도 다르며, ViT 백본에서 작동하는 레시피는 합성곱 백본에서는 붙일 것이 아무것도 없습니다.

RF-DETR는 DINOv2 백본의 attention `query`, `key` 및 `value` 투영에서 랭크 16과 알파 16으로 가중치 분해 LoRA인 DoRA를 사용하며, RF-DETR 레퍼런스와 일치합니다. ViT 백본은 고정되고, 프로젝터, 디코더 및 탐지 헤드는 정상적으로 학습을 계속합니다.

D-FINE, DEIM 및 RT-DETR v1, v2, v4는 컨볼루셔널 백본을 트랜스포머 하이브리드 인코더와 변형 가능한 디코더와 결합하므로 분할이 이동합니다. 컨볼루셔널 백본은 완전히 고정되며, 이는 역전파 과정도 건너뛰게 합니다. 트랜스포머 블록은 기본 가중치를 고정하고 선형 계층에서 동일한 랭크 16과 알파 16의 일반 LoRA 어댑터를 학습합니다: 피드포워드 `linear1` 및 `linear2`, 게이트, 그리고 변형 가능한 어텐션 프로젝션. 그 외 모든 것, 인코더 컨볼루션 융합, 입력 프로젝션, 예측 헤드 및 쿼리 임베딩은 밀집 학습을 계속합니다.

그 레시피의 두 가지 세부 사항은 의도적입니다. 디코더 자기-어텐션은 어댑터 없이 고정된 상태로 유지되는데, 이는 PyTorch의 `nn.MultiheadAttention`가 `out_proj.weight`를 직접 읽고 주입된 어댑터를 조용히 우회할 수 있기 때문입니다. 그리고 그것이 DoRA가 아닌 단순 LoRA인 이유는, 여러 디코더 선형 레이어가 설계상 0으로 초기화되어 있고 DoRA의 크기 정규화가 가중치 노름으로 나누기 때문입니다.

DEIMv2는 SwiGLU 피드포워드 계층 `w12`와 `w3`를 대상으로 동일한 레시피를 사용합니다. 그 S, M, L 및 X 크기 또한 DINOv3 ViT 백본을 가지고 있으며, 여기서 ViT 베이스는 고정되고 융합된 어텐션 `qkv` 계층은 어댑터를 받으며, Spatial Tuning Adapter 컨볼루션 피라미드는 프로젝터 아날로그로서 학습을 계속 유지합니다. 이러한 `qkv` 어댑터는 ViT가 고정된 상태로 구성되어 있더라도 적용됩니다. 왜냐하면 고정된 백본을 조정하는 것이 핵심이기 때문입니다. 하위 S 크기들은 컨볼루션 백본을 사용하고 단순 레시피를 따릅니다.

EC는 백본이 학습 가능한 컨볼루션 프로젝터 피라미드로 둘러싸인 ViT인 DETR입니다. ViT 베이스는 고정되고 그 `qkv` 레이어에는 어댑터가 추가되며, 트랜스포머 블록은 공유 레시피를 사용하고, 프로젝터와 헤드는 밀집 상태로 유지됩니다.

ConvNeXt 블록은 채널 마지막 선형 MLP, `fc1` 및 `fc2`를 포함하며, 이들은 일반 어댑터를 사용합니다. 깊이별 컨볼루션, 정규화 및 레이어 스케일 매개변수는 고정됩니다. 분류 헤드는 밀집 상태로 유지되어 사용자 정의 클래스 수가 계속 작동합니다.

탐지 및 분류 헤드는 모든 레시피에서 항상 학습 가능 상태로 유지됩니다. 이는 사용자 정의 클래스 수에 맞춰 새로 학습된 헤드가 필요하기 때문입니다.

## 체크포인트 및 내보내기

`best.pt` 및 `last.pt`는 어댑터 텐서를 유지하므로, LoRA 실행은 다른 실행처럼 다시 시작되거나 검사될 수 있습니다. 이러한 체크포인트 중 하나를 로드하려면 `lora`가 추가로 설치되어 있어야 하며, 로더가 어댑터 주입을 재생하여 키가 일치하도록 하기 때문입니다.

`export()`는 어댑터를 밀집 가중치에 병합하므로, 내보낸 산출물은 `peft`에 대한 의존성을 가지지 않습니다. 동일한 병합은 메모리 내 모델에서도 직접 사용할 수 있습니다.

<code-tabs name="merge" />

병합 후 모듈 트리가 완전히 밀집되며 두 번째 병합은 아무 작업도 하지 않습니다.

## 그것이 구하는 것과 구하지 않는 것

LoRA는 옵티마이저와 그래디언트 메모리를 줄이며, 백본을 완전히 고정하는 계열에서는 그 백본의 역전파도 건너뜁니다.

활성화 메모리는 변경되지 않습니다. 순방향 활성화는 여전히 남아 있는 학습 가능한 부분을 위해 유지되어야 하며, 이것이 보통 최대치를 결정합니다. 가장 제한적인 VRAM 예산을 위해서는 `batch` 또는 `imgsz`를 낮추십시오.

## 관련된

- [레이어 동결](/docs/train/layer-freezing)은 가중치의 일부를 학습하는 또 다른 방법으로, 모든 모델군에서 작동하며 추가 종속성이 필요하지 않습니다. `freeze`와 `lora=True`를 구성하면: 어댑터 매개변수는 상위 백본 그룹이 동결되어도 학습 가능 상태를 유지합니다.
- [하이퍼파라미터](/docs/train/hyperparameters)은 `batch`, `imgsz` 및 나머지 `train()`에 해당합니다.
