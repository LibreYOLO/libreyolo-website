---
title: 지식 증류
seo_title: LibreYOLO에서의 지식 증류
description: >-
  작은 검출기를 더 큰 교사나 고정된 DINOv2 백본에 대해 학습시키기: MGD, CWD 및 feature-MSE 손실, 탭 포인트, 그리고
  계열 지원.
lead: >-
  증류(distillation)는 학생(student)의 중간 특징 맵을 고정된(frozen) 교사(teacher)의 특징 맵 쪽으로
  끌어당기는 두 번째 손실 항을 추가합니다. LibreYOLO는 포워드 훅(forward hooks)으로 특징을 활용하므로, 교사의 자체
  헤드(head)와 손실(loss)은 절대 사용되지 않습니다.
keywords:
  - 지식 증류
  - 마스크 생성 증류
  - 채널 단위 증류
  - 특징 증류
  - 다이노V2 교사
  - 교사 학생 학습
  - mgd 손실
  - cwd 손실
last_verified: 1.5.0
snippets:
  detector:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # 같은 계열의 더 큰 검문소가 작은 검문소를 감독합니다.
        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=100,
            distill_model="LibreYOLO9c.pt",
            distill_loss_type="mgd",
        )
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          epochs=100 distill_model=LibreYOLO9c.pt distill_loss_type=mgd
  foundation:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # 동결된 자기지도식 ViT가 하나의 백본 단계를 감독합니다.
        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=100,
            distill_model="dinov2",
        )
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          epochs=100 distill_model=dinov2
  tuned:
    - label: 손실 조정
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(
            data="my-dataset.yaml",
            distill_model="LibreYOLO9c.pt",
            distill_loss_type="cwd",
            dis=1.0,           # 글로벌 증류 가중치
            distill_tau=1.0,   # CWD 소프트맥스 온도
        )
source_hash: 7210031328f6826f
---

## 더 큰 체크포인트에서 증류

`distill_model` 설정은 증류를 켭니다. 값은 교사 체크포인트로, 다른 모델과 동일한 방식으로 팩토리를 통해 로드됩니다.

<code-tabs name="detector" />

교사는 `no_grad` 아래에서 순전파를 수행하며, AMP가 켜져 있을 때 자동 캐스트 아래에서 수행하므로, 고정된 모델은 매 단계에서 전체 정밀도 연산을 수행하지 않습니다. 순전파 훅은 명명된 탭 포인트에서 특징 맵을 캡처하고, 손실은 이를 학생의 특징 맵과 비교하며, 그 결과는 학습 손실에 추가되고 `distill`라는 구성 요소로 보고됩니다.

## 얼어붙은 기초 골격에서 증류

자기 지도 ViT는 대신 단일 학생 백본 단계를 감독할 수 있습니다. 교사의 특징은 후크가 아닌 자체 특징 추출기에서 나오며, 손실은 패치 그리드와 컨볼루션 스트라이드 간의 불일치를 처리합니다.

<code-tabs name="foundation" />

`distill_model`는 `dinov2`를 인식하며, 이는 DINOv2-base이고, 또한 `dinov2_vits14`, `dinov2_vitb14`, `dinov2_vitl14`, `dinov2-small`, `dinov2-base`, `dinov2-large`와 `facebook/dinov2`로 시작하는 모든 원시 허브 ID를 포함합니다. 그 외의 모든 것은 교사 체크포인트 경로로 처리됩니다.

이 경로는 `distill_loss_type`와 관계없이 `feat_mse`를 사용하며, `transformers`가 설치되어 있어야 합니다. 누락된 가중치 키로 로드되는 교사는 부분적으로 무작위 백본에 대해 증류하기보다는 중단됩니다.

## 어떤 계열들

증류 지원은 학생 모델에 대한 방법이며, 그것에는 두 가지가 있습니다.

`get_distill_config()`는 탐지기 교사가 감독하는 다중 스케일 탭 포인트를 제공합니다. YOLOv9, YOLOX 및 RF-DETR이 이를 구현합니다.

`get_backbone_distill_config()`는 단일 백본 단계를 제공하며 교사가 감독합니다. YOLOv9가 이를 구현하며, 이것이 유일한 계열입니다.

손실 없이 학습하는 것보다 다른 것은 더 많이 증가합니다:

```text
LibreDFINE does not implement get_distill_config(). Distillation is not yet
supported for the 'dfine' family.
```

```text
Foundation-model distillation into the 'yolox' family is not supported yet
(no get_backbone_distill_config()).
```

## 탭 포인트

탭 포인트는 계열별 및 역할별로 고정되어 있으므로, 교사와 학생이 동일한 아키텍처일 필요는 없습니다; 그들은 일치하는 특징 스트라이드가 필요합니다.

| 계열 | 역할 | 탭 포인트 | 진보 |
|---|---|---|---|
| YOLOv9 | 교사 또는 학생 | `neck.elan_up2`, `neck.elan_down1`, `neck.elan_down2` | 8, 16, 32 |
| YOLOv9 | 파운데이션 학생 | `backbone.elan3` | 16 |
| YOLOX | 교사 또는 학생 | `backbone.C3_p3`, `backbone.C3_n3`, `backbone.C3_n4` | 8, 16, 32 |
| RF-DETR | 교사 또는 학생 | `model.backbone.0.projector.stages.0` | 설정에서 탐지됨 |

학습 시작 전에 보폭이 맞지 않음:

```text
Teacher and student must have matching strides. Teacher: [8, 16, 32],
Student: [16]
```

그 검사는 격자가 다르다는 것이 핵심인 기초 교사들에게는 생략됩니다.

## 세 번의 상실

`distill_loss_type`는 탐지기 교사를 위한 특징 손실을 선택합니다. 기본 교사는 항상 `feat_mse`를 사용합니다.

`mgd`, 마스킹된 생성 증류(masked generative distillation)는 학생의 공간 위치 일부를 마스킹하고 작은 두 개의 합성곱 생성기를 학습시켜 남은 것으로부터 교사의 전체 특징 맵을 재구성합니다. `distill_mask_ratio`는 마스킹된 비율을 설정하며, 기본값은 0.65입니다.

`cwd`, 채널별 증류는 각 채널의 공간적 활성화를 확률 분포로 변환하고 채널별로 KL 발산을 최소화합니다. `distill_tau`는 소프트맥스 온도이며 기본값은 1.0입니다.

`feat_mse`는 1x1 컨볼루션으로 학생의 채널을 교사의 채널에 맞추고, 교사의 그리드를 학생의 그리드에 맞게 바이리니어 방식으로 리사이즈한 후, 평균 제곱 오차를 계산합니다. `distill_normalize=True`는 먼저 채널 차원에서 두 피처 맵을 L2 정규화하여, 매치를 각도만 고려하고 스케일에 영향을 받지 않도록 만듭니다. 기본값은 `False`입니다.

`dis`는 위에 적용되는 전역 가중치입니다. 설정하지 않으면 각 손실은 자체적으로 발표된 기본값을 사용합니다: MGD의 경우 2e-5, CWD와 feature MSE의 경우 1.0입니다. 이 값들은 10만 배 차이가 나므로, 한 손실 유형에 맞춰 조정된 가중치는 다른 손실에는 의미가 없습니다.

<code-tabs name="tuned" />

`distill_mask_ratio`, `distill_tau` 및 `distill_normalize`에는 CLI 플래그가 없습니다. 이들은 Python 인수이거나 `cfg=` YAML 키입니다. RF-DETR 또한 전체적으로 증류를 위해 Python 전용이며, 그 이유는 CLI 인수 매핑이 증류 키를 포함하지 않기 때문입니다.

## 어댑터, 체크포인트 및 멀티 GPU

모든 손실은 학생 외부에 존재하는 작은 학습 가능한 모듈을 만듭니다: 1x1 채널 어댑터와 MGD의 생성기입니다. 이들은 실행의 유효 학습률에서 자체 옵티마이저 파라미터 그룹을 갖습니다.

해당 모듈들은 `distiller` 키 아래 체크포인트에 기록되고 재개 시 복원되므로, 재개된 실행은 프로젝터를 처음부터 다시 시작하지 않습니다.

DDP에서는 어댑터가 포장된 학생 바깥에 위치하므로 DDP 리듀서는 이들의 기울기를 전혀 보지 못합니다. 트레이너는 매 스텝마다 이를 명시적으로 올리듀스(all-reduce)하므로, 모든 랭크는 동일한 어댑터를 학습합니다.

CUDA 그래프 캡처는 증류 실행에서는 사용할 수 없습니다. `cuda_graph=True`를 전달하면 한 줄씩 로그가 기록되고 즉시 학습이 진행됩니다. [학습 성능](/docs/train/performance)을 참조하십시오.

## 관련된

- [레이어 동결](/docs/train/layer-freezing)과 [LoRA 파인튜닝](/docs/train/lora), 어느 것도 증류와 결합되는 것이 차단되지 않습니다.
- 나머지 `train()`에 대한 [하이퍼파라미터](/docs/train/hyperparameters).
