---
title: 하이퍼파라미터
seo_title: LibreYOLO에서의 학습 하이퍼파라미터
description: >-
  중요한 train() 인수: epochs, batch, lr0, optimizer, EMA, autobatch, gradient
  accumulation 및 resume, 그리고 왜 기본값이 각 계열마다 다른지.
lead: >-
  모든 학습 인자는 TrainConfig 데이터 클래스의 필드입니다. 기본 클래스는 필드와 그 기본값을 정의하며, 각 모델 계열는 이를
  서브클래싱하고 게시된 레시피가 변경하는 기본값을 재정의합니다.
keywords:
  - 학습 인수
  - 학습률
  - 배치 크기
  - 자동 배치
  - 지수 이동 평균
  - 그래디언트 누적
  - 학습을 재개하다
  - 얼리 스토핑 인내
  - 앰프 bfloat16
  - 학습 구성 YAML
last_verified: 1.5.0
snippets:
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        results = model.train(
            data="my-dataset.yaml",
            epochs=100,
            batch=16,
            imgsz=640,
            lr0=0.01,
        )

        print(results["best_mAP50_95"])
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          epochs=100 batch=16 imgsz=640 lr0=0.01
  defaults:
    - label: 계열의 해결된 채무 불이행을 읽으십시오
      language: python
      code: |
        from dataclasses import fields

        from libreyolo import LibreYOLO9
        from libreyolo.training.config import TrainConfig

        family_cfg = LibreYOLO9.TRAIN_CONFIG()
        base_cfg = TrainConfig()

        for f in fields(family_cfg):
            family_value = getattr(family_cfg, f.name)
            base_value = getattr(base_cfg, f.name, None)
            if not hasattr(base_cfg, f.name) or family_value != base_value:
                print(f"{f.name}: {family_value}")
    - label: CLI
      language: bash
      code: |
        # 계열 재정의를 포함하여 학습(train), val 및 예측(predict) 기본값을 출력합니다.
        libreyolo cfg
  autobatch:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # batch=-1은 GPU 메모리를 탐색하고 2의 구체적인 거듭제곱으로 결정됩니다.
        model.train(data="my-dataset.yaml", batch=-1, imgsz=640)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml batch=-1
  accumulate:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # 최적화 단계당 16개씩 4개의 마이크로 배치, 유효 배치 64.
        model.train(data="my-dataset.yaml", batch=16, nbs=64)
  resume:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # 중단된 실행의 체크포인트를 불러온 다음, 재개할지를 물어보십시오.
        model = LibreYOLO("runs/train/exp/weights/last.pt")
        model.train(data="my-dataset.yaml", epochs=100, resume=True)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=runs/train/exp/weights/last.pt \
          data=my-dataset.yaml epochs=100 resume=true
  cfg:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # yaml의 키는 TrainConfig 필드 이름입니다. 명시적인 kwargs가 우선합니다.
        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(data="my-dataset.yaml", cfg="my-recipe.yaml", epochs=50)
source_hash: d838d1abd45af40f
---

## 인수 설정

`train()`는 키워드 인수를 사용하며 CLI는 `key=value` 형식으로 동일한 이름을 사용합니다.

<code-tabs name="train" />

두 경로 모두 같은 장소에서 끝납니다. kwargs는 `TrainConfig.from_kwargs()`에 전달되며, 이 클래스는 계열의 구성 데이터 클래스를 만듭니다.

## 오타 처리 동작

`from_kwargs()`는 config에 필드가 아닌 모든 키를 제거하고 `UserWarning`라는 이름으로 내보냅니다. 그런 다음 기본값이 적용된 상태에서 학습이 시작됩니다:

```python
# 사용자 경고: 알 수 없는 학습 구성 키(무시됨): ['learning_rate']
model.train(data="my-dataset.yaml", learning_rate=0.001)
```

아무것도 실패하지 않고, 실행이 완료되며, 학습률는 호출자가 요청한 값과 달랐습니다. 새로운 레시피의 첫 번째 에포크에서 경고를 읽으십시오. CLI는 더 엄격합니다. 왜냐하면 설정이 만들어지기 전에 플래그 이름을 검증하기 때문에 잘못된 철자의 CLI 플래그는 즉시 거부되기 때문입니다.

## 계열별 기본값

`TrainConfig`는 필드와 기본 기본값을 정의합니다. 각 계열는 이를 서브클래싱하고 게시된 레시피가 변경하는 것을 재정의하므로 '기본 학습률이 무엇인가'에 대한 단일 정답은 없습니다.

기본 기본값은 `optimizer="sgd"`, `lr0=0.01`, `momentum=0.937`, `weight_decay=5e-4`, `scheduler="yoloxwarmcos"`, `epochs=300`, `batch=16`, `imgsz=640` 및 `amp=True`입니다. 계열이 그 기준에서 얼마나 멀리 이동하는지에 대한 세 가지 예는 다음과 같습니다:

| 필드 | 기초 | YOLOv9 | 디-파인 | YOLO-NAS |
|---|---|---|---|---|
| `optimizer` | `sgd` | `sgd` | `adamw` | `adamw` |
| `lr0` | `0.01` | `0.01` | `2e-4` | `5e-4` |
| `weight_decay` | `5e-4` | `5e-4` | `1e-4` | `1e-5` |
| `scheduler` | `yoloxwarmcos` | `linear` | `flat_cosine` | `cos` |
| `epochs` | `300` | `300` | `132` | `300` |
| `amp` | `True` | `True` | `False` | `False` |

D-FINE과 DEIM은 D-FINE 디코더가 활성화를 가장 큰 유한 float16 값인 65504로 제한하기 때문에 `amp=False`와 함께 제공됩니다. YOLO-NAS와 FOMO 또한 기본적으로 이를 끕니다. CLI의 `--amp` 플래그는 모든 계열에 대해 기본적으로 `True`로 설정되므로, 사용자 제공으로 간주되어 계열 기본값을 덮어씁니다; 변경하려는 의도가 아니라면 그대로 두십시오.

추측하기보다는 계열의 실제 채무 불이행을 읽기 위해:

<code-tabs name="defaults" />

## 배치 크기

`batch`는 글로벌 배치입니다. 멀티 GPU 학습에서는 각 랭크가 `batch // world_size`를 로드하므로, 전달하는 숫자는 얼마나 많은 GPU가 사용되든 관계없이 옵티마이저 단계당 이미지 수입니다. [멀티 GPU 학습](/docs/train/multi-gpu)을 참조하십시오.

`batch=-1`는 자동 배치를 켭니다. 트레이너는 학습 모드에서 모델을 실제 역전파로 두의 제곱 단위로 탐색하고, 메모리 곡선에 선을 맞추며, 전체 VRAM의 60% 내에 맞는 외삽 값보다 엄격히 작은 가장 큰 두의 제곱을 선택합니다.

<code-tabs name="autobatch" />

역전파와 함께 학습 모드에서 프로빙하는 것이 핵심입니다: 추론 모드의 프로브는 보존된 활성화와 그래디언트 텐서를 놓치는데, 깊은 CNN의 경우 이들은 추론 시 점유량보다 몇 배 더 많습니다. RF-DETR은 목표 비율을 45%로 낮추는데, 이는 프로브의 합성 역전파가 여전히 기준과 보조 디코더 층이 소모하는 비용을 과소평가하기 때문입니다.

Autobatch는 CUDA 기능입니다. CPU나 MPS에서는 한 줄을 기록하고 기본 배치를 유지합니다.

## 그래디언트 누적

`nbs`는 명목상의 또는 실제 배치 크기를 설정합니다. 트레이너는 옵티마이저 단계당 `round(nbs / batch)` 마이크로 배치를 누적합니다.

<code-tabs name="accumulate" />

기본값인 `None`로 두면 누적은 꺼져 있고 학습은 변경되지 않습니다.

## 학습률과 스케줄

`lr0`는 초기 학습률이고 `optimizer`는 `sgd`, `adam` 및 `adamw`를 허용합니다. `momentum`는 SGD 모멘텀 또는 Adam의 beta1이고, `weight_decay`는 L2 항이며, `nesterov`는 SGD에 적용됩니다.

일정은 `scheduler`, `warmup_epochs`, `warmup_lr_start` 및 `min_lr_ratio`에 의해 형성됩니다. `no_aug_epochs`는 강한 증강 없이 최종 에포크가 몇 번 실행될지를 설정하며, 여러 일정이 꼬리를 형성하기 위해 이것을 사용하기도 하므로 순수하게 증강 조절기만은 아닙니다. 각 계열이 증강 부분으로 무엇을 하는지는 [Augmentations](/docs/train/augmentations)에서 확인할 수 있습니다.

일부 가정에서는 자체 학습률 조절 장치를 추가합니다. `backbone_lr_mult`는 백본 그룹을 헤드에 맞춰 조정하고, `clip_max_norm`는 그래디언트 클리핑을 설정하며, SegFormer는 `head_lr_mult`를 사용하여 디코드 헤드를 백본 속도의 10배로 실행합니다. 이들은 기본 클래스가 아닌 가정의 구성 서브클래스에 존재합니다.

## EMA

`ema=True`는 학습된 가중치와 함께 가중치의 지수 이동 평균을 유지합니다. FOMO를 제외한 모든 곳에서 기본적으로 켜져 있습니다.

`ema_decay`은 목표 감쇠입니다. 감쇠는 목표에서 바로 시작하지 않고 점차 증가합니다: 업데이트 시점 `n`에서의 유효 값은 `ema_decay * (1 - exp(-n / tau))`이며, `tau`는 기본값이 2000으로 설정되어 있어 초기 업데이트는 모델을 더 가깝게 추적하고 나중 업데이트는 이를 부드럽게 만듭니다. 계열 기본값은 YOLO-NAS pose의 `0.997`에서부터 YOLOX의 `0.9998`, 그리고 YOLOv9와 DETR 계열의 `0.9999`까지 다양합니다.

EMA 가중치는 검증되는 것이며 `best.pt`와 `last.pt`가 운반하는 것입니다. 원시 학습 가중치도 `train_model` 키 아래에 저장되어 있어, 재개 시 평균이 아니라 학습된 경로에서 계속 진행됩니다.

## 정확성

`amp=True`는 CUDA 오토캐스트 하에서 순방향 패스를 실행합니다. `amp_dtype`는 `float16`(기본값) 또는 `bfloat16`를 선택합니다; `fp16`와 `bf16`는 허용되는 철자입니다.

Float16는 동적 손실 스케일링이 필요하며 라이브 `GradScaler`를 받습니다. Bfloat16의 더 넓은 지수 범위는 필요하지 않아서 스케일러가 구성되지만 비활성화되어, 옵티마이저 경로를 동일하게 유지합니다. bfloat16을 지원하지 않는 CUDA 장치에서 bfloat16을 요청하면 조용히 성능이 저하되는 대신 설정 시 오류가 발생합니다.

## 출력, 체크포인트 및 중지

실행은 `project/name`에 작성됩니다. `project`은 모든 곳에서 기본적으로 `runs/train`로 설정되어 있지만, `name`는 계열별 오버라이드 중 하나입니다: 기본 기본값은 `exp`이며, YOLOv9는 `yolo9_exp`를 사용하고 D-FINE은 `dfine_exp`를 사용합니다. `exist_ok=False`를 사용할 경우, 기본 설정으로 기존 디렉터리는 덮어쓰여지는 대신 번호가 증가된 접미사가 붙습니다.

`save_period`는 각 epoch 후 `weights/last.pt` 외에 N epoch마다 추가로 `weights/epoch_<N>.pt`를 기록하고, 추적된 지표가 개선될 때마다 `weights/best.pt`도 기록합니다. `eval_interval`는 검증이 얼마나 자주 수행되는지 설정하며, `patience`는 개선 없이 지정된 epoch 수가 지나면 실행을 중단하고, `0`는 조기 종료를 비활성화합니다.

`cache`는 디코딩된 이미지를 RAM(`True` 또는 `"ram"`)에 보관하거나 소스(`"disk"`) 옆에 `.npy` 파일로 저장하여 반복 에폭 속도를 높입니다. 캐시된 읽기는 새로 읽은 것과 바이트 단위로 동일합니다. 데이터로더 작업자와 함께, `"disk"`가 두 가지 중 더 안전합니다.

## 이력서

`resume=True`는 중단된 실행을 계속합니다. 먼저 체크포인트를 로드해야 합니다. 왜냐하면 재개는 별도의 인수가 아니라 모델에서 그것을 읽기 때문입니다.

<code-tabs name="resume" />

Resume는 학습된 가중치, 옵티마이저 상태, EMA 가중치 및 업데이트 수, 최고 메트릭 추적, `GradScaler` 스케일, 그리고 PyTorch, CUDA 및 NumPy 랜덤 상태를 복원합니다. 이는 체크포인트의 에포크에 1을 더한 시점에서 시작하며, 스케줄을 해당 위치로 빠르게 진행합니다.

두 가지는 하지 않습니다. `resume=True`은 `pretrained`와 결합할 수 없으며, 이는 오류를 발생시킵니다. 그리고 체크포인트의 최상의 메트릭 키가 현재 실행의 것과 다를 경우, 최상의 메트릭 추적은 값을 비교하지 않고 경고와 함께 0으로 재설정됩니다.

## 파일 속의 레시피

`cfg=`는 `TrainConfig` 필드 이름의 YAML 매핑을 로드하고 이를 명시적 키워드 인수 아래에 병합하므로, 키워드 인수가 항상 파일보다 우선합니다.

<code-tabs name="cfg" />

`size`와 `num_classes`는 모델 인스턴스가 이미 소유하고 있기 때문에 파일에서 제거되었습니다. CLI에는 `--cfg` 플래그가 없으며, 파일 경로는 Python 인수입니다.

## 관련된

- [데이터셋](/docs/train/datasets) 은 `data=`가 받는 것입니다.
- 증강 조절기용 [증강](/docs/train/augmentations)과 어떤 계열이 이를 지원하는지.
- [레이어 동결](/docs/train/layer-freezing)과 [LoRA](/docs/train/lora)를 사용하여 일부 가중치를 학습합니다.
- [실행 보고서의 검증 및 지표](/docs/train/validation)
