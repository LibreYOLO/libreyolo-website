---
title: libreyolo train
seo_title: libreyolo train 명령 참조
description: '명령줄에서 모델을 학습합니다: 59개 인자 전체와 각각의 기본값, 모델 계열 기본값이 이를 덮어쓰는 방식, 그리고 계열이 무시하는 인자.'
lead: >-
  하나의 데이터셋에서 하나의 모델을 학습하고 체크포인트, 지표, 로그를 실행 디렉터리에 기록합니다. 아래의 모든 인자에는 명령 정의에서 온
  기본값이 있으며, 모델 계열 자체의 학습 설정이 이를 대체할 수 있습니다.
keywords:
  - libreyolo train 명령
  - libreyolo 학습 인자
  - yolo cli 학습
  - yolo 커스텀 데이터셋 학습
  - libreyolo dry run
  - yolo 레이어 고정
last_verified: 1.5.0
meta:
  - label: 명령
    value: libreyolo train
    mono: true
  - label: 필수
    value: data
    mono: true
  - label: 출력
    value: 'runs/train/exp 아래의 체크포인트, 지표, 로그'
snippets:
  examples:
    - label: 기본
      language: bash
      code: >
        # coco8.yaml은 패키지에 함께 포함되며 처음 사용할 때 이미지 8장을 내려받습니다.

        libreyolo train model=LibreYOLO9s.pt data=coco8.yaml epochs=10 imgsz=640
        batch=8
    - label: 해석된 설정 먼저 확인하기
      language: bash
      code: >
        # 모델 계열 기본값을 포함해 이번 실행이 사용할 설정을 출력하고,

        # 학습이나 데이터 로드 없이 종료합니다.

        libreyolo train model=LibreDFINEn.pt data=coco8.yaml epochs=10
        dry_run=true
    - label: 명시적인 레시피로 이름을 지정한 실행
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=coco8.yaml \
          epochs=50 batch=8 optimizer=adamw lr0=0.001 weight_decay=0.0001 \
          patience=20 save_period=5 project=runs/train name=yolo9s-coco8 exist_ok=true
source_hash: 3aad4298310d3081
---

## 사용법

```bash
libreyolo train data=<dataset.yaml> [model=<name|path>] [key=value ...]
```

인자는 `key=value` 쌍이며 POSIX 형식도 동작하므로 `epochs=50`과 `--epochs 50`은
같은 인자입니다. 불리언은 `true`와 `false`를 받습니다: 플래그에 부정 형식이 있는
경우 `amp=false`는 `--no-amp`가 됩니다.

## 인자

### 모델과 데이터

| 인자 | 기본값 | 의미 |
|---|---|---|
| `data` | | 데이터셋 YAML 경로(YOLO 형식, 예: `coco8.yaml`). 필수 |
| `model` | `yolox-s` | 모델 이름 또는 가중치 경로 |
| `task` | | 명시적 작업 재정의: `detect`, `segment`, `semantic`, `pose`, `classify`, `gaze`, `obb`, `point`, `depth` |
| `pretrained` | `true` | 사전 학습된 가중치를 사용합니다. `false`는 아키텍처를 구성해 처음부터 학습합니다 |
| `allow_download_scripts` | `false` | 데이터셋 YAML 다운로드 블록에 포함된 Python 코드를 허용합니다 |

### 학습 루프

| 인자 | 기본값 | 의미 |
|---|---|---|
| `epochs` | `300` | 학습 에폭 수 |
| `batch` | `16` | 장치당 배치 크기 |
| `imgsz` | `640` | 학습 이미지 크기: `640`(정사각형) 또는 `480x640`(HxW) |
| `device` | `auto` | 장치: `0`, `cpu`, `mps`, `auto` |
| `workers` | `4` | 데이터로더 워커 수 |
| `cache` | `false` | 데이터 로딩 속도를 높이기 위한 이미지 캐시: `ram`, `disk`, `true`, `false` |
| `seed` | `0` | 랜덤 시드 |
| `resume` | | 학습 재개: `true` 또는 체크포인트 경로 |
| `amp` | `true` | 자동 혼합 정밀도(Automatic Mixed Precision) |
| `amp_dtype` | `float16` | CUDA AMP dtype: `float16` 또는 `bfloat16` |
| `cuda_graph` | `false` | 학습의 순전파와 역전파를 CUDA 그래프로 캡처합니다. 단일 GPU와 지원되는 계열에서만 동작하며, 나머지는 eager 모드로 실행됩니다 |
| `lora` | `false` | LoRA 파인튜닝. 참고에 나열된 트랜스포머 계열에 적용됩니다 |
| `freeze` | | 레이어 고정: 정수 개수, 인덱스 목록 또는 모듈 이름 |

### 증류

| 인자 | 기본값 | 의미 |
|---|---|---|
| `distill_model` | | 교사 모델: 탐지기 체크포인트, 또는 백본 특징 증류를 위한 `dinov2` 같은 파운데이션 교사 id |
| `dis` | | 증류 손실 가중치. 설정하지 않으면 해당 손실 유형의 공개된 기본값을 사용합니다 |
| `distill_loss_type` | `mgd` | 탐지기 교사에 사용할 특징 손실: `mgd`, `cwd`. 파운데이션 교사는 항상 `feat_mse`를 사용합니다 |

### 옵티마이저

| 인자 | 기본값 | 의미 |
|---|---|---|
| `optimizer` | `sgd` | 옵티마이저: `sgd`, `adam`, `adamw` |
| `lr0` | `0.01` | 초기 학습률 |
| `momentum` | `0.937` | SGD 모멘텀이자 Adam 계열 옵티마이저의 1차 모멘트 계수 |
| `weight_decay` | `0.0005` | L2 정규화 |
| `nesterov` | `true` | Nesterov 모멘텀 |

### 스케줄러

| 인자 | 기본값 | 의미 |
|---|---|---|
| `scheduler` | `yoloxwarmcos` | LR 스케줄 유형 |
| `warmup_epochs` | `5` | 워밍업 기간 |
| `warmup_lr_start` | `0.0` | 초기 워밍업 LR |
| `min_lr_ratio` | `0.05` | 최소 LR 비율 |
| `lr_drop` | `100` | RF-DETR 스텝 LR 감소 에폭 |

### 증강

| 인자 | 기본값 | 의미 |
|---|---|---|
| `mosaic` | `1.0` | 모자이크 확률 |
| `mixup` | `1.0` | 믹스업 확률 |
| `hsv_prob` | `1.0` | HSV 지터 확률 |
| `flip_prob` | `0.5` | 좌우 반전 확률 |
| `degrees` | `10.0` | 회전 범위, 도 단위이며 양방향으로 적용됩니다 |
| `translate` | `0.1` | 이동 비율 |
| `shear` | `2.0` | 전단 각도 |
| `mosaic_scale` | `(0.1,2.0)` | 모자이크 스케일 범위 |
| `mixup_scale` | `(0.5,1.5)` | 믹스업 스케일 범위 |
| `no_aug_epochs` | `15` | 마지막 N 에폭 동안 증강을 비활성화합니다 |

### EMA

| 인자 | 기본값 | 의미 |
|---|---|---|
| `ema` | `true` | 지수 이동 평균(Exponential Moving Average) |
| `ema_decay` | `0.9998` | EMA 감쇠 계수 |

### 학습 중 검증

| 인자 | 기본값 | 의미 |
|---|---|---|
| `val` | `true` | 학습 중에 검증합니다 |
| `eval_interval` | `10` | N 에폭마다 검증합니다 |
| `max_det` | `300` | 검증 NMS 이후 이미지당 최대 예측 수 |
| `eval_max_det` | | COCO 평가기 상한. 설정하지 않으면 pycocotools의 AP@100 관례를 따릅니다 |
| `faster_coco_eval` | `true` | 설치되어 있으면 COCO 지표에 faster-coco-eval C++ 백엔드를 사용하고, 없으면 pycocotools로 되돌아갑니다 |
| `save_plots` | `false` | 학습 중 최종 검증 플롯을 저장합니다 |
| `patience` | `50` | 조기 종료 대기 에폭 수. `0`은 이를 비활성화합니다 |

### 출력

| 인자 | 기본값 | 의미 |
|---|---|---|
| `project` | `runs/train` | 출력 디렉터리 루트 |
| `name` | `exp` | 실험 이름 |
| `exist_ok` | `false` | 기존 출력 디렉터리를 재사용합니다 |
| `save_period` | `10` | N 에폭마다 체크포인트를 저장합니다 |
| `log_interval` | `10` | N 배치마다 손실을 기록합니다 |

### 에이전트 플래그

| 인자 | 기본값 | 의미 |
|---|---|---|
| `json` | `false` | JSON을 stdout에 출력 |
| `quiet` | `false` | stderr 억제 |
| `dry_run` | `false` | 실행하지 않고 설정을 해석해 출력 |
| `help_json` | `false` | 명령 스키마를 JSON으로 덤프하고 종료 |

## 예제

<code-tabs name="examples" />

## 참고

### 위의 기본값이 항상 실제로 사용되는 값은 아닙니다

모든 모델 계열에는 자체 학습 설정이 있으며, 그 설정이 기본 설정과 다른 경우
명시적으로 지정하지 않은 인자에 대해서는 계열의 값이 명령 기본값을 대체합니다.
직접 지정한 인자가 항상 우선합니다. `libreyolo cfg`는 기본 설정값과 계열별
재정의를 출력하며, 특정 계열이 실제로 무엇을 사용할지 확인하는 방법입니다.

`imgsz`가 이 점이 가장 크게 작용하는 인자입니다. 명령 기본값은 `640`이며, 이는
모든 체크포인트의 고유 입력 크기가 아닙니다: 공개된 RF-DETR 탐지 크기는 384,
512, 576, 704이고 YOLOX의 `n`과 `t` 체크포인트는 416입니다. RF-DETR 계열과
DEIMv2는 `imgsz`가 명시적으로 지정된 경우에만 이를 전달하도록 처리되므로,
그렇지 않으면 각자의 크기가 그대로 유지됩니다. 다른 계열에는 주어진 값이 그대로
전달되고 그 크기로 학습합니다. FOMO는 엄격한 쪽입니다: 각 크기는 자신의 고유
입력(96, 192, 224)만 허용하므로 FOMO 실행에서는 `imgsz`를 여기에 맞춰 지정해야
하며, 그렇지 않으면 오류와 함께 중단됩니다. RF-DETR 계열은 값이 패치 크기와
윈도 개수의 곱으로 나누어떨어질 것도 요구하며, 그렇지 않은 경우 가장 가까운 두
개의 유효한 크기를 알려줍니다.

### 모델 계열이 무시하는 인자

모든 계열이 모든 인자를 읽는 것은 아니며, 그 점이 드러나는 곳이 증강 인자입니다.
RF-DETR, D-FINE, DEIM, DEIMv2, RT-DETRv4, DINOv2는 모자이크도 믹스업도 어파인
워프도 없는 패스스루 파이프라인으로 학습하므로 `mosaic`, `mixup`, `hsv_prob`,
`degrees`, `translate`, `shear`, `mosaic_scale`, `mixup_scale`은 그곳에서 아무
효과도 내지 않습니다. EC는 같은 파이프라인을 사용하지만, 작업이 pose일 때는
`hsv_prob`, `degrees`, `translate`를 읽습니다. 분류 계열과 SegFormer, NAFNet은
그 집합 전체를 무시하고 `flip_prob`도 함께 무시하는데, 이들의 반전이 설정 가능한
확률이 아니라 고정된 확률로 동작하기 때문입니다. YOLO-NAS는 `mosaic`만
무시하는데, 대신 항상 켜져 있는 샘플별 어파인으로 증강하기 때문입니다. RF-DETR
계열은 그 목록에 더해 세 가지를 더 무시합니다: `optimizer`, `momentum`,
`nesterov`.

이 중 하나를 설정하는 것은 오류가 아닙니다. 실행은 계열 이름과 무시할 인자를
밝히는 한 줄을 stderr에 기록한 뒤 학습하며, 그 줄이 설치된 버전에 대한 권위 있는
목록입니다. 또한 그것이 유일한 신호이므로, `quiet=true`로 스크립트에서 실행하면
stderr의 다른 모든 출력과 함께 그 경고도 억제됩니다.

`val=false`는 이와 관련된 경우입니다. 대부분의 계열에서는 `eval_interval`을
`0`으로 설정합니다; RF-DETR 계열은 그 방식으로 검증을 비활성화할 수 없어 요청을
무시했다고 기록합니다.

### 그 밖에 알아둘 동작

`lora=true`는 RF-DETR, D-FINE, DEIM, DEIMv2, RT-DETR v1, v2, v4, EC,
ConvNeXt에서 받아들여집니다. 그 밖의 계열은 LoRA 없이 학습하는 대신
`config_unsupported`로 종료합니다.

`pretrained=false`와 `resume`을 함께 쓰는 것은 처음부터 학습을 지원하는
계열에서는 거부되는데, 둘이 서로 반대되는 것을 요구하기 때문입니다.

`mosaic`와 `mixup`은 설정 필드 `mosaic_prob`과 `mixup_prob`의 명령줄 표기입니다.
믹스업이 모자이크 샘플에만 적용되는 계열에서는 `mosaic`가 0인 상태에서 `mixup`을
0보다 크게 두어도 전혀 동작하지 않으며, 실행이 그 사실을 알려줍니다.

`dry_run=true`는 모델 참조를 해석하고 계열 기본값을 적용한 뒤, 학습에 사용할
설정을 출력합니다. 데이터셋은 로드하지 않으므로, 어떤 인자가 기대한 값에
도달했는지 확인하는 비용이 적게 드는 방법입니다.

stdout에는 최종 결과 객체가 실리고, 진행 상황과 경고는 stderr로 갑니다. 종료
코드는 성공 시 `0`, 사용법이나 설정 오류일 때 `2`, 데이터셋을 찾거나 읽을 수 없을
때 `3`, 모델을 로드할 수 없을 때 `4`, 그 밖의 런타임 실패일 때 `1`입니다.

관련 항목: 실행을 시작하기 전에 데이터셋을 확인하려면
[`libreyolo doctor`](/docs/cli/doctor), 브라우저에서 실행을 지켜보려면
[`libreyolo monitor`](/docs/cli/monitor), 결과를 측정하려면
[`libreyolo val`](/docs/cli/val).
