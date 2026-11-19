---
title: 데이터 증강 매트릭스
seo_title: LibreYOLO 계열별 데이터 증강 반영 여부
description: '모든 데이터 증강 노브의 범위와 CLI 표기, 학습 가능한 31개 계열 각각에서 각 노브의 상태, 그리고 각 계열의 기본값을 정리합니다.'
lead: >-
  데이터 증강 노브를 설정한다고 해서 그 노브가 파이프라인에 도달한다는 보장은 없습니다. 이 페이지는 학습 가능한 각 계열이
  TrainConfig의 각 노브를 어떻게 다루는지와 각 계열의 시작값을 기록하며, 내용은 이 페이지가 다루는 릴리스의 라이브러리에서
  생성했습니다.
keywords:
  - libreyolo 데이터 증강
  - 데이터 증강 파라미터
  - 데이터 증강 기본값
  - mosaic_prob
  - mixup_prob
  - hsv_prob
  - mosaic_scale
  - no_aug_epochs
  - 데이터 증강 지원 매트릭스
  - TrainConfig 노브
last_verified: 1.8.0
verification: >-
  LibreYOLO 1.8.0에서 scripts/build-augmentation/collect.py로 생성했습니다. 노브 목록, 상태,
  메모는 libreyolo/data/augment/spec.py에서 가져왔고(라이브러리 테스트가 이 사양을 실제 파이프라인에 고정합니다),
  기본값은 각 계열의 TrainConfig 하위 클래스에서, CLI 표기는 libreyolo train 명령에서 가져왔습니다. 범위와, 사양과
  파이프라인이 일치하지 않는 유일한 지점은 같은 스크립트가 라이브러리와 대조해 확인합니다.
snippets:
  usage:
    - label: 사양 직접 조회
      language: python
      code: |
        from libreyolo.data.augment.spec import (
            AUG_KNOBS,
            aug_support,
            ignored_aug_params,
            uses_mosaic_gating,
        )

        print(sorted(AUG_KNOBS))

        table = aug_support("yolo9")
        print(table["mixup_prob"].status, table["mixup_prob"].note)

        print(sorted(ignored_aug_params("dfine")))
        print(uses_mosaic_gating("yolo9"), uses_mosaic_gating("yolonas"))
    - label: 계열의 기본값
      language: python
      code: |
        from libreyolo.training.config import YOLO9Config

        config = YOLO9Config()
        print(config.mosaic_scale, config.degrees, config.mixup_prob)
source_hash: a88efd1537732e6c
---

## 노브 목록

이 이름들은 `TrainConfig` 필드 이름입니다. `train()`과 CLI는 CLI 열의 짧은 표기도 받아들이므로, `mosaic`로 `mosaic_prob`을 설정할 수 있습니다. CLI에서 `mixup`은 작업에 따라 전달됩니다: 분류 모델에서는 분류 배치 혼합, 탐지 모델에서는 `mixup_prob`입니다. `perspective`는 CLI 플래그가 없으며 Python에서 설정합니다.

<aug-knobs />

베이스 기본값은 `TrainConfig` 자체에 정의된 값입니다. 대부분의 계열은 그중 일부를 재정의하며, 학습이 실제로 사용하는 시작값은 [계열별 기본값](#defaults-by-family)에 있습니다. 각 노브의 효과를 그림으로 보려면 [데이터 증강](/docs/train/augmentations)을 참조하십시오.

대괄호로 표시한 범위는 구간이므로, `[0, 1)`은 0을 포함하고 1을 제외합니다. 분류 노브는 해당 파이프라인을 만들 때 검사되며, 범위를 벗어나면 `ValueError`를 발생시킵니다. 탐지 노브는 `TrainConfig`가 검사하지 않으므로, 범위를 벗어난 값도 아무 경고 없이 학습됩니다. 예외는 추출된 스케일이 0 이하인 경우와 `perspective`가 음수인 경우로, 이때는 샘플을 변환할 때 `ValueError`가 발생합니다.

<code-tabs name="usage" />

## 세 가지 상태

| 상태 | 의미 |
|---|---|
| `used` | 노브가 계열의 학습 파이프라인에 도달해 샘플을 바꿉니다 |
| `gated_by_mosaic` | 노브가 모자이크 분기를 거친 샘플에만 적용되므로, `mosaic_prob == 0`이면 한 번도 작동하지 않습니다 |
| `ignored` | 노브가 파이프라인에 도달하지 않으며, 설정해도 아무 효과가 없습니다 |

아무것도 실패하지 않기 때문에, 실행 전에 확인할 가치가 있는 것은 `ignored`입니다. 명시적으로 설정한 학습 매개변수를 선택한 계열이 무시하면 CLI가 경고하고, 계열이 MixUp을 모자이크에 종속시키며 `mosaic_prob`이 0이어서 `mixup_prob > 0`이 작동할 수 없으면 트레이너가 경고합니다.

## 계열별 노브 반영 여부

<aug-matrix />

대상 계열은 모두 여섯 가지 파이프라인 중 하나를 따릅니다. YOLOX 방식 파이프라인에서는 샘플별 전처리가 HSV 지터와 뒤집기를 적용하고, 아핀 변환과 MixUp은 모자이크 분기 안에서만 실행됩니다. 반면 YOLO-NAS는 항상 켜져 있는 샘플별 아핀 변환을 실행하고, 모자이크를 무시하며, MixUp을 독립적으로 적용하고, `mosaic_scale`을 아핀 변환의 스케일 범위로 재사용합니다.

DETR 방식 파이프라인은 모자이크가 없는 패스스루 변환입니다. 이 파이프라인의 광도 왜곡, 줌아웃, IoU 크롭은 설정 가능한 노브가 아니라 레시피 상수이며, 그래서 `hsv_prob`과 기하 노브는 이 파이프라인에 도달하지 않습니다. 분류는 수평 뒤집기에 `flip_prob`을, 수직 뒤집기에 `flipud`를 사용합니다. 시맨틱 분할의 스케일 지터와 HSV는 설정 노브가 아니라 계열 클래스 속성에서 오며, 복원의 뒤집기는 입력과 타깃을 함께 처리하는 연산으로 확률이 0.5로 고정되어 있습니다.

`no_aug_epochs`는 모든 곳에서 반영되지만 끄는 대상은 다릅니다: YOLOX 방식은 모자이크와 MixUp, YOLO-NAS는 아핀 변환과 MixUp, DETR 방식은 강한 광도 및 크롭 증강과 학습률 마지막 구간, 분류는 자동 증강, 지우기, MixUp, CutMix입니다. 분류의 크롭과 뒤집기는 활성 상태로 유지됩니다.

계열이 같은 파이프라인의 다른 계열과 다른 경우, 그 이유는 다음 중 하나입니다:

| 계열 | 차이 |
|---|---|
| `rtmdet`, `picodet`, `rtdetr`, `rtdetrv2` | `flipud` 무시: 이 계열들의 변환에는 수직 뒤집기가 없습니다 |
| `fomo` | `perspective`, `flipud` 무시 |
| `ppyoloe` | `flipud` 무시: 원본 레시피는 대신 무작위 90도 회전을 사용합니다 |
| `yolo9`, `yolo9_e2e`, `yolo9_p2`, `yolo7`, `rtdetr`, `rtdetrv2` | `mixup_scale` 무시: 이 계열들의 MixUp은 전체 프레임을 무작위 비율로 혼합하며 어느 쪽도 리스케일하지 않습니다 |
| `gtr` | `mosaic_prob`, `mixup_prob` 사용, 처음 `mosaic_epochs` 에폭 동안; `degrees`, `translate`, `mosaic_scale`은 모자이크 뒤에 오는 아핀 변환을 결정합니다 |
| `rfdetr` | `hsv_prob` 무시: 네이티브 파이프라인에 HSV 지터가 없습니다 |
| `ec` | `hsv_prob`, `degrees`, `translate` 사용, `task="pose"` 전용; detect와 segment는 고정된 광도 레시피를 사용합니다 |
| `dinov2` | 분류 팩 사용, `task="classify"` 전용 |

`ec`와 `dinov2`는 다중 작업 계열이므로, 노브는 해당 계열의 학습 가능한 모든 작업이 그 노브를 무시할 때만 무시로 표시됩니다. 이렇게 하면 CLI 경고가 한 작업에서는 틀리고 다른 작업에서는 맞는 일이 생기지 않습니다.

Dome-DETR 계열은 D-FINE의 변환을 그대로 상속합니다. 유일하게 쓸 수 없는 것은 다중 스케일 학습이며, 이를 끄는 것은 데이터 증강 사양이 아니라 Dome-DETR 자체의 설정입니다.

대상 계열은 31개입니다. 이 목록에 없는 계열에 대해서는 `ignored_aug_params`가 빈 집합을 반환하므로 경고가 출력되지 않습니다.

## 계열별 기본값

각 탐지 계열의 설정이 시작하는 값입니다. 빈 칸은 해당 계열이 그 노브를 무시한다는 뜻이므로, 그 기본값은 아무것도 바꾸지 않습니다.

<aug-defaults />

개별 값보다 중요한 패턴이 두 가지 있습니다. RT-DETR, RT-DETRv2, PicoDet, FOMO는 `mosaic_prob=0`으로 시작하므로, 이 계열에서는 매트릭스에서 모자이크 종속으로 표시된 노브가 모자이크를 켜기 전까지 아무 일도 하지 않습니다. YOLOv9은 회전, 전단, MixUp이 꺼진 상태와 `mosaic_scale=(0.5, 1.5)`로 시작하며, 베이스 기본값은 `(0.1, 2.0)`입니다.

EdgeCrafter는 `task="pose"`에서만 `hsv_prob`, `degrees`, `translate`를 읽으므로, 이 계열의 행에는 이 세 노브의 자세 추정 기본값이 표시됩니다.

분류 계열은 베이스 기본값에서 시작합니다: `flip_prob=0.5`, `flipud=0.0`, `scale=(0.5, 1.0)`, 계열 고유의 `crop_pct`, 그리고 꺼진 상태의 `auto_augment`, `erasing`, `mixup`, `cutmix`입니다.

## 계열 전용 노브

일부 계열은 데이터 증강 노브를 베이스 클래스가 아니라 자체 `TrainConfig` 하위 클래스에 둡니다. CLI는 이 노브를 제공하지 않으므로 Python API로 설정하십시오.

| 계열 | 노브 | 의미 |
|---|---|---|
| `yolo9`, `yolo9_e2e`, `yolo9_p2` | `copy_paste` | copy-paste 인스턴스 증강 확률, `task="segment"` 전용 |
| `yolo9`, `yolo9_e2e`, `yolo9_p2` | `copy_paste_mode` | copy-paste 소스: `flip`은 같은 샘플을 미러링하고, `mixup`은 두 번째 샘플을 사용합니다 |
| `yolo9`, `yolo9_e2e`, `yolo9_p2` | `rot90` | 무작위 90도 회전 확률 |
| `rfdetr` | `copy_paste` | `task="segment"`용 copy-paste 확률, `flip` 모드 전용 |
| `rfdetr` | `copy_paste_mode` | `task="segment"`용 copy-paste 소스 모드 |
| `rfdetr` | `crop_resize_prob` | 네이티브 파이프라인의 무작위 크롭 리사이즈 확률 |
| `dfine` | `crop_resize_prob` | 무작위 크롭 리사이즈 확률, `task="segment"` |
| `ec` | `crop_resize_prob` | 무작위 크롭 리사이즈 확률, `task="segment"` |
| `ec`, `yolonas` | `brightness_contrast_prob` | 밝기 및 대비 지터 확률, `task="pose"` |
| `ec`, `yolonas` | `affine_prob` | 키포인트 인식 아핀 변환 확률, `task="pose"` |

`yolo9`에서 `rot90` 노브는 detect와 OBB에 적용됩니다.

## 사양 조회

| 헬퍼 함수 | 반환값 |
|---|---|
| `aug_support(family)` | 노브와 `Support`를 대응시킨 표, 알 수 없는 계열이면 `None` |
| `ignored_aug_params(family)` | 계열이 무시하는 노브 이름의 집합, 알 수 없는 계열이면 빈 집합 |
| `uses_mosaic_gating(family)` | 계열의 MixUp이 모자이크 샘플에서만 작동하는지 여부 |
| `display_name(family)` | 경고에 쓰이는, 사람이 읽기 위한 계열 이름 |
| `mixup_gating_warning(family, mosaic_prob, mixup_prob)` | MixUp이 결코 작동할 수 없을 때의 경고 문구, 그렇지 않으면 `None` |

`Support`는 `status`와 `note`로 이루어진 네임드 튜플이며, note는 해당 계열에서 노브가 무시되거나 모자이크에 종속되는 이유를 설명합니다. 1.8.0의 사양은 YOLOv9 모자이크 래퍼로 학습하는 계열에 대해 `mixup_scale`을 여전히 모자이크 종속으로 기재하므로, 파이프라인과 대조해 확인한 위의 매트릭스가 이를 무시로 표시하는데도 `aug_support`는 이 계열들에서 모자이크 종속으로 보고합니다.

## 모자이크 게이트

YOLOX 방식 계열에서는 MixUp이 모자이크 샘플에만 적용되므로, `mosaic_prob=0`과 함께 `mixup_prob=0.5`를 지정하면 MixUp이 완전히 비활성화됩니다. 이 조합은 학습 후반에 모자이크를 끌 때 쉽게 만들어집니다. 트레이너는 계열 이름을 명시한 경고를 기록하며, `mixup_gating_warning`은 그 경고의 바탕이 되는 순수 함수입니다.
