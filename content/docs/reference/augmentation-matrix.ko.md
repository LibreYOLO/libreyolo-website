---
title: 확장 행렬
seo_title: 어떤 LibreYOLO 계열이 어떤 증강을 존중하는가
description: >-
  계열별 증강 조절기 지원: 열여섯 개의 TrainConfig 조절기, 세 가지 상태, 여섯 가지 파이프라인 전형, 그리고 한 계열이 묵묵히
  무시하는 조절기들.
lead: >-
  증강 조절기를 설정한다고 해서 그것이 파이프라인에 도달한다는 보장은 없습니다. 이 페이지는 라이브러리가 단일 진실 소스로 제공하는 선언적
  테이블을 사용하여 각 학습 가능 계열이 TrainConfig의 각 조절기를 어떻게 다루는지 기록합니다.
keywords:
  - libreyolo 증강
  - 모자이크_확률
  - 혼합 확률
  - hsv_확률
  - 증강 없는 에포크
  - 증강 지원 매트릭스
  - TrainConfig 설정 조절
last_verified: 1.6.0
verification: '노브 목록, 상태, 전형, 계열별 편차 및 보조 함수는 v1.6.0에서 libreyolo/data/augment/spec.py로부터 읽습니다. 해당 테이블은 tests/unit/test_augment_spec.py.에 의해 실제 파이프라인에 고정되어 있습니다.'
snippets:
  usage:
    - label: 사양서에 직접 물어보십시오
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
source_hash: "f3cba41ceadf131f"
---

## 손잡이들

이들은 CLI 철자가 아니라 `TrainConfig` 필드 이름입니다. CLI는 자체 별칭을 이들에 매핑하므로, `--mosaic`는 `mosaic_prob`를 설정합니다.

| 손잡이 | 의미 |
|---|---|
| `mosaic_prob` | 4개 이미지 모자이크 샘플을 만들 확률 |
| `mixup_prob` | 두 번째 샘플에서 혼합될 확률 |
| `hsv_prob` | HSV 색상 변동 확률 |
| `flip_prob` | 수평 뒤집기 확률 |
| `degrees` | 어파인 변환의 임의 회전 범위(단위: 도) |
| `translate` | 어파인 워프를 위한 임의 변환 분수 |
| `mosaic_scale` | 어파인 왜곡을 위한 랜덤 스케일 범위 |
| `mixup_scale` | MixUp 파트너 이미지에 적용된 지터-스케일 범위 |
| `shear` | 어파인 변환을 위한 무작위 전단 범위(도 단위) |
| `perspective` | 아핀 왜곡을 위한 사영 변환 크기 |
| `flipud` | 수직 뒤집기 확률 |
| `no_aug_epochs` | 강한 증강을 사용하지 않고 최종 에포크 학습 |
| `auto_augment` | 분류 AutoAugment 정책: randaugment, autoaugment 또는 augmix |
| `erasing` | 분류 랜덤지우기 확률 |
| `mixup` | 분류 배치-MixUp 확률, 소프트 레이블과 함께 |
| `cutmix` | 분류 배치-CutMix 확률, 소프트 레이블과 함께 |

마지막 네 가지는 분류 설정입니다. 탐지 계열은 이를 무시합니다. CLI는 분류 모델에서는 `mixup`을 분류 배치 혼합에 전달하고 탐지 모델에서는 `mixup_prob`에 전달합니다.

<code-tabs name="usage" />

## 세 가지 상태

| 상태 | 의미 |
|---|---|
| `used` | 손잡이가 계열의 기차 파이프라인에 도달하여 샘플을 변경합니다 |
| `gated_by_mosaic` | 이 노브는 모자이크 브랜치를 선택한 샘플에만 적용되므로, `mosaic_prob == 0`에서는 절대 작동하지 않습니다 |
| `ignored` | 노브는 파이프라인에 거의 도달하지 못하며, 설정해도 아무 소용이 없습니다 |

`ignored`는 실행 전에 확인할 가치가 있는 것으로, 아무것도 실패하지 않기 때문입니다. CLI는 명시적으로 설정된 학습 매개변수가 선택된 계열가 무시할 경우 경고하고, 트레이너는 계열이 모자이크에서 MixUp을 차단하여 `mixup_prob > 0`를 실행할 수 없거나 `mosaic_prob`가 0일 경우 경고합니다.

## 파이프라인 전형

모든 보험 가입 계열은 여섯 가지 경로 중 하나를 따르며, 아래에는 계열별 몇 가지 예외 사항이 나열되어 있습니다.

| 설정 | YOLOX 방식 | YOLO-NAS | DETR 방식 | 분류 | 시맨틱 분할 | 복원 |
|---|---|---|---|---|---|---|
| `mosaic_prob` | 사용 | 무시 | 무시 | 무시 | 무시 | 무시 |
| `mixup_prob` | 조건부 | 사용 | 무시 | 무시 | 무시 | 무시 |
| `hsv_prob` | 사용 | 사용 | 무시 | 무시 | 무시 | 무시 |
| `flip_prob` | 사용 | 사용 | 사용 | 사용 | 무시 | 무시 |
| `degrees` | 조건부 | 사용 | 무시 | 무시 | 무시 | 무시 |
| `translate` | 조건부 | 사용 | 무시 | 무시 | 무시 | 무시 |
| `mosaic_scale` | 조건부 | 사용 | 무시 | 무시 | 무시 | 무시 |
| `mixup_scale` | 조건부 | 사용 | 무시 | 무시 | 무시 | 무시 |
| `shear` | 조건부 | 사용 | 무시 | 무시 | 무시 | 무시 |
| `perspective` | 조건부 | 사용 | 무시 | 무시 | 무시 | 무시 |
| `flipud` | 사용 | 사용 | 무시 | 사용 | 무시 | 무시 |
| `no_aug_epochs` | 사용 | 사용 | 사용 | 사용 | 사용 | 사용 |
| `auto_augment` | 무시 | 무시 | 무시 | 사용 | 무시 | 무시 |
| `erasing` | 무시 | 무시 | 무시 | 사용 | 무시 | 무시 |
| `mixup` | 무시 | 무시 | 무시 | 사용 | 무시 | 무시 |
| `cutmix` | 무시 | 무시 | 무시 | 사용 | 무시 | 무시 |

YOLOX 스타일 파이프라인에서는 샘플별 전처리가 HSV 지터와 플립을 적용하는 반면, 어파인 워프와 MixUp은 모자이크 브랜치 내부에서만 실행됩니다. 대신 YOLO-NAS는 항상 켜져 있는 샘플별 어파인을 실행하고, 모자이크를 무시하며, MixUp을 독립적으로 적용하며, 어파인 스케일 범위로 `mosaic_scale`를 재사용합니다.

DETR 방식 파이프라인은 모자이크가 없는 통과형 변환입니다. 광도 왜곡, 축소, IoU 크롭은 조정 가능한 설정이 아니라 레시피 상수이므로 `hsv_prob`와 기하 설정이 적용되지 않습니다. 분류는 수평 뒤집기에 `flip_prob`, 수직 뒤집기에 `flipud`를 사용합니다. 시맨틱 분할의 스케일 지터와 HSV는 설정값 대신 계열 클래스 속성을 사용하며, 복원 뒤집기는 고정 확률 0.5로 입력과 타깃에 함께 적용됩니다.

`no_aug_epochs`는 모든 곳에 적용되지만 끄는 기능은 다릅니다. YOLOX 방식은 모자이크와 MixUp, YOLO-NAS는 아핀 변환과 MixUp, DETR 방식은 강한 광도 및 크롭 증강과 학습률 마지막 구간, 분류는 자동 증강, 지우기, MixUp, CutMix를 제어합니다. 분류의 크롭과 뒤집기는 활성 상태를 유지합니다.

## 원형별 계열

| 원형 | 계열들 |
|---|---|
| YOLOX 스타일 | `yolox`, `yolo7`, `yolo9`, `yolo9_e2e`, `yolo9_p2`, `rtmdet`, `picodet`, `rtdetr`, `rtdetrv2`, `fomo` |
| YOLO-NAS | `yolonas` |
| DETR 스타일 | `dfine`, `domedetr`, `deim`, `deimv2`, `rtdetrv4`, `rfdetr`, `ec`, `dinov2` |
| 분류 | `resnet`, `convnext`, `mobilenetv4`, `efficientnetv2` |
| 의미론의 | `segformer` |
| 복원하다 | `nafnet` |

25개의 계열이 포함됩니다. 이 목록에 없는 계열은 비어 있는 무시 집합을 반환하므로 경고가 발생하지 않습니다.

## 편차

| 계열 | 원형과의 차이 |
|---|---|
| `rtmdet` | `flipud` 무시됨: 해당 변환에는 수직 뒤집기가 없습니다 |
| `picodet` | `flipud` 무시됨 |
| `rtdetr` | `flipud` 무시됨 |
| `rtdetrv2` | `flipud` 무시됨 |
| `fomo` | `perspective` 및 `flipud` 무시됨 |
| `ec` | `hsv_prob`, `degrees` 및 `translate` 사용, `task="pose"` 전용; 고정 광측정 레시피를 사용하여 탐지 및 분할 |
| `dinov2` | 분류 팩은 `task="classify"`에만 사용됩니다 |

`ec`와 `dinov2`는 다중 작업 계열이므로, 노브는 해당 계열의 모든 학습 가능한 작업이 이를 무시할 때만 무시된 것으로 표시됩니다. 이렇게 하면 한 작업에서는 CLI 경고가 잘못되었지만 다른 작업에서는 맞는 경우가 발생하지 않습니다.

Dome-DETR은 D-FINE의 변환을 변경 없이 그대로 계승합니다. 그것이 할 수 없는 한 가지는 다중 스케일 학습인데, 이는 증강 사양이 아니라 구성에서 비활성화됩니다.

## 계열별 노브

일부 계열은 기본 클래스가 아닌 자체 `TrainConfig` 하위 클래스에 증강 노브를 가지고 있습니다. CLI에서는 이를 노출하지 않으며, Python API를 통해 설정하십시오.

| 계열 | 손잡이 | 의미 |
|---|---|---|
| `yolo9`, `yolo9_e2e`, `yolo9_p2` | `copy_paste` | 복사-붙여넣기 인스턴스 증가 확률, `task="segment"`만 |
| `yolo9`, `yolo9_e2e`, `yolo9_p2` | `copy_paste_mode` | 복사-붙여넣기 소스: `flip`는 동일한 샘플을 반영하고, `mixup`는 두 번째 샘플을 사용합니다 |
| `yolo9`, `yolo9_e2e`, `yolo9_p2` | `rot90` | 무작위 90도 회전 확률 |
| `rfdetr` | `copy_paste` | `task="segment"`, `flip` 모드에서만 복사-붙여넣기 확률 |
| `rfdetr` | `copy_paste_mode` | `task="segment"`용 소스 모드 복사-붙여넣기 |
| `rfdetr` | `crop_resize_prob` | 원래 파이프라인에서 랜덤 자르기-크기 조정 확률 |
| `dfine` | `crop_resize_prob` | 무작위 자르기-크기 조정 확률, `task="segment"` |
| `ec` | `crop_resize_prob` | 무작위 자르기-크기 조정 확률, `task="segment"` |
| `ec`, `yolonas` | `brightness_contrast_prob` | 밝기 및 대비 지터 확률, `task="pose"` |
| `ec`, `yolonas` | `affine_prob` | 키포인트 인식 아핀 확률, `task="pose"` |

`rot90`는 `yolo9`에서 OBB를 탐지하는 데 적용됩니다.

## 사양 조회

| 도움이 되는 사람 | 반환 |
|---|---|
| `aug_support(family)` | 노브-대-`Support` 테이블, 또는 알 수 없는 계열의 경우 `None` |
| `ignored_aug_params(family)` | 계열이 무시하는 노브 이름의 집합; 알려지지 않은 계열의 경우 비어 있음 |
| `uses_mosaic_gating(family)` | 계열의 MixUp이 모자이크 샘플에서만 작동하는지 여부 |
| `display_name(family)` | 경고에 사용되는 사람을 향한 성 |
| `mixup_gating_warning(family, mosaic_prob, mixup_prob)` | MixUp가 절대 발생할 수 없을 때의 경고 문구, 그렇지 않으면 `None` |

`Support`는 `status`와 `note`의 명명된 튜플이며, 여기서 노트는 해당 계열에서 노브가 무시되거나 게이트 처리되는 이유를 설명합니다.

## 모자이크 문

YOLOX 스타일의 계열의 경우, `mixup_prob=0.5`와 `mosaic_prob=0` 조합은 MixUp을 완전히 비활성화합니다. MixUp은 모자이크 샘플에만 적용되기 때문입니다. 이 조합은 학습 후반에 모자이크를 끌 때 쉽게 도달할 수 있습니다. 트레이너는 계열를 명시하는 경고를 기록하며, `mixup_gating_warning`가 그 뒤에 있는 순수 함수입니다.
