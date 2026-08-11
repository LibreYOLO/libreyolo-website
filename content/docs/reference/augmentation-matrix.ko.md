---
title: 증강 매트릭스
seo_title: 어떤 LibreYOLO 계열이 어떤 증강을 적용하는지
description: '계열별 증강 조정 지원: 열여섯 개의 TrainConfig 조정, 세 가지 상태, 여섯 가지 파이프라인 유형, 그리고 계열이 묵인하는 조정'
lead: >-
  증강 조정을 설정해도 파이프라인에 적용된다는 보장은 없습니다. 이 페이지는 각 학습 가능 계열이 TrainConfig의 각 조정을 어떻게
  처리하는지, 라이브러리가 제공하는 선언적 테이블을 단일 진실 소스로 사용하여 기록합니다.
keywords:
  - libreyolo 증강
  - mosaic_prob
  - mixup_prob
  - hsv_prob
  - no_aug_epochs
  - 증강 지원 매트릭스
  - TrainConfig 조정
last_verified: 1.5.0
verification: >-
  libreyolo/data/augment/spec.py에서 v1.5.0 기준으로 읽은 노브 목록, 상태, 원형, 패밀리별 편차 및 도우미
  함수. 해당 테이블은 tests/unit/test_augment_spec.py.에 의해 실제 파이프라인에 고정됨
snippets:
  usage:
    - label: 사양서에 직접 문의
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
source_hash: d2e1b9f5c81072e1
---

## 노브

이는 CLI 철자가 아닌 `TrainConfig` 필드 이름임. CLI는 자체 별칭을 이들에 매핑하므로 `--mosaic`가 `mosaic_prob`를 설정함

| 노브 | 의미 |
|---|---|
| `mosaic_prob` | 4-이미지 모자이크 샘플을 생성할 확률 |
| `mixup_prob` | 두 번째 샘플을 혼합할 확률 |
| `hsv_prob` | HSV 색상 지터 확률 |
| `flip_prob` | 수평 뒤집기 확률 |
| `degrees` | 아핀 워프의 무작위 회전 범위(도 단위) |
| `translate` | 아핀 변환을 위한 무작위 변환 분율 |
| `mosaic_scale` | 아핀 변환을 위한 무작위 스케일 범위 |
| `mixup_scale` | MixUp 파트너 이미지에 적용되는 지터 스케일 범위 |
| `shear` | 아핀 변환을 위한 무작위 전단 범위(도 단위) |
| `perspective` | 아핀 변환을 위한 사영 변환 크기 |
| `flipud` | 수직 뒤집기 확률 |
| `no_aug_epochs` | 강한 증강을 비활성화한 상태로 학습된 최종 에포크 |
| `auto_augment` | 분류용 AutoAugment 정책: randaugment, autoaugment 또는 augmix |
| `erasing` | 분류용 RandomErasing 확률 |
| `mixup` | 분류용 배치 MixUp 확률, 소프트 레이블 포함 |
| `cutmix` | 분류 배치-CutMix 확률, 소프트 레이블 포함 |

마지막 네 개는 분류 패키지입니다. 탐지 패밀리는 이를 무시합니다. `mixup`는 API 전용 노브입니다: CLI `--mixup`는 탐지 `mixup_prob`의 별칭입니다.

<code-tabs name="usage" />

## 세 가지 상태

| 상태 | 의미 |
|---|---|
| `used` | 노브가 패밀리의 학습 파이프라인에 도달하여 샘플을 변경합니다 |
| `gated_by_mosaic` | 노브는 모자이크 브랜치를 거친 샘플에만 적용되므로 `mosaic_prob == 0`로는 절대 작동하지 않습니다 |
| `ignored` | 노브는 파이프라인에 절대 도달하지 않습니다; 설정해도 아무 작업도 수행하지 않습니다 |

`ignored`는 실행 전 확인할 가치가 있는 것으로, 실패하는 경우가 없기 때문이다. CLI는 명시적으로 설정된 학습 파라미터가 선택된 패밀리에서 무시되는 경우 경고를 발생시키고, 트레이너는 `mixup_prob > 0`가 패밀리가 모자이크에서 MixUp을 차단하거나 `mosaic_prob`가 0일 경우 작동할 수 없음을 경고한다.

## 파이프라인 아키타입

모든 포함된 패밀리는 여섯 가지 파이프라인 중 하나를 따르며, 몇 가지 패밀리별 변형이 아래에 나열되어 있다.

| 노브 | YOLOX 스타일 | YOLO-NAS | DETR 스타일 | 분류 | 의미론 | 복원 |
|---|---|---|---|---|---|---|
| `mosaic_prob` | 사용됨 | 무시됨 | 무시됨 | 무시됨 | 무시됨 | 무시됨 |
| `mixup_prob` | 게이트됨 | 사용됨 | 무시됨 | 무시됨 | 무시됨 | 무시됨 |
| `hsv_prob` | 사용됨 | 사용됨 | 무시됨 | 무시됨 | 무시됨 | 무시됨 |
| `flip_prob` | 사용됨 | 사용됨 | 사용됨 | 무시됨 | 무시됨 | 무시됨 |
| `degrees` | 게이트됨 | 사용됨 | 무시됨 | 무시됨 | 무시됨 | 무시됨 |
| `translate` | 게이트됨 | 사용됨 | 무시됨 | 무시됨 | 무시됨 | 무시됨 |
| `mosaic_scale` | 게이트됨 | 사용됨 | 무시됨 | 무시됨 | 무시됨 | 무시됨 |
| `mixup_scale` | 게이트됨 | 사용됨 | 무시됨 | 무시됨 | 무시됨 | 무시됨 |
| `shear` | 차단됨 | 사용됨 | 무시됨 | 무시됨 | 무시됨 | 무시됨 |
| `perspective` | 차단됨 | 사용됨 | 무시됨 | 무시됨 | 무시됨 | 무시됨 |
| `flipud` | 사용됨 | 사용됨 | 무시됨 | 무시됨 | 무시됨 | 무시됨 |
| `no_aug_epochs` | 사용됨 | 사용됨 | 사용됨 | 사용됨 | 사용됨 | 사용됨 |
| `auto_augment` | 무시됨 | 무시됨 | 무시됨 | 사용됨 | 무시됨 | 무시됨 |
| `erasing` | 무시됨 | 무시됨 | 무시됨 | 사용됨 | 무시됨 | 무시됨 |
| `mixup` | 무시됨 | 무시됨 | 무시됨 | 사용됨 | 무시됨 | 무시됨 |
| `cutmix` | 무시됨 | 무시됨 | 무시됨 | 사용됨 | 무시됨 | 무시됨 |

YOLOX 스타일 파이프라인에서는 샘플별 전처리가 HSV 지터링과 뒤집기를 적용하는 반면, 어파인 워프와 MixUp은 모자이크 브랜치 안에서만 실행됩니다. 반면 YOLO-NAS는 항상 적용되는 샘플별 어파인을 실행하고 모자이크는 무시하며, MixUp을 독립적으로 적용하고 어파인 스케일 범위로 `mosaic_scale`를 재사용합니다.

DETR 스타일 파이프라인은 모자이크가 없는 패스스루 변환입니다. 포토메트릭 왜곡, 줌 아웃 및 IoU 크롭은 설정 가능한 조정기가 아니라 레시피 상수이므로 `hsv_prob`와 기하학 조정기는 결코 도달하지 않습니다. 분류 파이프라인은 ImageFolder 변환을 사용하며, 수평 뒤집기는 `flip_prob`가 아닌 고정된 0.5입니다. 시맨틱 스케일 지터와 HSV는 구성 조정기가 아닌 패밀리 클래스 속성에서 가져오며, 복원 뒤집기는 입력과 대상 작업이 결합된 연산으로 고정된 0.5 확률로 수행됩니다.

`no_aug_epochs`는 어디에서나 준수되지만, 무엇을 끄는지는 다릅니다: YOLOX 스타일에서는 모자이크 및 MixUp, YOLO-NAS에서는 어파인 및 MixUp, DETR 스타일에서는 강한 포토메트릭 및 크롭 증강과 학습률 꼬리, 나머지에서는 스케줄러 꼬리입니다.

## 아키타입별 패밀리

| 아키타입 | 계열 |
|---|---|
| YOLOX 스타일 | `yolox`, `yolo7`, `yolo9`, `yolo9_e2e`, `yolo9_p2`, `rtmdet`, `picodet`, `rtdetr`, `rtdetrv2`, `fomo` |
| YOLO-NAS | `yolonas` |
| DETR 스타일 | `dfine`, `domedetr`, `deim`, `deimv2`, `rtdetrv4`, `rfdetr`, `ec`, `dinov2` |
| 분류 | `resnet`, `convnext`, `mobilenetv4`, `efficientnetv2` |
| 의미론 | `segformer` |
| 복원 | `nafnet` |

25개의 계열이 포함됩니다. 이 목록 외의 계열은 빈 무시 집합을 반환하므로 경고가 발생하지 않습니다.

## 편차

| 계열 | 모형과의 차이 |
|---|---|
| `rtmdet` | `flipud` 무시됨: 변형에 수직 뒤집기가 없음 |
| `picodet` | `flipud` 무시됨 |
| `rtdetr` | `flipud` 무시됨 |
| `rtdetrv2` | `flipud` 무시됨 |
| `fomo` | `perspective` 및 `flipud` 무시됨 |
| `ec` | `hsv_prob`, `degrees` 및 `translate` 사용됨, `task="pose"`에만 해당; 감지 및 분할은 고정된 광도 레시피 사용 |
| `dinov2` | 분류 팩이 사용됨, `task="classify"`에만 해당 |

`ec` 및 `dinov2`는 다중 작업 패밀리이므로, 패밀리의 모든 학습 가능한 작업이 무시할 때만 조정기가 무시됨으로 표시됨. 이는 한 작업에 대해서는 잘못되고 다른 작업에 대해서는 올바른 CLI 경고가 발생하지 않도록 유지함.

Dome-DETR는 D-FINE의 변환을 변경 없이 상속합니다. 단 하나 적용할 수 없는 것은 다중 스케일 학습이며, 이는 증강 사양이 아닌 구성에서 비활성화됩니다.

## 계열별 설정

일부 계열은 기본 클래스가 아닌 `TrainConfig` 하위 클래스 자체에 증강 설정을 가지고 있습니다. CLI에서는 노출되지 않으므로 Python API를 통해 설정하십시오.

| 계열 | 설정 | 의미 |
|---|---|---|
| `yolo9`, `yolo9_e2e`, `yolo9_p2` | `copy_paste` | 복사-붙여넣기 인스턴스 증강 확률, `task="segment"`만 해당 |
| `yolo9`, `yolo9_e2e`, `yolo9_p2` | `copy_paste_mode` | 복사-붙여넣기 소스: `flip`는 동일한 샘플을 사용하고, `mixup`는 두 번째 샘플을 사용 |
| `yolo9`, `yolo9_e2e`, `yolo9_p2` | `rot90` | 무작위 90도 회전 확률 |
| `rfdetr` | `copy_paste` | `task="segment"`, `flip` 모드에서만 복사-붙여넣기 확률 |
| `rfdetr` | `copy_paste_mode` | `task="segment"`의 복사-붙여넣기 소스 모드 |
| `rfdetr` | `crop_resize_prob` | 기본 파이프라인에서 무작위 자르기-크기 변경 확률 |
| `dfine` | `crop_resize_prob` | 무작위 자르기-크기 변경 확률, `task="segment"` |
| `ec` | `crop_resize_prob` | 무작위 자르기-크기 변경 확률, `task="segment"` |
| `ec`, `yolonas` | `brightness_contrast_prob` | 밝기 및 대비 지터 확률, `task="pose"` |
| `ec`, `yolonas` | `affine_prob` | 키포인트 인식 아핀 확률, `task="pose"` |

`rot90`는 `yolo9`에서 검출 및 OBB에 적용됩니다.

## 사양 조회

| 도우미 | 반환 |
|---|---|
| `aug_support(family)` | 노브-투-`Support` 테이블 또는 알 수 없는 패밀리를 위한 `None` |
| `ignored_aug_params(family)` | 패밀리가 무시하는 노브 이름 세트; 알 수 없는 패밀리의 경우 비어 있음 |
| `uses_mosaic_gating(family)` | 패밀리의 MixUp이 모자이크 샘플에서만 활성화되는지 여부 |
| `display_name(family)` | 경고에 사용되는 사용자 대상 패밀리 이름 |
| `mixup_gating_warning(family, mosaic_prob, mixup_prob)` | MixUp이 결코 발동할 수 없을 때의 경고 텍스트, 그렇지 않으면 `None` |

`Support`는 `status`와 `note`의 이름이 있는 튜플이며, 노트는 해당 패밀리에서 노브가 무시되거나 제한되는 이유를 설명합니다.

## 모자이크 게이트

YOLOX 스타일의 패밀리의 경우, `mixup_prob=0.5`와 `mosaic_prob=0` 조합은 MixUp을 완전히 비활성화합니다. MixUp은 모자이크 샘플에만 적용되기 때문입니다. 이 조합은 학습 후반에 모자이크를 끌 때 쉽게 발생할 수 있습니다. 트레이너는 패밀리 이름을 명시하며 경고를 기록하고, `mixup_gating_warning`가 그 뒤의 순수 함수입니다.
