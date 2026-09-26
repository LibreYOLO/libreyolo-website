---
title: 증강
seo_title: LibreYOLO의 학습 증강
description: >-
  TrainConfig의 증강 노브, 그 뒤에 있는 네 가지 파이프라인 모양, 그리고 어떤 노브가 사용되거나 제한되거나 무시되는지를 나타내는
  계열별 표.
lead: >-
  증강은 TrainConfig의 조절기로 구성되지만, 각 모델 계열는 자체 학습 파이프라인을 실행하며, 모자이크 분기가 없는 파이프라인은 이를
  근사하지 않고 mosaic_prob을 무시합니다.
keywords:
  - yolo 데이터 증강
  - 모자이크 증강
  - 혼동
  - HSV 지터
  - 랜덤 어파인
  - 복사 붙여넣기 증강
  - 랜덤증강
  - 컷믹스
  - 증강_없음_에포크
last_verified: 1.5.0
snippets:
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=100,
            mosaic_prob=1.0,
            mixup_prob=0.15,
            hsv_prob=1.0,
            flip_prob=0.5,
            no_aug_epochs=15,
        )
    - label: CLI
      language: bash
      code: |
        # CLI는 mosaic_prob를 mosaic으로, mixup_prob를 mixup으로 표기합니다.
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          epochs=100 mosaic=1.0 mixup=0.15 hsv_prob=1.0 \
          flip_prob=0.5 no_aug_epochs=15
  support:
    - label: 계열을 위한 지원표를 읽으십시오
      language: python
      code: |
        from libreyolo.data.augment.spec import AUG_KNOBS, aug_support

        for knob, description in AUG_KNOBS.items():
            support = aug_support("yolo9")[knob]
            print(f"{knob:16} {support.status:16} {support.note or description}")
    - label: 그냥 무시당한 사람들
      language: python
      code: |
        from libreyolo.data.augment.spec import ignored_aug_params

        print(sorted(ignored_aug_params("rfdetr")))
  classify:
    - label: 분류 팩
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreConvNeXtt-cls.pt")
        model.train(
            data="my-classification-dataset",
            epochs=50,
            auto_augment="randaugment",
            erasing=0.25,
            mixup=0.2,
            cutmix=0.2,
        )
source_hash: 47461cd13aab580c
---

## 노브 설정

증강 노브는 일반적인 `train()` 인수입니다.

<code-tabs name="train" />

그 중 두 개는 CLI 철자가 더 짧습니다: `mosaic`는 `mosaic_prob`에 매핑되고 `mixup`는 `mixup_prob`에 매핑됩니다. 다른 모든 노브는 두 곳 모두에서 동일하게 철자됩니다.

## 두 개가 아니라 세 개의 상태

노브가 어떤 기능을 하는지는 계열에 따라 다릅니다. 라이브러리는 이에 대한 선언적 테이블을 유지하며, 각 항목은 세 가지 상태 중 하나입니다.

`used`는 노브가 파이프라인에 도달하여 샘플을 변경함을 의미합니다. `ignored`는 절대 파이프라인에 도달하지 않으므로 설정해도 아무런 효과가 없습니다. `gated_by_mosaic`는 모자이크 분기를 거친 샘플에만 적용되므로 `mosaic_prob=0`와 함께 연결되어 있어도 절대 작동하지 않습니다.

세 번째 상태는 사람들을 놀라게 하는 것입니다. YOLOX 스타일 파이프라인에서는 어파인 워프가 모자이크 캔버스에서 실행되고 MixUp이 모자이크 샘플을 블렌드하므로 `mosaic_prob=0`는 `degrees`, `translate`, `shear`, `perspective`, `mosaic_scale`, `mixup_prob`, `mixup_scale`를 동시에 조용히 비활성화합니다. 트레이너는 MixUp 경우에 대해 특별히 경고를 로그로 남깁니다:

```text
mixup_prob=0.15 has no effect for YOLOv9: mixup only applies to mosaic samples
and mosaic_prob=0. Set mosaic_prob > 0 to enable mixup.
```

CLI는 무시된 노브에 대해서도 경고하며, 실제로 입력한 것만 나열합니다:

```text
Warning: RF-DETR ignores these parameters: degrees, mosaic
```

## 네 가지 파이프라인 모양

계열들은 네 가지 학습 경로로 모이며, 그 경로가 거의 모든 답을 결정합니다.

YOLOX 스타일 모자이크 파이프라인은 샘플마다 HSV 지터와 플립을 적용한 후, 모자이크 분기 내에서 어핀 변환과 MixUp을 실행합니다. 이는 YOLOX, YOLOv7, YOLOv9 및 그 E2E와 P2 변형, RTMDet, PicoDet, RT-DETR, RT-DETRv2 및 FOMO를 포함합니다.

DETR 스타일의 패스스루 파이프라인에는 모자이크나 아핀 왜곡이 없습니다. 포토메트릭 왜곡, 줌아웃 및 IoU 크롭은 설정 조정이 아니라 레시피 상수이므로 `flip_prob`와 `no_aug_epochs`만 활성화됩니다. 이는 D-FINE, Dome-DETR, DEIM, DEIMv2, RT-DETRv4, EC 및 한 가지 변경 사항과 함께 RF-DETR을 포함합니다.

분류 ImageFolder 파이프라인은 모든 탐지 노브를 무시합니다. 그 수평 플립은 고정된 0.5이며 `flip_prob`는 도달하지 못합니다. 대신 아래에 설명된 자체 노브 팩을 가지고 있습니다.

YOLO-NAS는 자체적인 형태를 가지고 있습니다: 모자이크는 전혀 없고, 항상 켜져 있는 개별 샘플 아핀(affine), 그리고 MixUp은 게이트 방식이 아닌 독립적으로 적용됩니다. 그 `mosaic_scale` 값은 아핀 스케일 범위로 재사용됩니다.

SegFormer와 NAFNet은 각각 임의성을 구성할 수 있는 것이 아니라 동일 계열 내에서 고정된 작업별 파이프라인을 실행합니다. SegFormer의 경우 조정 가능한 항목은 `mosaic_scale`와 `hsv_prob`가 아니라 클래스 속성 `semantic_scale_jitter`와 `semantic_hsv_prob`입니다. NAFNet의 크롭과 플립은 고정된 0.5 확률로 입력과 타깃 연산이 결합된 형태입니다.

## 어떤 계열이 어떤 손잡이를 기리는가

아래 표는 `libreyolo/data/augment/spec.py`에서 출하된 사양으로, 라이브러리 자체 테스트에 의해 실제 파이프라인 배관과 비교 검증됩니다. 아키텍처에서 추론하지 말고 거기에서 확인하십시오.

<code-tabs name="support" />

기본 노브에 대해 파이프라인별로 요약:

| 손잡이 | YOLOX 스타일 | YOLO-NAS | DETR 스타일 | 분류 |
|---|---|---|---|---|
| `mosaic_prob` | 사용된 | 무시된 | 무시된 | 무시된 |
| `mixup_prob` | 모자이크로 차단된 | 사용된 | 무시된 | 무시된 |
| `hsv_prob` | 사용된 | 사용된 | 무시된 | 무시된 |
| `flip_prob` | 사용된 | 사용된 | 사용된 | 무시된 |
| `flipud` | 사용된 | 사용된 | 무시된 | 무시된 |
| `degrees` | 모자이크로 차단된 | 사용된 | 무시된 | 무시된 |
| `translate` | 모자이크로 차단된 | 사용된 | 무시된 | 무시된 |
| `shear` | 모자이크로 차단된 | 사용된 | 무시된 | 무시된 |
| `perspective` | 모자이크로 차단된 | 사용된 | 무시된 | 무시된 |
| `mosaic_scale` | 모자이크로 차단된 | 사용된 | 무시된 | 무시된 |
| `mixup_scale` | 모자이크로 차단된 | 사용된 | 무시된 | 무시된 |
| `no_aug_epochs` | 사용된 | 사용된 | 사용된 | 사용된 |

그 열 안의 예외들, 모두 좁아지고 있는:

- RTMDet, PicoDet, RT-DETR, RT-DETRv2 및 FOMO에는 수직 뒤집기가 없으므로 `flipud`는 무시됩니다. FOMO의 모자이크 래퍼 또한 원근 없이 제작되었습니다.
- RF-DETR의 기본 파이프라인에는 HSV 지터가 없으므로, DETR 스타일 열 위에서는 `hsv_prob`가 무시됩니다.
- EC는 `hsv_prob`, `degrees` 및 `translate`를 인정하지만, 오직 `task="pose"`에 대해서만, 그 키포인트 인식 변환이 이들을 읽습니다. 그 검출 및 분할 경로는 고정된 광학 레시피를 사용합니다.
- DINOv2는 DETR 스타일의 컬럼을 탐지 및 의미 작업에 따라 따르며 `task="classify"`를 위한 분류 팩을 추가합니다.

`no_aug_epochs`는 어디에서나 `used`이지만, 모든 곳에서 같은 의미를 가지는 것은 아닙니다. 모자이크 파이프라인에서는 최종 에포크에서 모자이크와 MixUp을 끕니다. DETR 스타일 파이프라인에서는 포토메트릭, 줌아웃, 크롭 증강을 중단하고 스케줄의 꼬리를 조정합니다. 분류 및 시맨틱 파이프라인에서는 단지 꼬리만 조정합니다.

## 분류 팩

네 개의 노브가 분류 파이프라인을 구동하며 그 외에는 아무것도 구동하지 않습니다. 탐지 계열은 이 네 개 모두를 무시합니다.

<code-tabs name="classify" />

`auto_augment`는 `"randaugment"`, `"autoaugment"`, `"augmix"` 또는 `None`를 사용합니다. `erasing`는 RandomErasing 확률입니다. `mixup`와 `cutmix`는 소프트 레이블을 생성하는 배치별 확률입니다; 배치당 최대 하나만 실행되며, MixUp이 먼저 실행되므로 두 확률은 더해질 수 있으며 합이 최대 1이 되어야 합니다.

모두 기본적으로 꺼져 있으므로 요청하지 않는 한 분류 학습은 변경되지 않습니다.

한 가지 명명 충돌은 분명히 언급할 가치가 있습니다: CLI에서 `mixup`는 탐지 `mixup_prob`의 별칭입니다. 분류 `mixup` 필드는 자체적인 CLI 철자가 없으며 Python에서 `model.train(mixup=...)`를 통해서만 접근할 수 있습니다.

## 계열별 조절기

일부 노브는 베이스 클래스가 아니라 계열의 설정 하위 클래스에 존재하므로, 해당 계열에만 존재하며 CLI 플래그가 없습니다.

| 계열 | 손잡이 | 효과 |
|---|---|---|
| YOLOv9, YOLOv9-E2E, YOLOv9-P2 | `copy_paste` | 복사-붙여넣기 인스턴스 증강 확률, `task="segment"`만 |
| YOLOv9, YOLOv9-E2E, YOLOv9-P2 | `copy_paste_mode` | `"flip"`는 동일한 샘플을 반사하여 재사용하고, `"mixup"`는 두 번째 샘플을 가져옵니다 |
| YOLOv9, YOLOv9-E2E, YOLOv9-P2 | `rot90` | 무작위 90도 회전 확률 |
| YOLOv9 | `max_labels` | 학습 변환에서 이미지별 실제 캡, 기본값 100 |
| RF-DETR | `copy_paste`, `copy_paste_mode` | `task="segment"`, `"flip"` 모드에서만 복사-붙여넣기 |
| RF-DETR, D-FINE, EC | `crop_resize_prob` | 랜덤 자르기-크기 조정 확률 |
| EC, YOLO-NAS | `brightness_contrast_prob`, `affine_prob` | 포즈 경로 지터 및 키포인트 인식 아핀 확률 |

`max_labels`는 조용히 데이터를 잃는 장치입니다. 캡을 넘어선 박스는 오류 없이 삭제되므로, 항공 사진과 같은 밀집된 이미지에는 이를 높여야 합니다.

회전된 상자에 대한 코너 인식 증강이 구현되지 않았기 때문에, 회전 상자 학습에서는 설정에 관계없이 모자이크와 MixUp이 비활성화됩니다.

## 관련된

- `no_aug_epochs`의 스케줄 인수와 나머지 `train()`에 대한 [하이퍼파라미터](/docs/train/hyperparameters).
- [데이터셋](/docs/train/datasets)은 이러한 변환이 사용하는 레이블 형식입니다.
