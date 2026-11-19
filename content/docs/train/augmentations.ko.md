---
title: 데이터 증강
seo_title: LibreYOLO의 학습 데이터 증강과 예시
description: >-
  LibreYOLO의 각 데이터 증강 노브가 하는 일을 라이브러리 자체의 학습 코드로 실제 이미지에 적용해 보여 주고, 범위와 기본값, 그리고
  해당 노브를 읽는 계열을 함께 정리합니다.
lead: >-
  학습 데이터 증강은 모든 학습 샘플에 적용되는 무작위 변환입니다. TrainConfig의 노브로 설정하며, 모델 계열마다 자체 파이프라인을
  실행하므로 노브는 해당 파이프라인이 그 노브를 읽는 경우에만 샘플을 바꿉니다.
keywords:
  - yolo 데이터 증강
  - 모자이크 증강
  - mixup
  - hsv 증강
  - random affine
  - random erasing
  - 작은 객체 탐지
  - 데이터 증강 예시
  - copy paste 증강
  - randaugment
  - cutmix
  - no_aug_epochs
last_verified: 1.8.0
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
        # CLI에서는 mosaic_prob을 mosaic로, mixup_prob을 mixup으로 씁니다.
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          epochs=100 mosaic=1.0 mixup=0.15 hsv_prob=1.0 \
          flip_prob=0.5 no_aug_epochs=15
  small:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # YOLOX는 mosaic_scale=(0.1, 2.0)에서 시작하므로 객체가 원래 크기의
        # 10분의 1까지 줄어들 수 있습니다. (0.5, 1.5)는 절반에서 멈춥니다.
        model = LibreYOLO("LibreYOLOXs.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=100,
            mosaic_scale=(0.5, 1.5),
        )
    - label: CLI
      language: bash
      code: |
        # CLI에서 쌍은 쉼표로 구분한 두 숫자로 씁니다.
        libreyolo train model=LibreYOLOXs.pt data=my-dataset.yaml \
          epochs=100 mosaic_scale=0.5,1.5
  support:
    - label: 계열의 지원 표 읽기
      language: python
      code: |
        from libreyolo.data.augment.spec import AUG_KNOBS, aug_support

        for knob, description in AUG_KNOBS.items():
            support = aug_support("yolo9")[knob]
            print(f"{knob:16} {support.status:16} {support.note or description}")
    - label: 무시되는 노브만
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
source_hash: 72d22a38d647de0a
---

## 노브 설정

데이터 증강 노브는 일반적인 `train()` 인수입니다.

<code-tabs name="train" />

그중 세 개에는 `train()`과 CLI가 똑같이 받아들이는 더 짧은 표기가 있습니다: `mosaic`, `fliplr`, 그리고 탐지에서의 `mixup` 표기는 각각 `mosaic_prob`, `flip_prob`, `mixup_prob`에 대응합니다. 짧은 표기와 해당 필드를 서로 다른 값으로 함께 전달하면 예외가 발생합니다. 1.8.0에서 `perspective`는 CLI 플래그가 없으므로 Python에서 설정합니다.

분류 노브는 해당 파이프라인을 만들 때 검증되며, CLI는 `flip_prob`, `flipud`, `cutmix`, `erasing`도 검사합니다. `TrainConfig`는 탐지 노브의 범위를 검사하지 않으므로, `degrees=100`으로 설정하면 최대 100도까지 회전하며 학습합니다. [데이터 증강 매트릭스](/docs/reference/augmentation-matrix)에 모든 노브의 범위가 나와 있습니다.

## 두 가지가 아닌 세 가지 상태

노브가 실제로 작동하는지는 계열에 따라 다릅니다. 라이브러리는 이를 선언적 표로 관리하며, 각 항목은 세 가지 상태 중 하나입니다.

`used`는 노브가 파이프라인에 도달해 샘플을 바꾼다는 뜻입니다. `ignored`는 노브가 파이프라인에 도달하지 않아 설정해도 아무 효과가 없다는 뜻입니다. `gated_by_mosaic`는 노브가 모자이크 분기를 거친 샘플에만 적용된다는 뜻이므로, 연결되어 있더라도 `mosaic_prob=0`이면 한 번도 작동하지 않습니다.

사람들이 의외라고 느끼는 것은 바로 세 번째 상태입니다. YOLOX 방식 파이프라인에서는 아핀 변환이 모자이크 캔버스에서 실행되고 MixUp이 모자이크 샘플을 혼합하므로, `mosaic_prob=0`은 `degrees`, `translate`, `shear`, `perspective`, `mosaic_scale`, `mixup_prob`, `mixup_scale`을 아무 알림 없이 한꺼번에 비활성화합니다. 트레이너는 MixUp의 경우에 한해 다음 경고를 로그로 남깁니다:

```text
mixup_prob=0.15 has no effect for YOLOv9: mixup only applies to mosaic samples
and mosaic_prob=0. Set mosaic_prob > 0 to enable mixup.
```

CLI도 무시되는 노브에 대해 경고하며, 실제로 입력한 노브만 나열합니다:

```text
Warning: RF-DETR ignores these parameters: degrees, mosaic
```

Python에서 호출하면 `train()`은 무시되는 노브에 대해 경고하지 않습니다. [데이터 증강 매트릭스](/docs/reference/augmentation-matrix)에는 모든 계열에 대한 모든 노브의 상태와 각 계열의 시작값이 정리되어 있습니다. 이 표는 `libreyolo/data/augment/spec.py`에 포함되어 배포되는 사양으로, 라이브러리 자체 테스트가 파이프라인과 대조해 확인하며, 다음과 같이 직접 읽을 수도 있습니다:

<code-tabs name="support" />

## 각 노브의 역할

아래 그림은 LibreYOLO 1.8.0 자체의 학습 변환을 이미지 한 장짜리 데이터셋을 통해 샘플 이미지에 실행해 만든 것이므로, 모든 모자이크 타일과 모든 MixUp 상대 샘플이 같은 사진입니다. 박스는 LibreYOLO9t가 이 이미지에서 탐지한 사람입니다. 각 패널 아래에는 설정과 가장 큰 박스의 크기를 640 픽셀 학습 해상도 기준의 픽셀 단위로 표시합니다. 데이터 증강을 적용하지 않으면 이 사람의 크기는 291 × 217 px입니다.

몇몇 기하 노브는 범위입니다: `degrees=10`은 -10도에서 +10도 사이 어디서든 추출한 각도를 뜻합니다. 이 노브들의 패널은 추출값을 캡션의 값으로 고정하므로, 전형적인 샘플이 아니라 해당 설정이 허용하는 가장 강한 효과를 보여 줍니다. 탐지 그림은 YOLOv9 파이프라인에서 나온 것입니다. YOLOv7은 이 파이프라인을 그대로 사용해 학습하고, RT-DETR, RT-DETRv2는 이 파이프라인의 모자이크 래퍼에 자체 샘플별 변환을 결합해 사용합니다. YOLOX, RTMDet, PicoDet, FOMO는 별도의 래퍼에서 같은 방식으로 모자이크와 아핀 변환을 구성하지만, 그 래퍼의 MixUp은 다릅니다.

### mosaic_prob

샘플이 모자이크가 될 확률입니다: 입력 크기의 두 배인 캔버스에 이미지 네 장을 배치하고, 이후 아핀 변환이 이를 입력 크기로 축소해 매핑합니다. YOLOX 방식 파이프라인에서는 아핀 변환과 MixUp이 모자이크 샘플에서만 실행되므로, 이 값이 0이면 둘 다 멈춥니다. RT-DETR, RT-DETRv2, PicoDet, FOMO는 기본값이 0입니다.

<aug-figure name="mosaic" />

두 번째 패널은 아핀 변환의 스케일을 캔버스 전체가 들어가는 0.5×로 고정합니다. 마지막 두 패널은 YOLOv9 기본값으로 뽑은 일반적인 추출이며, 캡션에 각각 추출된 스케일을 표시합니다.

### mosaic_scale

아핀 변환의 배율을 추출하는 범위입니다. 1×에서는 객체가 추론 시의 크기를 유지하고, 0.5×에서는 그 절반이 됩니다. 베이스 기본값은 `(0.1, 2.0)`이며, YOLOX, YOLOv7, RTMDet, PicoDet, RT-DETR, RT-DETRv2, FOMO는 이 값을 그대로 유지하므로 이들 계열의 객체는 원래 크기의 10분의 1까지 줄어들 수 있습니다. YOLOv9의 기본값은 `(0.5, 1.5)`입니다. YOLO-NAS와 PP-YOLOE는 모자이크가 없으며, 이 범위를 샘플별 아핀 변환의 스케일로 재사용합니다.

<aug-figure name="mosaic_scale" />

1×보다 크면 출력에는 캔버스의 왼쪽 위 부분만 나타나므로, 객체가 가장자리에서 잘리거나 프레임 밖으로 나갑니다.

### degrees

회전 범위입니다: 아핀 변환되는 각 샘플은 `-degrees`에서 `+degrees` 사이에서 추출한 각도만큼 회전합니다. 회전의 중심은 캔버스의 왼쪽 위 모서리이므로, 각도가 크면 이미지의 상당 부분이 프레임 밖으로 돌아 나갑니다. YOLOX와 YOLOv7은 베이스 기본값 10을 유지하고, YOLOv9의 기본값은 0입니다.

<aug-figure name="degrees" />

### translate

입력 한 변에 대한 비율로 나타낸 이동 범위입니다: `imgsz=640`에서 아핀 변환되는 각 샘플은 각 축으로 최대 `translate`와 640 px의 곱만큼 이동합니다. 베이스 기본값은 0.1입니다. YOLO-NAS와 PP-YOLOE의 기본값은 0.25이고, RTMDet, PicoDet, FOMO의 기본값은 0입니다.

<aug-figure name="translate" />

### shear

도 단위의 전단 범위로, 축마다 따로 추출합니다. YOLOX와 YOLOv7은 베이스 기본값 2를 유지하고, 이 노브를 읽는 다른 모든 계열의 기본값은 0입니다.

<aug-figure name="shear" />

### perspective

회전, 스케일, 전단, 이동 위에 더해지는 사영 변환입니다. 캔버스의 각 모서리는 축마다 최대 캔버스 한 변의 절반에 `min(perspective * 100, 0.2)`를 곱한 만큼의 무작위 양으로 안쪽으로 이동하므로, `perspective=0.001`은 어떤 해상도에서든 모서리를 최대 한 변의 5퍼센트만큼 움직이고, `0.002` 이상의 모든 값은 같은 최댓값에 도달합니다. 기본값 `0`은 변환을 아핀으로 유지하며, 음수 값은 첫 샘플을 변환할 때 `ValueError`를 발생시킵니다. 패널은 한 번의 무작위 추출을 공유하고, 이를 값에 비례해 조정합니다. YOLOv9과 그 E2E 및 P2 변형, YOLOX, YOLOv7, YOLO-NAS, PP-YOLOE, RTMDet, PicoDet, RT-DETR, RT-DETRv2가 이를 적용합니다.

<aug-figure name="perspective" />

### hsv_prob

HSV 색상 지터의 확률입니다. 노브로 설정할 수 있는 것은 확률뿐입니다. 지터는 OpenCV의 0에서 179까지의 색상 척도에서 색상을 최대 5만큼, 0에서 255까지의 척도에서 채도와 명도를 최대 30만큼 이동하며, 세 가지 이동은 각각 2분의 1 확률로 건너뜁니다. `hsv_prob`을 읽는 모든 탐지 파이프라인이 이 양을 사용합니다. 마지막 세 패널은 119번의 추출 가운데 변화가 가장 잘 보이는 세 개입니다.

<aug-figure name="hsv" />

### flip_prob 및 flipud

수평 뒤집기와 수직 뒤집기의 확률입니다. `flip_prob`은 FOMO를 제외한 모든 계열에서 0.5로 시작하며, FOMO의 기본값은 0입니다. `flipud`는 0으로 시작합니다. RTMDet, PicoDet, RT-DETR, RT-DETRv2, FOMO, PP-YOLOE 및 DETR 방식 계열은 수직 뒤집기가 없어 이 노브를 무시합니다. PP-YOLOE의 레시피는 대신 무작위 90도 회전을 사용합니다.

<aug-figure name="flip" />

### mixup_prob 및 mixup_scale

MixUp은 두 번째 샘플을 현재 샘플에 혼합하고 두 샘플의 박스를 모두 유지합니다. YOLOv9 파이프라인에서 상대 샘플은 모자이크가 없는 온전한 샘플이고, 혼합 비율은 0과 1 사이에서 균등하게 추출되며, `mixup_scale`은 라이브러리 사양에 모자이크 종속으로 기재되어 있지만 한 번도 읽히지 않습니다. YOLOX, RTMDet, PicoDet, FOMO가 사용하는 YOLOX 래퍼와 YOLO-NAS의 래퍼에서는 상대 샘플을 먼저 `mixup_scale`에서 추출한 배율로 리스케일한 뒤, 두 이미지를 고정된 0.5 비율로 평균합니다. `mixup_prob`의 베이스 기본값은 1.0이고, YOLOv9의 기본값은 0입니다.

<aug-figure name="mixup" />

마지막 두 패널은 두 번째 샘플을 0.5×의 모자이크에 혼합합니다. 이 패널들의 캡션에는 모자이크 샘플의 가중치인 혼합 비율 λ를 표시합니다.

### no_aug_epochs

`no_aug_epochs`는 모든 곳에서 `used`이지만, 모든 곳에서 같은 의미는 아닙니다. 모자이크 파이프라인에서는 마지막 에폭 동안 모자이크와 MixUp을 끕니다. DETR 방식 파이프라인에서는 광도, 줌아웃, 크롭 증강을 멈추고 스케줄의 마지막 구간을 결정합니다. 분류 파이프라인에서는 자동 증강, 지우기, MixUp, CutMix를 비활성화하고, 크롭과 뒤집기는 유지됩니다.

## DETR 방식 레시피

D-FINE, Dome-DETR, DEIM, DEIMv2, RT-DETRv4, TinyFormer는 모자이크도 아핀 변환도 없는 하나의 torchvision 변환으로 학습하며, EdgeCrafter(EC)도 탐지에서는 마찬가지입니다. 이 변환의 세 가지 강한 연산은 노브가 아니라 레시피 상수입니다:

- `RandomPhotometricDistort`는 각 색상 변화(밝기 0.875에서 1.125, 대비와 채도 0.5에서 1.5, 색상 ±0.05, 채널 섞기)를 각각 0.5의 확률로 적용합니다.
- `RandomZoomOut`은 0.5의 확률로 이미지를 이미지 한 변의 최대 네 배 크기인 검은 캔버스에 배치하므로, 객체가 원래 크기의 4분의 1까지 줄어들 수 있습니다.
- `RandomIoUCrop`은 0.8의 확률로 적용되며, 각 변이 원래의 0.3에서 1.0배인 영역으로 크롭하고, 이 영역은 박스의 적어도 일부가 남도록 선택됩니다.

이 단계를 거친 뒤 어느 한 변이라도 1 px 이하인 박스는 제거됩니다. 세 연산 모두 해당 계열의 중단 에폭에서 멈추며, 늦어도 마지막 `no_aug_epochs` 에폭이 시작될 때 멈춥니다. 이 파이프라인이 읽는 노브는 `flip_prob`과 `no_aug_epochs`뿐입니다.

<aug-figure name="detr_ops" />

D-FINE 학습 초반에 세 연산이 모두 활성화된 상태로 나온 샘플 세 개입니다:

<aug-figure name="detr_samples" />

RF-DETR에는 자체 파이프라인이 있습니다: 수평 뒤집기를 적용하고, 이어서 `crop_resize_prob` 확률로 짧은 변을 400, 500, 600 px 중 하나로 리사이즈한 뒤 384에서 600 px 크기의 정사각형을 무작위로 크롭하며, 마지막으로 정사각형 학습 해상도로 바로 리사이즈하는데, 이 리사이즈는 레터박스가 아니라 이미지를 늘립니다. `crop_resize_prob`의 기본값은 0.5이며 Python에서 설정합니다.

<aug-figure name="rfdetr_crop" />

## 작은 객체

`erasing`은 탐지에 적용되지 않습니다. Random Erasing은 분류 팩에 속하고 지우기 단계가 있는 탐지 파이프라인은 없으므로, 이 값이 무엇이든 탐지 모델의 학습은 똑같으며 끌 것도 없습니다. 작은 객체를 축소하거나 제거하는 단계는 다음과 같습니다:

- `mosaic_scale`. 모자이크 파이프라인에서는 객체의 크기에 추출된 스케일이 곱해지므로, 하한이 객체의 최소 크기를 정합니다. 베이스 기본값 `(0.1, 2.0)`에서는 20 px 객체가 2 px 크기가 될 수 있습니다. 이어서 YOLOv9 파이프라인은 어느 한 변이라도 2 px 이하인 박스를 제거하고, 그 최종 변환도 YOLOX의 최종 변환과 마찬가지로 어느 한 변이라도 1 px 이하인 박스를 제거합니다. 객체의 픽셀은 레이블 없이 이미지에 남습니다. YOLOv9의 `(0.5, 1.5)`처럼 하한을 0.5로 두면 모든 객체가 추론 크기의 절반 이상으로 유지됩니다.
- `degrees`, `translate`, `shear`, `perspective`는 객체를 가장자리로 밀어내며, 그곳에서 박스가 잘리고 같은 크기 필터를 거칩니다.
- DETR 방식 레시피에서는 `RandomZoomOut`이 객체를 원래 크기의 4분의 1까지 축소할 수 있고, `RandomIoUCrop`이 객체를 잘라 낼 수 있습니다. 둘 다 노브가 아니며, 둘 다 중단 에폭에서 멈춥니다.
- RF-DETR의 크롭은 크롭 가장자리에서 객체를 자를 수 있습니다. `crop_resize_prob=0`이면 이 크롭이 꺼집니다.
- 학습 해상도는 모든 객체의 크기를 바꿉니다: 레터박스는 `imgsz`를 이미지의 긴 변으로 나눈 비율로 이미지를 리사이즈하므로, `imgsz=1280`에서는 모든 객체의 너비와 높이가 640일 때의 두 배가 됩니다. [YOLOv9](/docs/models/yolov9)에는 작은 객체를 위한 스트라이드 4 헤드가 있는 P2 변형도 있습니다.

<code-tabs name="small" />

## 분류 팩

여섯 개의 노브는 분류 파이프라인에만 작용합니다. 탐지 계열은 여섯 개 모두를 무시합니다.

<code-tabs name="classify" />

`scale`은 무작위 크롭의 면적 범위입니다. `scale=0.5`는 `(0.5, 1.0)`을 뜻하며, 쌍을 명시하면 양쪽 경계를 모두 설정합니다. 세 번째 패널처럼 하한이 작으면 객체가 완전히 잘려 나갈 수 있습니다.

<aug-figure name="cls_crop" />

`auto_augment`는 `"randaugment"`, `"autoaugment"`, `"augmix"`, `None` 중 하나를 받으며, 각 샘플에 해당 정책에서 한 번 추출한 변환을 적용합니다.

<aug-figure name="auto_augment" />

`erasing`은 Random Erasing의 확률이며, 범위는 `[0, 1)`입니다. 정규화 이후에 실행되며, 이미지의 2에서 33퍼센트를 덮는 직사각형을 0으로 채우는데, 정규화 이후의 0은 ImageNet 평균 색에 해당합니다.

<aug-figure name="erasing" />

`mixup`과 `cutmix`는 소프트 레이블을 생성하는 배치별 확률입니다. 배치마다 최대 하나만 실행되고 MixUp이 먼저 적용되므로, 두 확률은 더해지며 합이 1 이하여야 합니다. 패널에서 A와 B는 같은 사진에서 잘라 낸 두 크롭이며, 각 캡션에는 혼합된 이미지가 학습에 사용하는 소프트 레이블을 표시합니다.

<aug-figure name="batch_mix" />

`auto_augment`, `erasing`, `mixup`, `cutmix`는 기본적으로 꺼져 있으므로, 따로 요청하지 않는 한 분류 학습은 달라지지 않습니다. `train()`과 CLI는 작업에 따라 `mixup`을 전달합니다: 분류는 배치 혼합을 사용하고, 탐지는 `mixup_prob`을 사용합니다.

`crop_pct=None`은 계열의 평가용 리사이즈 비율을 유지합니다. 값을 재정의하면 학습 중에 실행되는 검증의 평가 크롭이 바뀌지만, 내보내기는 계열 고유의 전처리를 유지합니다.

CLI는 `auto_augment`, `erasing`, `cutmix`, `scale`, `crop_pct`, `fliplr`, `flipud`를 제공합니다. 분류의 `mixup`은 배치 혼합을 뜻하며 기본값은 0.0입니다. 수평 뒤집기의 기본값은 0.5, 수직 뒤집기는 0.0입니다.

`no_aug_epochs`는 마지막 에폭 동안 자동 증강, 지우기, MixUp, CutMix를 비활성화하고 크롭과 뒤집기는 유지합니다. 분류 레시피에서 이 마지막 구간의 기본값은 0입니다.

## 계열 전용 노브

일부 노브는 베이스 클래스가 아니라 계열의 설정 하위 클래스에 있으므로, 해당 계열에만 존재하고 CLI 플래그가 없습니다.

| 계열 | 노브 | 효과 |
|---|---|---|
| YOLOv9, YOLOv9-E2E, YOLOv9-P2 | `copy_paste` | copy-paste 인스턴스 증강 확률, `task="segment"` 전용 |
| YOLOv9, YOLOv9-E2E, YOLOv9-P2 | `copy_paste_mode` | `"flip"`은 같은 샘플을 미러링해 재사용하고, `"mixup"`은 두 번째 샘플을 가져옵니다 |
| YOLOv9, YOLOv9-E2E, YOLOv9-P2 | `rot90` | 무작위 90도 회전 확률 |
| YOLOv9, YOLOv9-E2E, YOLOv9-P2 | `max_labels` | 학습 변환에서 이미지당 정답(ground truth) 개수 상한, 기본값 300 |
| RF-DETR | `copy_paste`, `copy_paste_mode` | `task="segment"`용 copy-paste, `"flip"` 모드 전용 |
| RF-DETR | `crop_resize_prob` | 네이티브 파이프라인의 무작위 크롭 리사이즈 확률, 기본값 0.5 |
| D-FINE, EdgeCrafter | `crop_resize_prob` | 무작위 크롭 리사이즈 확률, `task="segment"` 전용 |
| EdgeCrafter, YOLO-NAS | `brightness_contrast_prob`, `affine_prob` | 자세 추정 경로의 지터 확률과 키포인트 인식 아핀 변환 확률 |

데이터를 잃는 노브는 `max_labels`입니다. 상한을 넘는 박스는 오류 대신 한 번만 경고를 남기고 제거되므로, 항공 사진처럼 밀집된 이미지에서는 이 값을 높여야 합니다. 레이블 할당 메모리는 상한이 아니라 배치에서 가장 밀집된 이미지를 따릅니다.

회전 바운딩 박스를 위한 꼭짓점 인식 증강이 구현되어 있지 않으므로, 회전 바운딩 박스 학습에서는 노브 설정과 관계없이 모자이크와 MixUp이 비활성화됩니다.

## 관련 문서

- [데이터 증강 매트릭스](/docs/reference/augmentation-matrix): 모든 계열에 대한 모든 노브의 상태, 범위, 각 계열의 기본값
- [하이퍼파라미터](/docs/train/hyperparameters): 스케줄 인수로서의 `no_aug_epochs`와 `train()`의 나머지 인수
- [데이터셋](/docs/train/datasets): 이 변환이 사용하는 레이블 형식
