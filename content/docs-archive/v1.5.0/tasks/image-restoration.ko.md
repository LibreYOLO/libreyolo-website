---
title: 이미지 복원
seo_title: LibreYOLO에서 이미지 복원 및 업스케일링
description: >-
  LibreYOLO에서 이미지를 노이즈 제거, 블러 제거 및 업스케일하십시오. 복원된 RGB 이미지를 예측하고, 페어 데이터로 NAFNet을
  학습하며, PSNR 및 SSIM 키를 읽으십시오.
lead: >-
  이미지 복원은 손상된 이미지를 받아서 깨끗한 이미지로 반환합니다. LibreYOLO는 이를 복원 작업으로 제공하며, 하나의 출력 계약으로
  노이즈 제거, 블러 제거, 초해상도를 모두 포함합니다: RGB 이미지 하나 입력, RGB 이미지 하나 출력.
keywords:
  - 이미지 복원 파이썬
  - 이미지 노이즈 제거 모델
  - 이미지 초해상도 파이썬
  - 디블러링 모델
  - PSNR SSIM 검증
last_verified: 1.5.0
snippets:
  predict:
    - label: 이미지 확대
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 컴팩트한 4x 제너레이터; 큰 소스에서 타일 경계가 최대 메모리를 사용합니다.
        model = LibreYOLO("LibreRealESRGANx4t-restore.pt")
        result = model(SAMPLE_IMAGE, tile=512, tile_pad=10)

        result.restored.save("upscaled.png")
        print(result.restored.array.shape)   # 각 축에서 입력의 4배
    - label: 이미지의 노이즈 제거
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # SIDD 실제 이미지 노이즈로 학습됨; 출력은 입력 크기를 유지합니다.
        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")
        result = model(SAMPLE_IMAGE)

        result.restored.save("denoised.png")
        print(result.restore_scale)   # 1: 이 체크포인트에는 업스케일 없음
  train:
    - label: Paired 이미지에서 NAFNet을 파인튜닝하기
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")

        model.train(data="my-dataset.yaml", epochs=100, imgsz=256, batch=16,
        lr0=1e-3)
    - label: 체크포인트에 출처를 기록하십시오
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")

        # 저하와 데이터셋이 저장된 체크포인트에 기록됩니다
        # 출처; 그들은 학습에 참여하지 않습니다.
        model.train(
            data="my-dataset.yaml",
            epochs=100,
            degradation="denoise",
            dataset="MyDataset",
        )
  val:
    - label: 메트릭 키를 검증하고 읽습니다
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")

        # val()는 객체가 아니라 일반 딕셔너리를 반환합니다.
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/PSNR"])   # 체력
        print(metrics["metrics/SSIM"])
  export:
    - label: 내보내기
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")

        # imgsz는 그래프에 고정되어 있으므로, 배포 시 크기를 전달하십시오
        # 실제로 모델에 데이터를 공급합니다.
        model.export(format="onnx", imgsz=256)
    - label: 내보낸 파일을 실행하십시오
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 팩토리는 파일 접미사에 따라 경로를 지정하므로, 내보낸 아티팩트가 로드됩니다
        # 모든 체크포인트와 마찬가지로 같은 Results 객체를 반환합니다.
        model = LibreYOLO("LibreNAFNetl-restore-sidd.onnx")
        result = model(SAMPLE_IMAGE)

        result.restored.save("denoised.png")
source_hash: 9dc81cadb3ebf18b
---

## 정의

`restore` 작업은 하나의 이미지를 다른 이미지로 매핑합니다. 여기에 있는 노이즈 제거, 블러 제거 및 초고해상도는 모두 같은 작업입니다. 왜냐하면 이들은 한 가지 계약을 공유하기 때문입니다: 모델은 RGB 이미지를 입력으로 받고 RGB 이미지를 반환하며, 모델이 학습된 열화(degradation)를 되돌리는 기능은 API가 아니라 체크포인트의 속성입니다.

예측은 `result.restored`를 채우며, `RestoredImage` 페이로드에는 `(H, W, 3)` uint8 RGB 배열이 들어 있습니다. `.array`는 이를 NumPy로 반환하고 `.save(path)`는 디스크에 기록합니다. `result.restore_scale`는 출력 캔버스가 갖는 업스케일 요소를 기록하며, 이는 해상도를 보존하는 체크포인트의 경우 `1`입니다. `result.boxes`는 비어 있으므로 `conf`, `iou`, `max_det`는 서명 일치성에는 허용되지만 아무런 영향을 주지 않으며, `save=True`는 복원된 이미지를 주석이 달린 사진 대신 직접 기록합니다.

## 모델들

세 계열이 `restore`를 섬기며, 그들이 되돌리는 손상에 따라 나뉩니다.

[NAFNet](/docs/models/nafnet)은 디노이저이며, LibreYOLO가 학습할 수 있는 유일한 복원 계열입니다. 그 아키텍처는 UNet 블록의 비선형 활성화를 요소별 곱셈으로 대체하며, 공개된 체크포인트는 SIDD 실제 이미지 노이즈로 학습되었습니다. 출력은 입력 해상도를 유지합니다.

[Real-ESRGAN](/docs/models/real-esrgan)은 실용적인 업스케일러입니다: 오직 바이큐빅 다운스케일링이 아니라 합성 열화에 대해 학습된 세 가지 체크포인트가 있으며, 4배, 2배, 그리고 낮은 지연 시간을 위해 만들어진 더 작고 빠른 4배 생성기가 포함되어 있습니다.

[SwinIR](/docs/models/swinir)은 Swin Transformer 백본을 사용하여 4배 업스케일하며, 공식 경량 생성기와 두 개의 실제 생성기를 포함하는 세 가지 크기로 제공됩니다.

## 예측

가중치는 처음 사용할 때 Hugging Face에서 다운로드되며 로컬에 캐시됩니다.

<code-tabs name="predict" />

복원은 고정된 네트워크 캔버스가 아닌 원본 이미지의 해상도로 실행되며, 네트워크의 다운샘플링 비율에 맞춰 패딩만 적용되므로 시간과 메모리는 입력 이미지의 픽셀 수에 따라 결정됩니다. `tile`는 순방향 계산을 겹치는 타일로 나누고 이음새를 다시 합치며, `tile_pad`는 각 타일을 다시 잘라내기 전에 추가되는 가장자리 영역입니다. 두 가지 모두 Python 키워드 인자입니다. 소스 코드, 스트리밍 및 결과 처리에 대해서는 [prediction](/docs/predict)를 참조하십시오.

## 데이터셋 형식

복원은 각 손상된 입력 이미지를 파일 이름의 접두사로 맞춘 동일한 해상도의 깨끗한 목표 이미지와 페어링합니다.

```text
dataset/
  data.yaml
  inputs/
    train/photo.jpg
    val/photo.jpg
  targets/
    train/photo.jpg
    val/photo.jpg
```

```yaml
path: dataset
train: inputs/train
val: inputs/val
input_dir: inputs
target_dir: targets
degradation: denoise
dataset: MyDataset
nc: 1
names: {0: image}
```

`nc`와 `names`는 스키마 자리 표시자입니다; 복원 모델은 `Results.restored`를 반환하며, 탐지는 반환하지 않습니다. `degradation`와 `dataset`는 선택적 출처 레이블입니다. `target_stem_suffix`는 깨끗한 이미지의 이름이 열화된 쌍과 다르게 지정된 데이터셋을 포함합니다. 검증은 원본 해상도를 유지하며 배치를 쌓기 위해 필요한 만큼만 패딩하므로, 지표는 원본 캔버스에서 계산됩니다. 전체 계약 내용은 [dataset formats](/docs/reference/dataset-formats)을 참조하십시오.

## 학습

NAFNet은 학습 구현이 있는 유일한 복원 계열입니다. `Real-ESRGAN.train()`와 `SwinIR.train()`는 모두 `NotImplementedError`를 발생시킵니다: 해당 체크포인트는 합성 열화 파이프라인을 통한 GAN 학습에서 나온 것이며, 페어 복원 트레이너는 그 레시피를 재현하지 않고 실행됩니다.

<code-tabs name="train" />

트레이너는 입력과 타겟 쌍의 결합된 크롭을 가져오므로 양쪽이 정렬된 상태를 유지합니다. 데이터셋, 멀티 GPU 및 로거에 대해서는 [training](/docs/train)를 참조하고, 이 계열의 기본값과 학습 중에 분리하는 추론 시 풀링에 대해서는 [NAFNet 페이지](/docs/models/nafnet)를 참조하십시오.

## 검증

`val()`는 복원된 출력을 원본 캔버스에서 RGB로 깨끗한 목표와 비교하며, 테두리 자르기나 크기 조정 없이 수행합니다.

<code-tabs name="val" />

`metrics/PSNR`는 데시벨 단위의 최대 신호 대 잡음 비율이며, 또한 `fitness`는 최상의 체크포인트 선택 횟수입니다. `metrics/SSIM`는 `[0, 1]`에서 구조적 유사성으로, 시그마 1.5의 11x11 가우시안 창으로 계산하고 세 가지 색상 채널에 대해 평균을 냈습니다. 두 가지 모두 값이 클수록 좋습니다.

## 내보내기

내보낸 복원 모델은 파일 접미사를 통해 `LibreYOLO()`로 다시 로드되므로, `.onnx` 또는 `.engine` 파일은 체크포인트처럼 작동하며 동일한 `Results`를 반환하고, `restored`는 출력 이미지를 담고 있습니다.

<code-tabs name="export" />

Restore export는 그래프에서 공간 해상도를 고정하므로, `imgsz`를 전달하면 실제로 배포가 모델에 입력됩니다. NAFNet의 경우 해당 크기는 네트워크의 다운샘플링 계수로 나눠떨어져야 하며, `dynamic=True` 하에서는 배치 차원만 동적으로 유지됩니다. Real-ESRGAN과 SwinIR의 경우 `imgsz`를 생략하면 작업 해상도 대신 작은 내부 패치 크기로 기본 설정됩니다. 형식별 지원 범위는 각 모델 페이지와 [전체 내보내기 매트릭스](/docs/reference/export-matrix)에서 확인할 수 있습니다. [Export](/docs/export)에서는 각 형식이 허용하는 인수를 나열합니다.
