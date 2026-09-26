---
title: 배경 제거
seo_title: LibreYOLO에서 배경 제거
description: >-
  LibreYOLO에서 배경에서 피사체를 잘라냅니다. 부드러운 알파 매트를 예측하고, 투명 PNG를 작성하며, MAE와 S-측정으로
  검증합니다.
lead: >-
  배경 제거는 피사체를 그 뒤의 모든 것과 분리합니다. LibreYOLO는 이를 매트 작업으로 노출하며, 이 작업은 단단한 전경 마스크가
  아니라 픽셀마다 부드러운 알파 값을 반환합니다.
keywords:
  - 파이썬 배경 제거
  - 알파 매팅 모델
  - 이분화 이미지 분할
  - 투명 png 컷아웃
  - 소프트 알파 매트
last_verified: 1.5.0
snippets:
  predict:
    - label: 매트를 예측하다
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreBiRefNetl-matte.pt")
        result = model(SAMPLE_IMAGE)

        matte = result.matte
        print(matte.array.shape, matte.array.dtype)   # (H, W) float32 범위 [0, 1]
    - label: 투명한 PNG를 작성하십시오
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreBiRefNetl-matte.pt")
        result = model(SAMPLE_IMAGE)

        # save()는 소스를 매트와 알파 채널로 합성합니다.
        result.save("subject.png")

        rgba = result.cutout()   # 메모리 내 동일한 (H, W, 4) uint8 배열
        print(rgba.shape)
    - label: 새 배경에 합성하기
      language: python
      code: >
        import numpy as np

        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreBiRefNetl-matte.pt")

        result = model(SAMPLE_IMAGE)


        rgba = result.cutout()

        alpha = rgba[..., 3:4].astype(np.float32) / 255.0

        backdrop = np.full_like(rgba[..., :3], 255)          # 흰색

        composited = (rgba[..., :3] * alpha + backdrop * (1 -
        alpha)).astype(np.uint8)

        print(composited.shape)
  val:
    - label: 메트릭 키를 검증하고 읽습니다
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreBiRefNetl-matte.pt")

        # images/와 matte 디렉토리를 포함하는 디렉토리가 대신 작동합니다
        # 데이터셋 YAML.
        metrics = model.val(data="my-matte-dataset/")

        print(metrics["metrics/MAE"])        # 낮을수록 좋다
        print(metrics["metrics/Smeasure"])   # 적합도, 높을수록 좋음
  export:
    - label: 내보내기
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreBiRefNetl-matte.pt")
        model.export(format="torchscript")
    - label: 내보낸 파일을 실행하십시오
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 팩토리는 파일 접미사에 따라 경로를 지정하므로, 내보낸 아티팩트가 로드됩니다
        # 모든 체크포인트와 마찬가지로 같은 Results 객체를 반환합니다.
        model = LibreYOLO("LibreBiRefNetl-matte.torchscript")
        result = model(SAMPLE_IMAGE)

        print(result.matte.array.shape)
source_hash: f7d88c74d9729268
---

## 정의

`matte` 작업는 단일 RGB 이미지에서 픽셀당 하나의 알파 값을 예측합니다: `1`는 완전히 전경이고 `0`는 완전히 배경입니다. 이 값은 이진 값이 아니라 연속 값이며, 이것이 바로 작업의 핵심입니다. 하드 마스크는 0.5에서 임계값 하나로 결정되지만, 소프트 매트는 이진 마스크가 버리는 머리카락, 털, 모션 블러가 있는 가장자리의 부분 커버리지도 추가로 포함합니다.

예측은 `result.matte`를 채우며, zxQP27Q1XZP 페이로드에는 원본 이미지 캔버스의 `[0, 1]`에서 `(H, W)` float32 배열이 포함되어 있으며, `.array`를 통해 NumPy로 접근할 수 있습니다. `result.cutout()`는 소스 이미지를 해당 알파와 합성하여 `(H, W, 4)` uint8 RGBA 배열을 만들고, `result.save(path)`는 같은 내용을 투명 배경 PNG로 저장합니다. `result.boxes`는 비어 있으므로 `conf`, `iou` 및 `max_det`는 아무런 영향을 미치지 않습니다.

## 모델들

두 계열이 `matte`를 제공하며, 그들은 하나의 순방향 경로를 공유합니다.

[BiRefNet](/docs/models/birefnet)은 이 작업이 중심으로 구축된 양방향 참조 네트워크로, 여기에서는 Swin-L 계층 체크포인트 하나로 공개되었습니다.

[FeyNobg](/docs/models/feynobg)은 Feyn Inc.의 심화 버전입니다: BiRefNet의 아키텍처에서 세 번째 Swin 단계를 18블록에서 24블록으로 확장한 후 재학습한 것입니다. LibreYOLO는 BiRefNet의 순전파 경로, 전처리 및 단일 로짓 출력을 재사용하므로 예측, 검증 및 체크포인트 처리 방식이 동일하게 작동합니다; 가중치와 계열 정체성은 FeyNobg 고유의 것입니다.

두 가지는 서로 다른 중요도의 라이선스를 가지고 있습니다. 두 라이선스 모두 모델 페이지에 명시되어 있으며, 특정 체크포인트의 Hugging Face 저장소에 있는 라이선스가 권위 있는 것입니다.

## 예측

가중치는 처음 사용할 때 Hugging Face에서 다운로드되며 로컬에 캐시됩니다.

<code-tabs name="predict" />

두 계열 모두 고정된 네이티브 1024x1024 캔버스에서 실행되며, 매트는 원본 이미지로 다시 크기 조정됩니다. 다른 해상도는 지원되지 않습니다. Swin 백본의 상대 위치 테이블이 해당 크기에 맞춰져 있기 때문에, 해상도가 맞지 않으면 오류를 발생시키는 대신 잘못 보간됩니다. `Results.save()`는 매트 결과에만 정의되며, 소스 이미지가 필요합니다. 소스 이미지는 전달하지 않으면 `Results.path`에서 다시 로드됩니다. 소스, 스트리밍 및 결과 처리에 대해서는 [예측](/docs/predict)을 참조하십시오.

## 데이터셋 형식

매트 검증은 각 RGB 이미지를 동일한 스템을 가지는 단일 채널 실제 알파 매트와 쌍으로 연결하며, 여기서 0은 배경이고 255는 전경입니다.

```text
my-matte-dataset/
  images/
    subject.jpg
  mattes/
    subject.png
```

`data=`로 해당 루트를 전달하는 것으로 충분합니다: 무광 디렉토리는 `mattes/`, `matte/`, `gt/`, `masks/`, `mask/` 및 `alpha/` 중에서 자동으로 탐지됩니다. 데이터셋 YAML은 대안이며, `path`과 `val_images`, `val_mattes`이 이에 상대적인 디렉토리 이름입니다:

```yaml
path: my-matte-dataset
val_images: images
val_mattes: mattes
nc: 1
names: {0: matte}
```

`nc`와 `names`는 스키마 자리표시자입니다; 매트 모델은 탐지를 반환하지 않고 `Results.matte`를 반환합니다. 매트 값은 `[0, 1]`에서 255로 나누어 알파로 읽습니다, 그리고 예측 캔버스와 형태가 다른 매트는 일치하도록 양선형으로 크기가 조정됩니다. 전체 계약은 [데이터셋 형식](/docs/reference/dataset-formats)을 참조하십시오.

## 학습

두 매트 계열 모두 학습 구현이 없습니다: `train()`는 두 계열 모두에서 `NotImplementedError`를 상승시키며, 매트 지원은 예측, 검증 및 내보내기만 다룹니다. 각 모델 페이지에는 학습 코드를 제공하는 업스트림 프로젝트와 체크포인트를 가져오는 변환 스크립트가 명시되어 있습니다.

## 검증

`val()`는 모델 자체의 `predict`를 구동하므로, 검증 시에는 계열의 정확한 전처리를 사용하고, 두 가지 지표 모두 원본 이미지 캔버스에서 계산됩니다.

<code-tabs name="val" />

`metrics/MAE`는 `[0, 1]`에서 실제 알파 값에 대한 평균 절대 오차이며, 값이 낮을수록 좋습니다. `metrics/Smeasure`는 Fan et al. (ICCV 2017)의 S-측정치로, 구조적 유사도를 나타내며, 피사체의 형태와 구멍을 정확히 맞추는 것을 평가합니다. 단순한 픽셀별 평균만으로는 놓치는 부분입니다. 값이 높을수록 좋습니다. S-측정치는 `fitness`로, 최상의 체크포인트 선택값을 나타냅니다. 두 지표 모두 해상도에 의존하지 않습니다.

## 내보내기

내보낸 매트 모델은 파일 접미사를 통해 `LibreYOLO()`로 다시 로드되므로, 아티팩트는 체크포인트처럼 작동하고 동일한 `Results`를 반환합니다.

<code-tabs name="export" />

TorchScript는 이 작업에 검증된 경로입니다. ONNX 변환은 실행되지만 동일한 동등성 기준을 통과하지 못했으며, 나머지 형식은 사용할 수 없습니다. 형식별 지원 범위는 [BiRefNet](/docs/models/birefnet) 및 [FeyNobg](/docs/models/feynobg) 페이지와 [전체 내보내기 매트릭스](/docs/reference/export-matrix)에서 확인할 수 있습니다.
