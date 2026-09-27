---
title: 깊이 추정
seo_title: LibreYOLO에서 단안 깊이 추정
description: 'LibreYOLO에서 한 이미지로부터 조밀한 상대 깊이 맵을 예측합니다. 깊이 계열을 비교하고, 깊이 지표를 읽고, 깊이 모델을 내보냅니다.'
lead: >-
  깊이 추정은 단일 이미지를 사용하여 각 픽셀이 카메라로부터 얼마나 떨어져 있는지를 예측합니다. LibreYOLO는 이를 깊이 작업으로
  제공하며, 원본 이미지 캔버스에서 촘촘한 상대 역-깊이 지도를 반환합니다.
keywords:
  - 단안(depth) 깊이 추정 파이썬
  - 단일 이미지로부터 깊이 맵
  - 상대 깊이 모델
  - 깊이 어떤 것 libreyolo
  - 밀집 깊이 추정
last_verified: 1.5.0
snippets:
  predict:
    - label: 깊이 맵 예측
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")
        result = model(SAMPLE_IMAGE, save=True)

        depth = result.depth_map
        print(depth.data.shape)              # (원본 캔버스에서 H, W)
        print(depth.min, depth.max, depth.mean)
    - label: 값을 사용하십시오
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")
        result = model(SAMPLE_IMAGE)

        depth = result.depth_map
        raw = depth.data          # 높을수록 가까움; 단위 없음, 척도 없음
        gray = depth.normalized() # 시각화를 위해 [0, 1] 범위로 재조정됨
        print(raw.shape, float(gray.max()))
    - label: 컴팩트한 대안
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 같은 작업 계약, 엣지 런타임을 위해 구축된 훨씬 작은 네트워크.
        model = LibreYOLO("LibreZipDepthb-depth.pt")
        result = model(SAMPLE_IMAGE)

        print(result.depth_map.data.shape)
  val:
    - label: 메트릭 키를 검증하고 읽습니다
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/abs_rel"])
        print(metrics["metrics/rmse"])
        print(metrics["metrics/delta1"])   # 체력
        print(metrics["metrics/delta2"], metrics["metrics/delta3"])
  export:
    - label: 내보내기
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")
        model.export(format="onnx")
    - label: 내보낸 파일을 실행하십시오
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 팩토리는 파일 접미사에 따라 경로를 지정하므로, 내보낸 아티팩트가 로드됩니다
        # 모든 체크포인트와 마찬가지로 같은 Results 객체를 반환합니다.
        model = LibreYOLO("LibreDepthAnythingV2s-depth.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.depth_map.data.shape)
source_hash: e0612c59f9c999b4
---

## 정의

`depth` 작업은 단일 RGB 이미지에서 픽셀마다 하나의 값을 예측합니다. LibreYOLO는 그 값을 상대적 역(depth)으로 정의하며, 값이 클수록 카메라에 더 가깝다는 의미이고, 숫자는 어떤 측정 단위도 없고 두 이미지 간에 유지되는 스케일도 없습니다. 동일한 예측의 두 픽셀 간의 깊이를 비교하는 것은 의미가 있지만, 다른 이미지의 값과 값을 비교하는 것은 의미가 없습니다.

예측은 `result.depth_map`를 채우며, 이는 원본 이미지 캔버스에 `(H, W)` 배열을 보유한 `DepthMap` 페이로드입니다. `.min`, `.max` 및 `.mean`는 유한 값을 읽고, `.normalized()`는 표시를 위해 맵을 `[0, 1]`로 재조정합니다. `result.boxes`는 비어 있으므로 `conf`, `iou` 및 `max_det`는 아무런 효과가 없고, `save=True`는 주석이 달린 사진 대신 맵의 컬러맵 이미지를 작성합니다.

## 모델들

여섯 가문이 `depth`를 섬깁니다.

[Depth Anything V2](/docs/models/depth-anything-v2)은 DINOv2 인코더와 DPT 디코더를 결합한 것으로, 여기서 일반적으로 사용하는 기본 모델입니다. 라이선스는 정확도만큼 모델 크기에도 영향을 미칩니다. Small 체크포인트는 Apache-2.0 라이선스를 가지며, Base와 Large는 비상업용이므로 선택하기 전에 해당 페이지의 체크포인트 표를 확인하십시오.

[Depth Anything 3](/docs/models/depth-anything-3)은 DA3MONO-LARGE 체크포인트를 포팅하며, 깊이에 대한 구조적 특성이 없는 일반 트랜스포머입니다.

[ZipDepth](/docs/models/zipdepth)는 소형 계층입니다: Depth Anything V2 Large에서 증류된 재매개변수화 가능한 CNN으로, 두 번째 체크포인트에서는 디코더가 gather 및 unfold 연산을 피하여 해당 연산이 없는 NPU 컴파일러를 지원합니다.

[MiDaS](/docs/models/midas)은 다른 계열들이 측정되는 제로-샷 상대 깊이 프로토콜을 확립한 작업 계열입니다. LibreYOLO가 재배포하지 않는 유일한 깊이 계열입니다: 체크포인트를 요청하면 공식 제작자의 GitHub 릴리스에서 자산을 다운로드하고 고정된 SHA-256을 확인합니다.

[LibreMODUS](/docs/models/libremodus)은 전용 헤드가 아닌 any-to-any 모델의 한 타깃으로서 깊이에 도달합니다. 이는 `modus` 추가와 본인의 인증된 Hugging Face 계정을 필요로 하며, `val()`나 `export()`은 제공하지 않습니다.

[SenseNova-Vision](/docs/models/sensenova-vision)은 여섯 가지 다른 작업을 수행하는 동일한 7B 체크포인트에서 디퓨전 디코드를 통해 깊이 맵을 이미지로 생성합니다. 추가적으로 `sensenova`가 필요하며, 가중치는 비상업적 사용으로 제한됩니다. 라이선스는 해당 페이지에서 확인할 수 있습니다.

## 예측

가중치는 위에서 언급한 두 가지 계열을 제외하고, 처음 사용할 때 Hugging Face에서 다운로드되어 로컬에 캐시됩니다.

<code-tabs name="predict" />

입력 해상도는 계열별로 제한됩니다. Depth Anything V2와 Depth Anything 3는 DINOv2 패치 그리드를 기반으로 구축되므로, `imgsz`는 14로 정확히 나누어져야 하며, LibreYOLO는 실행 전에 이를 확인합니다. `Results.plot()`는 이 작업을 다루지 않으며, 표면 법선과 엣지 전용으로 정의되어 있습니다. 소스, 스트리밍 및 결과 처리에 대해서는 [예측](/docs/predict)을 참조하십시오.

## 데이터셋 형식

깊이 검증은 각 이미지를 동일한 해상도를 가진 밀집 단일 채널 깊이 맵과 짝지으며, 이미지 경로에 깊이 디렉토리를 대입하여 찾습니다.

```text
dataset/
  data.yaml
  images/
    val/room.jpg
  depths/
    val/room.png
```

```yaml
path: dataset
val: images/val
depths_dir: depths
nc: 1
names: {0: depth}
```

맵은 단일 채널 PNG 또는 TIF, 혹은 `.npy`입니다. 값은 데이터셋이 일관되게 유지하는 단위의 일반 깊이이며, `0`에서 음수, NaN 및 무한 픽셀은 지표에서 제외되는 유효하지 않은 샘플을 표시합니다. 정수 맵은 `depth_scale`로 나누며, 기본값은 16비트 PNG 규약인 `256.0`입니다; 부동 소수점 `.npy` 맵은 그대로 사용됩니다. `depth_stem_suffix`와 `depth_mask_suffix`는 깊이 파일이나 유효성 마스크의 이름이 다르게 지정된 데이터셋을 다룹니다. 전체 규약은 [데이터셋 형식](/docs/reference/dataset-formats)을 참조하십시오.

## 학습

LibreYOLO에는 학습 구현이 없는 깊이 없는 계열가 있습니다: `train()`는 여섯 개 모두에 대해 `NotImplementedError`를 발생시킵니다. 각 모델 페이지에는 업스트림에서 학습된 체크포인트를 LibreYOLO에서 로드할 수 있는 것으로 변환하는 스크립트의 이름이 나와 있습니다.

## 검증

`val()`는 공유된 깊이 검증기를 실행합니다. 상대 깊이에는 절대적인 척도가 없으므로, 각 예측값은 먼저 이미지별 최소제곱 스케일과 이동으로 실제 값의 역수에 맞춰 조정된 후, 다시 깊이로 반전됩니다. 아래의 모든 지표는 해당 정렬된 맵에서 이미지별로 계산되고, 데이터셋이 유효하다고 표시한 픽셀만 계산하여 데이터셋 전체에서 평균됩니다.

<code-tabs name="val" />

`metrics/abs_rel`는 평균 절대 상대 오차로, 잔차를 실제 깊이로 나눈 값이며, 낮을수록 좋습니다. `metrics/rmse`는 데이터셋 자체 깊이 단위의 제곱근 평균 제곱 오차로, 역시 낮을수록 좋습니다. `metrics/delta1`, `metrics/delta2`, `metrics/delta3`는 임계 정확도로, 유효 픽셀 중 실제 깊이와의 비율(어느 쪽이든 큰 쪽을 취함)이 각각 1.25, 1.25 제곱, 1.25 세제곱 아래에 속하는 비율을 나타내며, 높을수록 좋습니다. `metrics/delta1`는 `fitness`와 같으며, 가장 좋은 체크포인트 선택을 나타냅니다.

## 내보내기

내보낸 깊이 모델은 파일 접미사를 통해 `LibreYOLO()`로 다시 로드되므로, `.onnx` 또는 `.engine` 파일은 체크포인트처럼 작동하며 `Results`와 동일하게 반환하고, 박스 대신 `depth_map`가 들어갑니다.

<code-tabs name="export" />

보장은 계열마다 다르며, Depth Anything 3는 검증된 세트 외의 형식을 시도하지 않고 거부합니다. 대상 형식으로 변환하기 전에 모델 페이지와 [전체 내보내기 매트릭스](/docs/reference/export-matrix)를 확인하십시오. LibreMODUS와 SenseNova-Vision은 전혀 내보내지 않습니다. [내보내기](/docs/export)에서는 각 형식이 허용하는 인수를 나열합니다.
