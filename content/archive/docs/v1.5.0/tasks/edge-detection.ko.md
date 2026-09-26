---
title: 엣지 검출
seo_title: LibreYOLO에서의 엣지 검출
description: >-
  LibreYOLO에서 한 이미지로부터 조밀한 엣지 확률 지도를 예측합니다. 체크포인트를 변환하고, 지도를 임계값 처리하며, ODS와 OIS로
  검증한 후 내보냅니다.
lead: >-
  엣지 검출은 각 픽셀이 객체 경계에 있을 가능성을 예측합니다. LibreYOLO는 이를 엣지 작업으로 제공하며, 선 세트 대신 원본 이미지
  캔버스에 대한 밀집 확률 지도를 반환합니다.
keywords:
  - 엣지 검출 파이썬
  - 경계 탐지 딥러닝
  - 엣지 확률 맵
  - ODS OIS F-측정
  - 밀집 엣지 예측
last_verified: 1.5.0
snippets:
  predict:
    - label: 엣지 맵을 예측하십시오
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # LibreYOLO에는 엣지 체크포인트가 포함되어 있지 않습니다; 먼저 하나를 변환하십시오(아래 참조).
        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")
        result = model(SAMPLE_IMAGE, save=True)

        edges = result.edges
        print(edges.array.shape)          # (H, W) float32 범위 [0, 1]
        print(edges.binary(0.5).sum())    # 0.5에서의 엣지 픽셀 수
    - label: 자신의 임계값을 선택하십시오
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")
        result = model(SAMPLE_IMAGE)

        # 임계값을 직접 선택할 수 있도록 연속 맵을 보존함.
        for t in (0.3, 0.5, 0.7):
            print(t, int(result.edges.binary(t).sum()))
    - label: 시각화를 저장하십시오
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")
        result = model(SAMPLE_IMAGE)

        # plot()는 지도를 렌더링합니다; 이는 엣지 및 일반 결과에 대해 정의됩니다.
        result.plot().save("edges.png")
  val:
    - label: 메트릭 키를 검증하고 읽습니다
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")
        metrics = model.val(data="my-dataset.yaml", imgsz=352)

        print(metrics["metrics/ODS"])              # 체력
        print(metrics["metrics/OIS"])
        print(metrics["metrics/best_threshold"])
    - label: 스윕과 매치 허용 오차를 변경하십시오
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")
        metrics = model.val(
            data="my-dataset.yaml",
            imgsz=352,
            edge_thresholds=(0.1, 0.2, 0.3, 0.4, 0.5),
            edge_max_dist=0.0075,
        )

        print(metrics["metrics/ODS"], metrics["metrics/best_threshold"])
  export:
    - label: 내보내기
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")
        model.export(format="onnx", imgsz=352)
    - label: 내보낸 파일을 실행하십시오
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 팩토리는 파일 접미사에 따라 경로를 지정하므로, 내보낸 아티팩트가 로드됩니다
        # 모든 체크포인트와 마찬가지로 같은 Results 객체를 반환합니다.
        model = LibreYOLO("weights/LibreDexiNedb-edge.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.edges.array.shape)
source_hash: bc286345540ed966
---

## 정의

`edge` 작업은 단일 RGB 이미지에서 픽셀당 하나의 확률을 예측합니다: `0`은 비-엣지, `1`은 엣지를 의미합니다. 맵은 연속성을 유지하므로, 이를 이진 경계 이미지로 변환하기 위한 임계값 선택은 호출자에게 맡겨지며, 적절한 임계값은 데이터셋과 후속 사용에 따라 달라집니다.

예측은 `result.edges`를 채우며, `EdgeMap` 페이로드는 원본 이미지 캔버스의 `[0, 1]`에 `(H, W)` float32 배열을 보관합니다. `.array`는 그 맵을 NumPy로 반환하고 `.binary(threshold)`는 불리언 마스크를 반환합니다. `result.boxes`는 비어 있으므로 `conf`, `iou` 및 `max_det`는 아무런 영향을 주지 않습니다. `Results.plot()`는 이 작업을 다루며 맵을 직접 렌더링합니다.

## 모델들

세 계열이 `edge`를 섬깁니다.

[DexiNed](/docs/models/dexined), Dense Extreme Inception Network는 여러 측면 출력을 하나의 확률 맵으로 합치며 기본 해상도 352px에서 실행됩니다.

[TEED](/docs/models/teed), 작고 효율적인 엣지 검출기, 는 동일한 네이티브 352 px에서 작은 네트워크이며, DexiNed의 16에 비해 다운샘플 스트라이드가 4이므로 `imgsz`의 더 많은 값을 수용할 수 있습니다.

[LibreMODUS](/docs/models/libremodus)는 any-to-any 모델의 한 목표로 캐니 스타일 엣지를 생성합니다. 이 기능은 `modus` 추가 기능과 본인의 인증된 Hugging Face 계정을 필요로 하며, `val()`나 `export()`는 제공하지 않으므로 아래의 검증 및 내보내기 섹션에는 참여하지 않습니다.

## 예측

LibreYOLO는 엣지 체크포인트를 공개하지 않습니다. 공식적으로 배포된 DexiNed 및 TEED 가중치는 BIPED에서 학습되었으며, 공개된 데이터셋 이용 약관은 비상업적 용도로만 사용을 제한하므로 LibreYOLO는 이를 미러링하지 않습니다. 사용 권한이 있는 체크포인트를 변환한 후, 변환된 파일을 경로로 로드하십시오:

```bash
python weights/convert_dexined_weights.py upstream.pth weights/LibreDexiNedb-edge.pt --verify
```

<code-tabs name="predict" />

로더가 이를 인식하려면 파일 이름에 `-edge` 작업 접미사를 포함해야 합니다. `imgsz`는 네트워크의 다운샘플 스트라이드로 나누어떨어져야 하며, 나누어떨어지지 않으면 LibreYOLO가 나눗셈 값을 명시하는 명확한 오류를 발생시킵니다. 소스, 스트리밍 및 결과 처리에 대해서는 [prediction](/docs/predict)을 참조하십시오.

## 데이터셋 형식

엣지 검증은 각 RGB 이미지를 동일한 해상의 동일한 줄기(single-stem) 단일 채널 맵과, 선택적으로 유효성 마스크와 짝지어 처리합니다.

```text
dataset/
  data.yaml
  images/
    val/scene.jpg
  edges/
    val/scene.png
  masks/
    val/scene.png
```

```yaml
path: dataset
train: images/train
val: images/val
edges_dir: edges
masks_dir: masks
nc: 1
names: {0: edge}
```

대상은 RGB 시각화가 아닌 단일 채널 PNG 또는 TIF입니다. 정수 맵은 dtype의 최댓값으로 나눕니다. 부동 소수점 맵은 이미 유한하고 `[0, 1]`에 있어야 합니다. 마스크 픽셀은 0이 아닐 때 유효한 것으로 간주되며, 패딩된 픽셀은 절대 지표에 기여하지 않습니다. `edge_invert: true`는 흰색 배경에 검은색 가장자리를 저장하는 소스를 다룹니다. 전체 계약은 [데이터셋 형식](/docs/reference/dataset-formats)을 참조하십시오.

## 학습

LibreYOLO에는 학습 구현이 있는 edge 계열이 없습니다: `train()`는 세 가지 모두에서 `NotImplementedError`를 발생시킵니다. 각 모델 페이지는 다른 곳에서 학습된 체크포인트를 LibreYOLO가 로드할 수 있는 것으로 변환하는 스크립트의 이름을 명시합니다.

## 검증

`val()`는 BSDS 스타일의 F-측정을 보고합니다. 연속적인 예측은 먼저 사방향 그래디언트 비최대 억제를 통해 희소화되며, 그 후 예측된 경계 픽셀과 실제 경계 픽셀은 거리 허용 범위 내에서 1:1로 매치됩니다.

<code-tabs name="val" />

`metrics/ODS`는 최적 데이터셋 규모 F-측정값입니다: 매치 수는 각 임계값에서 데이터셋 전체에 걸쳐 합산되며, 이러한 합산된 F-측정값 중 최적 값을 보고합니다. 또한 `fitness`는 최적 체크포인트 선택 횟수입니다. `metrics/OIS`는 최적 이미지 규모 F-측정값으로, 각 이미지의 최적 F-측정값의 평균을 이미지별로 계산하며, 따라서 각 이미지가 자신의 임계값을 선택할 수 있게 합니다. `metrics/best_threshold`는 ODS를 산출한 단일 임계값으로, 이것이 `edges.binary()`에서 추론 시 재사용되는 값입니다.

두 가지 인자가 스윕을 결정합니다. `edge_thresholds`는 시도할 임계값의 집합으로, 기본값은 0.01에서 0.99까지 백분위 단위입니다. `edge_max_dist`는 이미지 대각선의 비율로서 일치 허용치이며, 기본값은 `0.0075`입니다; 그보다 더 떨어진 쌍은 일치하지 않습니다.

## 내보내기

내보낸 엣지 모델은 파일 접미사를 통해 `LibreYOLO()`로 다시 로드되므로, `.onnx` 파일은 체크포인트처럼 작동하며 동일한 `Results`를 반환합니다.

<code-tabs name="export" />

엣지 내보내기는 고정 해상도, 배치 1 런타임 계약을 사용합니다: `dynamic`, 1이 아닌 `batch`는 거부되며, 내보낸 그래프는 단일 융합 확률 맵을 출력합니다. 포맷별 적용 범위는 [DexiNed](/docs/models/dexined) 및 [TEED](/docs/models/teed) 페이지와 [전체 내보내기 매트릭스](/docs/reference/export-matrix)에서 확인할 수 있습니다. [내보내기](/docs/export)는 각 포맷이 허용하는 인수를 나열합니다.
