---
title: 점 탐지
seo_title: LibreYOLO에서 포인트 탐지 및 개수 세기
description: >-
  LibreYOLO에서 객체를 박스 대신 단일 점으로 찾습니다. 중심점을 예측하고, 객체 수를 세고, FOMO를 학습하며, 점 메트릭을
  읽습니다.
lead: >-
  포인트 탐지는 바운딩 박스 대신 객체당 하나의 x, y 위치를 반환합니다. LibreYOLO는 이를 포인트 작업으로 제공합니다. 예측은
  객체마다 하나의 x, y, 클래스 및 신뢰도 행을 포함합니다.
keywords:
  - 포인트 탐지 파이썬
  - 파이썬 객체 세기
  - 중심점 탐지
  - FOMO 포인트 현지화
  - 이미지 속 물체 세기
  - 점 위치 지정
last_verified: 1.5.0
snippets:
  predict:
    - label: 점을 예측하고 계산하십시오
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # LibreFOMO 가중치는 자동으로 다운로드되지 않습니다. 체크포인트를 가져오십시오
        # https://huggingface.co/LibreYOLO를 먼저 로컬 경로에서 로드하십시오.
        model = LibreYOLO("./LibreFOMOs-point.pt")
        result = model(SAMPLE_IMAGE, save=True)

        points = result.points
        print(len(points))     # 객체 수
        print(points.xy)       # (N, 2) 원본 이미지 픽셀의 중심
        print(points.cls, points.conf)
    - label: 정규화된 좌표 및 클래스별 개수
      language: python
      code: |
        from collections import Counter

        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("./LibreFOMOs-point.pt")
        result = model(SAMPLE_IMAGE)

        points = result.points.numpy()
        print(points.xyn)                          # [0, 1]에서 동일한 중심
        print(Counter(points.cls.astype(int).tolist()))
  train:
    - label: YOLO 데이터셋에서 FOMO를 학습시키다
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("./LibreFOMOs-point.pt")
        model.train(data="my-dataset.yaml", epochs=40, batch=32, lr0=3e-4)
    - label: 학습된 체크포인트로 예측하기
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("./LibreFOMOs-point.pt")
        results = model.train(data="my-dataset.yaml", epochs=40)

        # train()는 최고의 체크포인트를 동일한 객체에 다시 로드하므로,
        # 모델은 호출이 반환될 때 학습된 가중치로 예측합니다.
        print(results["best_checkpoint"])
        print(model(SAMPLE_IMAGE).points.xy)
  val:
    - label: 메트릭 키를 검증하고 읽습니다
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("./LibreFOMOs-point.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/precision"], metrics["metrics/recall"])
        print(metrics["metrics/f1"])
        print(metrics["metrics/mAP@[0.01:0.10]"])   # 체력
        print(metrics["metrics/MLE"])               # 평균 위치 오차
        print(metrics["metrics/MAE"], metrics["metrics/RMSE"])   # 카운트 오류
    - label: 거리 임계값을 변경하십시오
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("./LibreFOMOs-point.pt")


        # 스윕 범위는 주요 텍스트의 일부이므로, 사용자 지정 스윕

        # 생성하는 mAP 키의 이름을 변경합니다.

        metrics = model.val(data="my-dataset.yaml", dist_thresholds=[0.02,
        0.05])


        print(metrics["metrics/mAP@0.02"])

        print(metrics["metrics/mAP@[0.02:0.05]"])
  export:
    - label: 내보내기
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("./LibreFOMOs-point.pt")
        model.export(format="onnx")
    - label: 내보낸 파일을 실행하십시오
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 팩토리는 파일 접미사에 따라 경로를 지정하므로, 내보낸 아티팩트가 로드됩니다
        # 모든 체크포인트와 마찬가지로 같은 Results 객체를 반환합니다.
        model = LibreYOLO("./LibreFOMOs-point.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.points.xy)
source_hash: 932153c8870d1c7c
---

## 정의

`point` 작업은 객체마다 너비, 높이 또는 마스크 없이 단일 x, y 좌표와 클래스로 각각의 객체를 위치시킵니다. 예측이 객체의 평면 목록이기 때문에 행 수는 객체 수이며, 이것이 이 작업을 집계 작업으로 만드는 이유입니다.

예측은 `result.points`를 채우며, 이는 `(N, 4)` 배열의 `x, y, class, confidence` 행을 원본 이미지 픽셀로 감싸는 `Points` 페이로드입니다. `.xy`는 좌표를 반환하고, `.xyn`는 이미지 크기로 나눈 동일한 좌표를 반환하며, `.cls`는 클래스 인덱스를, `.conf`는 점수를 반환합니다; `len()`는 포인트 수를 반환합니다. `result.boxes`는 비어 있으므로, `iou`와 `max_det`는 작동할 것이 없습니다.

## 모델들

세 계열이 `point`를 사용하며, 서로 교환할 수 없습니다.

[FOMO](/docs/models/fomo)은 고정 어휘 옵션입니다: 저해상도 그리드의 각 셀을 배경 또는 객체 중심으로 표시하는 그리드 분류기입니다. 이것은 LibreYOLO가 학습할 수 있는 유일한 포인트 계열이며, 내보낼 수 있는 유일한 계열이기도 합니다.

[LocateAnything](/docs/models/locate-anything)은 클래스 인덱스 대신 텍스트를 사용하므로, 어휘는 작성하는 문구에 따라 달라집니다. `vlm`가 추가로 필요하며, `LibreYOLO()` 팩토리를 통해서가 아니라 `LibreLocateAnything`로 구성되며, 가중치는 비상업적 사용으로 제한됩니다. 정확한 조건과 체크포인트가 구성하는 두 가지 추가 라이선스는 해당 페이지에 나와 있습니다.

[SenseNova-Vision](/docs/models/sensenova-vision)은 여섯 가지 다른 작업에 사용하는 동일한 프롬프트 생성 체크포인트를 통해 `point`에 도달하며, `LibreVLM("sensenova-vision", task="point")`로 로드됩니다. `sensenova` 추가가 필요하며, 모든 예측은 7B 모델에 대한 생성 패스이므로, 목적에 맞게 제작된 탐지기보다 이미지당 지연 시간이 눈에 띄게 높을 것으로 예상됩니다. 그 가중치는 비상업적이며, 라이선스는 해당 페이지에 있습니다.

## 예측

LibreFOMO 가중치는 이 사이트에서 자동 다운로드의 유일한 예외입니다. `LibreYOLO("LibreFOMOs-point.pt")`는 디스크에서 해당 파일을 찾고 가져오는 대신 `ValueError`를 이름으로 지정합니다. 먼저 Hugging Face의 [LibreYOLO 조직](https://huggingface.co/LibreYOLO)에서 체크포인트를 다운로드한 후 로컬 경로로 로드하거나 직접 학습시키십시오.

<code-tabs name="predict" />

로더가 파일을 인식하려면 파일 이름에 `-point` 작업 접미사가 포함되어야 합니다. `predict(..., nms_radius=1)`는 두 개의 FOMO 탐지가 모두 살아남기 위해 얼마나 많은 그리드 셀 간격이어야 하는지를 제어합니다. 소스, 스트리밍 및 결과 처리는 [prediction](/docs/predict)을 참조하십시오.

## 데이터셋 형식

`point`는 자체적인 레이블 형식이 없습니다. 포인트 계열들은 표준 YOLO 탐지 레이아웃을 읽고 각 박스 행에서 하나의 중심을 도출하며, 따라서 `cx cy`가 포인트이고 `w h`는 단지 그 행이 유효한지 여부만 결정합니다.

```text
dataset/
  data.yaml
  images/
    train/scene.jpg
    val/scene.jpg
  labels/
    train/scene.txt
    val/scene.txt
```

각 레이블 파일은 객체당 한 줄을 가지며, 정규화된 좌표를 포함합니다:

```text
<class_id> <cx> <cy> <w> <h>
```

```yaml
path: dataset
train: images/train
val: images/val
nc: 1
names: {0: seedling}
```

레이블 파일이 없거나 비어 있다는 것은 객체가 없음을 의미합니다. 전체 조건은 [데이터셋 형식](/docs/reference/dataset-formats)을 참조하십시오.

## 학습

FOMO는 학습 구현이 있는 유일한 포인트 계열입니다. LocateAnything과 SenseNova-Vision에서 `train()`는 `NotImplementedError`를 상승시키며; 해당 업스트림 모델을 파인튜닝하고 결과를 로드하십시오.

<code-tabs name="train" />

`imgsz`는 FOMO에 대해 자유롭게 선택할 수 있는 것이 아닙니다: 이는 로드된 체크포인트의 기본 해상도로 설정되며, 다른 값을 전달하면 예상하는 크기를 명명하는 `ValueError`가 발생합니다. 데이터셋, 로거 및 다중 GPU에 대해서는 [training](/docs/train)를 참조하고, 이 계열의 기본값에 대해서는 [FOMO 페이지](/docs/models/fomo)를 참조하십시오.

## 검증

`val()`는 헝가리안 알고리즘을 사용하여 예측된 점을 기준 진리 점과 일대일로 매칭하며, 거리 임계값 범위에 걸쳐 수행됩니다. 임계값은 정규화된 이미지 좌표에서의 유클리드 거리이며, 기본 범위는 0.01에서 0.10까지의 10개 값입니다.

<code-tabs name="val" />

`metrics/precision`, `metrics/recall` 및 `metrics/f1`는 스윕에서 가장 엄격한 임계값인 기본값 0.01에서 클래스별 매크로 평균입니다. `metrics/mAP@0.01`는 동일한 임계값에서의 평균 정밀도이고, `metrics/mAP@[0.01:0.10]`는 스윕 전체에 대한 평균입니다. 해당 스윕 값은 `fitness`이기도 하며, 이는 최적 체크포인트 선택이 읽는 숫자입니다. 두 mAP 키는 사용 중인 임계값에서 생성되므로, `dist_thresholds=`를 전달하면 이름이 바뀝니다.

`metrics/MLE`는 동일한 정규화 단위에서 가장 엄격한 임계값에서 매칭된 쌍 간의 평균 거리입니다. `metrics/MAE`와 `metrics/RMSE`는 위치 지정 지표가 아니라 카운팅 지표로, 이미지별로 예측된 점과 실제 점의 수 차이를 측정합니다.

FOMO는 이러한 것들 위에 두 번째 그리드 수준 그룹을 추가합니다. 그것은 신뢰도와 `nms_radius`를 정리하고 최고의 F1 조합을 `metrics/grid_F1`, `metrics/grid_precision`, `metrics/grid_recall`, `metrics/grid_mean_distance`, `metrics/grid_TP`, `metrics/grid_FP` 및 `metrics/grid_FN`로 게시하며, 이를 생성한 설정은 `decode/threshold`와 `decode/nms_radius`에 있습니다.

## 내보내기

FOMO는 공유된 내보내기 경로를 통해 내보내며, 내보낸 아티팩트는 파일 확장자를 통해 `LibreYOLO()`로 다시 로드되므로 `.onnx` 또는 `.engine` 파일은 체크포인트처럼 동작하고 동일한 `Results`를 반환합니다.

<code-tabs name="export" />

포맷별 커버리지는 [FOMO 페이지](/docs/models/fomo)와 [전체 내보내기 매트릭스](/docs/reference/export-matrix)에 있습니다. LocateAnything과 SenseNova-Vision은 내보내지 않습니다: `export()`는 둘 다에서 경고를 발생시키는데, 이는 생성 모델에는 추적 가능한 탐지 그래프가 없기 때문입니다.
