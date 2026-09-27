---
title: 표면 법선
seo_title: LibreYOLO에서의 표면 법선 추정
description: >-
  LibreYOLO에서 한 이미지로부터 조밀한 표면 법선 필드를 예측합니다. 카메라 프레임 규칙을 읽고, 각도 오차를 검증하며, 모델을
  내보냅니다.
lead: >-
  표면 법선 추정은 보이는 각 표면이 향하는 방향을 예측합니다. LibreYOLO는 이를 normal 작업으로 제공하며, 이는 원본 이미지
  캔버스 상의 단위 벡터로 이루어진 밀집 필드를 반환합니다.
keywords:
  - 표면 법선 추정 파이썬
  - 이미지로 만든 노멀 맵
  - 단안 기하학
  - 각도 오류 측정
  - 밀집한 정규 예측
last_verified: 1.5.0
snippets:
  predict:
    - label: 정상적인 필드를 예측하다
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreMoGe2s-normal.pt")
        result = model(SAMPLE_IMAGE, save=True)

        normals = result.normal_map
        print(normals.data.shape)      # (H, W, 3) float32 단위 벡터
        normals.assert_normalized()    # 어떤 픽셀이 단위 길이가 아니면 예외를 발생시킵니다
    - label: 한 픽셀 읽기
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreMoGe2s-normal.pt")
        result = model(SAMPLE_IMAGE)

        # OpenCV 카메라 프레임: +x는 오른쪽, +y는 아래, +z는 장면 안쪽. 표면
        # 카메라를 바라보면 읽은 값은 대략 (0, 0, -1)입니다.
        field = result.normals.data
        h, w = field.shape[:2]
        print(field[h // 2, w // 2]
    - label: 시각화를 저장하십시오
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreMoGe2s-normal.pt")
        result = model(SAMPLE_IMAGE)

        # plot()는 필드를 렌더링합니다; 이것은 일반 및 엣지 결과에 대해 정의되어 있습니다.
        result.plot().save("normals.png")
  val:
    - label: 메트릭 키를 검증하고 읽습니다
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMoGe2s-normal.pt")
        metrics = model.val(data="my-dataset.yaml", imgsz=518)

        print(metrics["metrics/mean_angular_error"])     # 도
        print(metrics["metrics/median_angular_error"])   # 도
        print(metrics["metrics/within_11_25"])           # 픽셀의 백분율
        print(metrics["metrics/within_22_5"], metrics["metrics/within_30"])
  export:
    - label: 내보내기
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMoGe2s-normal.pt")
        model.export(format="onnx", imgsz=518)
    - label: 내보낸 파일을 실행하십시오
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 팩토리는 파일 접미사에 따라 경로를 지정하므로, 내보낸 아티팩트가 로드됩니다
        # 모든 체크포인트와 마찬가지로 같은 Results 객체를 반환합니다.
        model = LibreYOLO("LibreMoGe2s-normal.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.normal_map.data.shape)
source_hash: d26d26d894b436ff
---

## 정의

`normal` 작업은 단일 RGB 이미지에서 픽셀마다 세 성분 단위 벡터를 예측합니다: 해당 픽셀의 표면이 향하는 방향입니다. 깊이와 달리 출력에는 자유로운 스케일이 없으므로 두 예측을 정렬 없이 직접 비교할 수 있습니다.

예측은 `result.normal_map`를 채우며, 원본 이미지 캔버스에 `(H, W, 3)` float32 배열을 보유한 `NormalMap` 페이로드이며 `result.normals`로 접근할 수도 있습니다. 벡터는 LibreYOLO의 OpenCV 카메라 프레임을 사용하며, 오른쪽은 `+x`, 아래는 `+y`, 장면 안쪽은 `+z`이고 카메라를 향하므로, 정면 평행 표면은 `(0, 0, -1)`를 읽습니다. `.assert_normalized()`는 모든 픽셀이 허용 오차 내에서 유한하고 단위 길이인지 확인합니다. `result.boxes`는 비어 있으므로 `conf`, `iou` 및 `max_det`는 영향을 미치지 않으며, `Results.plot()`가 이 작업을 처리합니다.

## 모델들

두 계열이 `normal`를 제공합니다.

[MoGe-2](/docs/models/moge-2)는 전용 모델입니다: 세 가지 인코더 크기의 단일-포워드 단안 기하 모델입니다. LibreYOLO는 이러한 체크포인트를 자체 조직으로 복사하지 않습니다; 하나를 로드하면 고정된 리비전의 공식 저장소에서 일치하는 크기를 다운로드하고 기록된 SHA-256과 대조하여 검증합니다.

[LibreMODUS](/docs/models/libremodus)은 any-to-any 모델의 한 타겟으로 노멀을 생성하며, 입력으로 RGB 이미지 대신 깊이 맵을 사용할 수 있습니다. 이 모델은 `modus` 추가 요소와 사용자의 인증된 Hugging Face 계정이 필요하며, `val()`나 `export()`는 제공하지 않으므로 아래의 검증 및 내보내기 섹션에는 참여하지 않습니다.

## 예측

MoGe-2 가중치는 처음 사용할 때 다운로드되며 로컬에 캐시됩니다.

<code-tabs name="predict" />

`imgsz`는 ViT 인코더의 패치 크기로 나눠떨어져야 하며, LibreYOLO는 실행 시작 전에 이를 확인합니다. 이미지 목록을 예측할 때는 이미지당 한 번의 순전파가 수행되며, 이 작업에는 쌓인 배치의 빠른 경로가 없습니다. 소스, 스트리밍 및 결과 처리는 [prediction](/docs/predict)을 참조하십시오.

## 데이터셋 형식

일반 검증은 각 이미지를 동일한 해상도의 같은 이름의 3채널 16비트 PNG와 선택적인 유효성 마스크와 함께 연결합니다.

```text
dataset/
  data.yaml
  images/
    val/room.jpg
  normals/
    val/room.png
  masks/
    val/room.png
```

```yaml
path: dataset
train: images/train
val: images/val
normals_dir: normals
masks_dir: masks
nc: 1
names: {0: normal}
```

대상 PNG는 채널이 RGB로 저장된 정확히 3채널 `uint16`입니다. 디코딩은 `n = png / 65535 * 2 - 1`를 수행한 후 각 벡터를 재정규화하는 방식이며, 디코딩된 벡터는 예측과 동일한 OpenCV 카메라 프레임을 사용합니다. 마스크 픽셀은 0이 아닐 때 유효한 것으로 간주됩니다. 마스크 파일이 없으면 유한하며 0이 아닌 모든 디코딩된 벡터가 유효합니다. 잘못되었거나 패딩된 대상 픽셀은 내부적으로 `(0, 0, 0)`로 보관되며 메트릭 계산에 절대 기여하지 않습니다. 전체 계약 내용은 [데이터셋 형식](/docs/reference/dataset-formats)을 참조하십시오.

## 학습

어느 정상 계열도 학습 실행을 가지고 있지 않습니다: `train()`는 둘 다에서 `NotImplementedError`를 발생시킵니다. MoGe-2의 페이지는 예측, 검증 및 내보내기를 위한 고정된 공식 체크포인트를 가리킵니다.

## 검증

`val()`는 데이터셋이 유효하다고 표시한 픽셀에 대해 각 예측 벡터와 그 실제 벡터 사이의 각도를 측정합니다.

<code-tabs name="val" />

`metrics/mean_angular_error`와 `metrics/median_angular_error`는 각도를 도 단위로 나타낸 것이며, 낮을수록 좋습니다. `metrics/within_11_25`, `metrics/within_22_5`, `metrics/within_30`는 각 오차가 11.25도, 22.5도, 30도 이내인 유효 픽셀의 백분율로, 높을수록 좋습니다. 단위를 주의하십시오: 이 세 가지는 백분율이며, 분수가 아닙니다. `fitness`는 `metrics/within_11_25`를 100으로 나눈 것이며, 이는 최적 체크포인트 선택을 다른 모든 작업과 동일한 `[0, 1]` 척도로 맞추는 것입니다.

## 내보내기

내보낸 일반 모델은 파일 접미사를 통해 `LibreYOLO()`로 다시 로드되므로, `.onnx` 파일은 체크포인트처럼 동작하며 동일한 `Results`를 반환합니다.

<code-tabs name="export" />

일반 내보내기는 고정 해상도, 배치 1 런타임 계약을 사용합니다: `dynamic` 및 1이 아닌 `batch`는 거부되며, `imgsz`는 인코더의 패치 크기로 나누어 떨어져야 합니다. 형식별 적용 범위는 [MoGe-2 페이지](/docs/models/moge-2)와 [전체 내보내기 매트릭스](/docs/reference/export-matrix)에 나와 있습니다. [내보내기](/docs/export)는 각 형식이 허용하는 인수를 나열합니다.
