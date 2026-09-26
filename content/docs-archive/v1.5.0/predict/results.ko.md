---
title: 결과 작업
seo_title: LibreYOLO 결과 객체
description: >-
  이미지당 하나의 Results 객체로, 각 페이로드 유형별로 슬롯이 있음: boxes, masks, keypoints, probs,
  depth, panoptic, OCR 등. 플로팅, 저장 및 JSON.
lead: >-
  모든 예측은 이미지당 하나의 Results 객체를 반환합니다. 각 페이로드 종류마다 하나의 이름이 지정된 슬롯이 있으며, 모델이 생성하는 것
  외에는 모두 비어 있고, 내보낸 아티팩트에도 동일한 슬롯이 있습니다.
keywords:
  - 파이썬에서 YOLO 결과 객체
  - results.boxes xyxy
  - 결과를 JSON으로
  - 주석이 달린 이미지 저장
  - 세분화 마스크 파이썬
  - 핵심점 결과
  - 깊이 지도 결과
  - 결과 요약
  - onnx 동일한 결과
last_verified: 1.5.0
verification: >-
  페이로드 클래스, 슬롯, 무브 시맨틱, summary(), to_json(), plot(), save() 및 cutout()는
  libreyolo/utils/results.py.의 Annotation에서 읽고, 디스크 쓰기 동작은
  libreyolo/models/base/inference.py의 InferenceRunner._save_annotated_image와
  libreyolo/utils/general.py.의 resolve_save_path에서 수행됩니다. 접미사 디스패치는
  libreyolo/models/__init__.py.의 LibreYOLO()에서 수행됩니다.
snippets:
  basic:
    - label: 상자들
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")
        result = model(SAMPLE_IMAGE)

        print(result.orig_shape)   # 원본 이미지의 (높이, 너비)
        print(result.path)         # 소스 경로, 메모리 내 입력의 경우 None

        for xyxy, conf, cls in zip(
            result.boxes.xyxy.tolist(),
            result.boxes.conf.tolist(),
            result.boxes.cls.tolist(),
        ):
            print(result.names[int(cls)], round(float(conf), 3), xyxy)
    - label: 정규화된 좌표
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy[:1])    # 픽셀, x1 y1 x2 y2
        print(result.boxes.xywh[:1])    # 픽셀, 중심 x, 중심 y, 너비, 높이
        print(result.boxes.xyxyn[:1])   # 너비와 높이로 나뉜 같은 상자
        print(result.boxes.xywhn[:1])
    - label: NumPy와 장치
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")
        result = model(SAMPLE_IMAGE)

        # 이 각각은 새로운 Results를 반환합니다; 원본은 변경되지 않습니다.
        as_numpy = result.numpy()
        on_cpu = result.cpu()

        print(type(as_numpy.boxes.xyxy).__name__)
        print(type(on_cpu.boxes.xyxy).__name__)
  json:
    - label: 요약 및 to_json
      language: python
      code: |
        import json

        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")
        result = model(SAMPLE_IMAGE)

        rows = result.summary()
        print(json.dumps(rows[:2], indent=2))

        # 같은 키워드 인수를 사용하여 문자열로 같은 내용.
        print(result.to_json(normalize=True, decimals=3)[:200])
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt --json \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
  saving:
    - label: 주석이 달린 이미지
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")

        # save=True는 페이로드를 그리고 runs/detect/predict*. 아래에 저장합니다
        result = model(SAMPLE_IMAGE, save=True)
        print(result.saved_path)
  exported:
    - label: 내보내기 추가 기능을 설치하십시오
      language: bash
      code: |
        pip install "libreyolo[onnx]"
    - label: 내보낸 아티팩트에서 동일한 결과
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")
        path = model.export(format="onnx")   # 작성된 경로를 반환합니다

        # LibreYOLO()는 파일 접미사에 따라 분기합니다.
        exported = LibreYOLO(path)
        result = exported(SAMPLE_IMAGE)

        print(type(result).__name__, len(result.boxes))
source_hash: 548dbc9c7f5552ec
---

## 하나의 객체, 페이로드당 하나의 슬롯

한 이미지에 대한 예측은 하나의 `Results`를 반환합니다. 이는 18개의 페이로드 슬롯을 가지고 있으며, 모델은 자신의 작업이 생성하는 슬롯만 채웁니다. 나머지 모든 슬롯은 `None`이므로, 탐지기에서 `result.masks`를 읽는 것은 오류가 아니라 `None`입니다.

| 슬롯 | 수업 | 모양 | 제작 |
|---|---|---|---|
| `boxes` | `Boxes` | `(N, 4)` 플러스 점수와 수업 | 탐지 및 먼저 위치를 지정하는 모든 작업 |
| `masks` | `Masks` | `(N, H, W)` | 인스턴스 분할 |
| `keypoints` | `Keypoints` | `(N, K, 2)` 또는 `(N, K, 3)` | 자세 |
| `probs` | `Probs` | `(C,)` | 분류 |
| `obb` | `OBB` | `(N, 7)` 또는 `(N, 8)` | 방향이 지정된 상자 |
| `gaze` | `Gaze` | `(N, 2)` 피치와 요 요각(라디안 단위) | 시선 추정 |
| `points` | `Points` | `(N, 4)`를 x, y, 클래스, 신뢰도로 | 점 위치 지정 |
| `semantic_mask` | `SemanticMask` | `(H, W)` 클래스 아이디 | 시맨틱 분할 |
| `panoptic` | `PanopticSegmentation` | `(H, W)` 세그먼트 ID와 `segments_info` | 판옵틱 분할 |
| `depth_map` | `DepthMap` | `(H, W)` 뜨다 | 깊이 추정 |
| `normal_map` | `NormalMap` | `(H, W, 3)` 단위 벡터 | 표면 법선 |
| `edges` | `EdgeMap` | `(H, W)`는 `[0, 1]`에서 떠다닌다 | 엣지 검출 |
| `restored` | `RestoredImage` | `(H, W, 3)` uint8 RGB | 복원 및 초해상도 |
| `matte` | `Matte` | `(H, W)`는 `[0, 1]`에서 떠다닌다 | 알파 매팅 및 배경 제거 |
| `ocr` | `OCRRegions` | `(N, 4, 2)` 다각형 및 전사본 | 텍스트 탐지 및 인식 |
| `embeddings` | `Embeddings` | `(N, D)` L2-정규화된 행 | `embed` 작업 |
| `identities` | `Identities` | N개의 이름과 점수 | 갤러리가 있는 `embed` 작업 |
| `meshes` | `Meshes` | 바디 매개변수 및 선택적 정점 | 바디 메시 복구 |

그들 옆에는 모든 결과가 갖는 필드가 있습니다: `orig_shape`는 `(height, width)`로, `path`(원본 경로 또는 메모리 입력의 경우 `None`), `names`는 클래스 ID를 클래스 이름에 매핑, `frame_idx`는 비디오 및 실시간 프레임용, `track_id`는 추적 시, 그리고 `restore_scale`는 복원 결과의 정수 업스케일 계수입니다.

`result.normals`는 `result.normal_map`의 별칭입니다.

`result.speed`는 모든 결과에 존재하지만 [앙상블](/docs/predict/ensembling)에서만 채워지며, 그 키는 밀리초 단위의 `member_0`, `member_1` 및 `fusion`입니다. 단일 모델의 경우 빈 딕셔너리로 유지됩니다.

## 상자들

<code-tabs name="basic" />

`Boxes`는 하나의 통합된 텐서 대신 좌표와 점수를 별도의 배열로 저장합니다.

| 속성 | 내용 |
|---|---|
| `xyxy` | `(N, 4)` 절대 픽셀, x1 y1 x2 y2 |
| `xywh` | `(N, 4)` 절대 픽셀, 중심 x, 중심 y, 너비, 높이 |
| `xyxyn`, `xywhn` | 같은 값을 이미지 너비와 높이로 나눈 값 |
| `conf` | `(N,)` 자신감 |
| `cls` | `(N,)` 클래스 ID, 부동 소수점으로 |
| `id` | `(N,)` 트랙 ID, 또는 `None` |
| `is_track` | `id`가 설정되어 있는지 여부 |
| `data` | 모든 것이 연결됨: 상자, 선택적 ID, conf, cls |

`cls`는 float 배열이므로 `result.names[int(cls)]`로 사용하십시오.

`xyxyn`와 `xywhn`에는 `orig_shape`가 필요하며, `Results`가 이 값을 채웁니다.

## 밀집된 페이로드

전체 이미지를 포함하는 페이로드는 개별 인스턴스 페이로드와 다르게 동작하며, 슬라이싱할 때 중요합니다.

`SemanticMask`는 원본 캔버스에서 `(H, W)` 클래스 ID를 보유하며, `255`는 클래스에 포함되지 않는 무시 값으로 예약됩니다. `classes`는 존재하는 ID를 나열하고 이를 제외하며; `class_mask(id)`는 불리언 `(H, W)`를 반환합니다.

`PanopticSegmentation`는 `(H, W)` 세그먼트 ID를 가지고 있으며, `0`를 공란 ID로 사용하고, 최소한 `id`와 `category_id`를 포함하는 `segments_info` 딕셔너리 목록을 가지고 있습니다. `segment_ids`는 존재하는 ID를 나열하고, `segment_mask(id)`는 하나를 선택합니다.

`DepthMap`는 `(H, W)` 상대 역깊이를 나타냅니다: 값이 클수록 가까움을 의미하며, 값은 미터 단위가 아닙니다. 이것은 `min`, `max`, `mean`를 유한한 값으로 노출하며, `normalized()`를 `[0, 1]`으로 리스케일링합니다.

`NormalMap`는 OpenCV 카메라 좌표계에서 `(H, W, 3)` 단위 벡터를 가지고 있으며, `+x`는 오른쪽, `+y`는 아래쪽, `+z`는 장면 안쪽을 향하므로 카메라를 향한 표면은 `(0, 0, -1)`입니다. `assert_normalized()`는 모든 픽셀이 유한하고 단위 길이인지 확인합니다.

`EdgeMap`는 `[0, 1]`에 `(H, W)` float32를 포함합니다. 연속 지도가 임계값 처리되지 않고 유지되므로, `binary(threshold=0.5)`는 컷오프를 선택하는 위치입니다.

`Matte`는 `[0, 1]`에 `(H, W)` float32를 보유하며, `1`는 완전히 전경입니다. `array`는 그것을 float32로 잘라 반환합니다.

`RestoredImage`는 `(H, W, 3)` uint8 RGB를 보유하며, 원시 ndarray를 위한 `array`와 그것을 쓰기 위한 `save(path)`를 갖습니다.

`Probs`는 이미지에 대한 하나의 확률 벡터를 가지고 있습니다. `top1`와 `top5`는 클래스 인덱스이며, `top1conf`와 `top5conf`는 매칭 점수입니다.

`Embeddings`는 이미 L2 정규화된 `(N, D)` 행을 가지고 있으므로 코사인 유사도는 점 곱입니다. `similarity(other)`는 갤러리에 대해 `(N, M)`를 반환하거나 단일 벡터에 대해 `(N,)`를 반환하며, `verify(i, j, threshold=0.4)`는 두 행을 비교합니다.

`OCRRegions`는 읽기 순서대로 `(N, 4, 2)` 다각형을 포함하며, 꼭짓점은 좌상, 우상, 우하, 좌하 순으로 정렬됩니다. 전사는 `texts`에 있고, 인식 점수는 `conf`에 있으며, 검출 점수는 `det_conf`에 있습니다. 이것들은 진짜 회전된 다각형이기 때문에 `boxes`에는 포함되지 않습니다; `ocr.xyxy`는 사각형이 필요할 때 축에 맞춘 외각을 제공합니다.

## 자르기와 이동

`result[i]`는 하나의 인스턴스를 보유한 새 `Results`를 반환합니다. 인스턴스별 페이로드는 분할되며, 전체 이미지 페이로드는 그대로 전달되므로, 분류 결과를 분할한다고 해서 확률 벡터를 단일 클래스만으로 줄일 수 없고, 깊이 결과를 분할한다고 해서 `(H, W)` 레이아웃이 손상되지 않습니다.

`len(result)`는 인스턴스를 계산합니다: 상자, 포인트, 임베딩, OCR 영역 또는 메시. 모든 밀집 전체 이미지 페이로드는 `1`로 계산됩니다. 아무 것도 없는 결과는 `0`입니다.

`to()`, `cpu()`, `cuda()` 및 `numpy()`는 각기 모든 채워진 슬롯이 변환된 새로운 `Results`를 반환합니다. 이들은 원본을 수정하지 않습니다.

`update()`는 제자리에서 변이하며, 이름이 지정된 슬롯을 교체하고 동일한 객체를 반환하는 유일한 방법입니다.

## 제이슨

<code-tabs name="json" />

`summary()`는 일반 딕셔너리 목록을 반환하고, `to_json()`는 그 목록을 `json.dumps`를 통해 통과시킨 것입니다. 둘 다 동일한 세 가지 인자를 사용합니다: `normalize=False`는 좌표를 `[0, 1]`로 전환하고, `decimals=5`는 반올림을 설정하며, `embeddings=False`는 임베딩 벡터가 포함될지 여부를 제어합니다.

행 모양은 페이로드를 따릅니다. 탐지 행에는 `name`, `class`, `confidence` 및 `box` 사전이 포함되며, 마스크가 존재할 때 `segments`를 가져오고, 방향이 있는 박스의 경우 `obb` 및 `corners`, 라디안 및 도 단위의 `gaze` 각도, 추적 시 `track_id`, 메시가 존재할 때 `mesh` 매개변수가 포함됩니다.

상자가 없는 경우, 한 페이로드가 행을 결정합니다: OCR은 `text`가 있는 각 영역마다 한 행을 출력하고, 포인트는 각 포인트마다 한 행을 출력하며, 파노라마틱은 `pixel_count`와 `pixel_fraction`가 있는 각 세그먼트마다 한 행을 출력하고, 시맨틱은 존재하는 각 클래스마다 한 행을 출력하며, 분류는 상위 다섯 클래스만 출력합니다. 깊이, 노멀, 가장자리, 복원 및 매팅은 각 픽셀 대신 맵을 설명하는 단일 요약 행을 출력합니다.

두 개의 페이로드는 의도적으로 축약되었습니다. 임베딩 벡터는 얼굴당 약 2KB인 512-플롯 행이므로 `embedding_dim`로만 보고됩니다; 값을 포함하려면 `embeddings=True`를 전달하십시오. 메시 정점은 절대로 포함되지 않는데, 이는 사람당 수만 개의 좌표이기 때문입니다. 지오메트리에 대해서는 `result.meshes.vertices`를 읽거나 `result.meshes.save_obj(path)`를 호출하십시오.

## 그리기 및 저장

<code-tabs name="saving" />

`predict(save=True)`는 주석을 달고 기록하는 경로입니다. 이 경로는 채워진 슬롯에서 그리기 루틴을 선택하므로, 의미론적 결과는 색상 마스크로 기록되고, 깊이 결과는 깊이 시각화로, 파노프틱 결과는 해당 세그먼트와 함께, 매트는 투명 배경 RGBA PNG로, 디텍터는 그 아래에 마스크가 있는 박스로 기록됩니다. 기록된 경로는 결과에 `result.saved_path`로 첨부됩니다.

`Results.plot()`는 이름이 시사하는 것보다 더 좁습니다. 이것은 일반 지도와 가장자리 지도에만 정의되며, 다른 것은 `NotImplementedError`를 발생시킵니다. 다른 작업에는 `save=True`를 사용하십시오.

`Results.save(path)` 또한 좁습니다: 투명 배경의 RGBA PNG 컷아웃으로 무광 결과를 기록하고 그렇지 않으면 `NotImplementedError`를 발생시킵니다. `Results.cutout()`는 그것을 기록하지 않고 동일한 RGBA 배열을 반환합니다. 두 함수 모두 원본 이미지가 필요하며, 이는 `result.path`에서 가져오거나 `image=`로 전달됩니다.

두 개의 페이로드는 각각 자체 작성기를 가지고 있습니다: 복원된 이미지를 위한 `result.restored.save(path)`와 메시를 위한 `result.meshes.save_obj(path, index=0)`.

파일이 어디에 위치하는지와 `output_path` 및 `output_file_format`가 어떻게 작동하는지에 대해서는 [예측 소스](/docs/predict/sources)를 참조하십시오.

## 내보낸 아티팩트는 동일한 객체를 반환

<code-tabs name="exported" />

`LibreYOLO()`는 파일 확장자에 따라 디스패치되므로, 내보낸 아티팩트도 `.pt` 체크포인트와 동일한 호출을 통해 로드되며 동일한 `Results`를 반환합니다. `.onnx`, `.engine`, `.pte` 및 `.mnn` 파일은 확장자에 따라 인식되며, OpenVINO, Paddle 및 ncnn 디렉터리와 Triton 모델 URL도 마찬가지입니다. `result.boxes.xyxy`를 읽는 코드는 모델을 내보낸 빌드로 교체해도 바뀌지 않습니다. 전체 형식 집합은 [Export](/docs/export)를 참조하십시오.

대신 런타임 자체의 API를 사용한다는 것은 전처리, 후처리 및 클래스 이름을 스스로 관리한다는 것을 의미합니다.
