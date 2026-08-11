---
title: 결과 작업
seo_title: LibreYOLO 결과 객체
description: >-
  이미지당 하나의 결과 객체, 페이로드 유형별 슬롯 포함: 박스, 마스크, 키포인트, 확률, 깊이, 파노라믹, OCR 등. 플로팅, 저장 및
  JSON 지원.
lead: >-
  모든 예측은 이미지당 하나의 결과 객체를 반환합니다. 각 페이로드 종류별로 명명된 슬롯이 있으며, 모델이 생성한 것 외에는 모두 비어 있고,
  내보낸 아티팩트에도 동일한 슬롯이 있습니다.
keywords:
  - 파이썬 yolo 결과 객체
  - results.boxes xyxy
  - 결과를 json으로 변환
  - 주석이 달린 이미지 저장
  - 파이썬 세그멘테이션 마스크
  - 키포인트 결과
  - 깊이 맵 결과
  - 결과 요약
  - onnx 같은 결과
last_verified: 1.5.0
verification: >-
  페이로드 클래스, 슬롯, 이동 의미론, summary(), to_json(), plot(), save() 및 cutout()는
  libreyolo/utils/results.py.에서 Annotation과 디스크 기록 동작을 읽고,
  InferenceRunner._save_annotated_image는 libreyolo/models/base/inference.py에서,
  resolve_save_path는 libreyolo/utils/general.py.에서 처리됨. 접미사 디스패치는
  libreyolo/models/__init__.py.에서 LibreYOLO()에서 처리됨.
snippets:
  basic:
    - label: 박스
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")
        result = model(SAMPLE_IMAGE)

        print(result.orig_shape)   # 원본 이미지의 (높이, 너비)
        print(result.path)         # 원본 경로, 메모리 내 입력의 경우 None

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
        print(result.boxes.xywh[:1])    # 픽셀, 중심 x, 중심 y, w, h
        print(result.boxes.xyxyn[:1])   # 너비와 높이로 나눈 동일한 박스
        print(result.boxes.xywhn[:1])
    - label: NumPy와 디바이스
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")
        result = model(SAMPLE_IMAGE)

        # 이들 각각은 새로운 Results를 반환; 원본은 변경되지 않음.
        as_numpy = result.numpy()
        on_cpu = result.cpu()

        print(type(as_numpy.boxes.xyxy).__name__)
        print(type(on_cpu.boxes.xyxy).__name__)
  json:
    - label: summary와 to_json
      language: python
      code: |
        import json

        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")
        result = model(SAMPLE_IMAGE)

        rows = result.summary()
        print(json.dumps(rows[:2], indent=2))

        # 동일한 내용의 문자열, 동일한 키워드 인수로
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

        # save=True는 페이로드를 저장하고 runs/detect/predict*. 아래에 기록합니다
        result = model(SAMPLE_IMAGE, save=True)
        print(result.saved_path)
  exported:
    - label: 추가 내보내기(export extra)를 설치합니다
      language: bash
      code: |
        pip install "libreyolo[onnx]"
    - label: 내보낸 아티팩트에서 같은 결과를 가져옵니다
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")
        path = model.export(format="onnx")   # 기록된 경로를 반환합니다

        # LibreYOLO()는 파일 접미사에 따라 처리합니다.
        exported = LibreYOLO(path)
        result = exported(SAMPLE_IMAGE)

        print(type(result).__name__, len(result.boxes))
source_hash: 548dbc9c7f5552ec
---

## 하나의 객체, 하나의 슬롯당 페이로드 하나

하나의 이미지에 대한 예측은 하나의 `Results`를 반환합니다. 이는 18개의 페이로드 슬롯을 가지며, 모델은 자신의 작업이 생성하는 슬롯만 채웁니다. 나머지 슬롯은 모두 `None`이며, 따라서 탐지기에서 `result.masks`를 읽는 것은 `None`이지 오류가 아닙니다.

| 슬롯 | 클래스 | 형태 | 생성자 |
|---|---|---|---|
| `boxes` | `Boxes` | `(N, 4)` 및 점수와 클래스 | 감지, 및 먼저 위치를 지정하는 모든 작업 |
| `masks` | `Masks` | `(N, H, W)` | 인스턴스 분할 |
| `keypoints` | `Keypoints` | `(N, K, 2)` 또는 `(N, K, 3)` | 포즈 |
| `probs` | `Probs` | `(C,)` | 분류 |
| `obb` | `OBB` | `(N, 7)` 또는 `(N, 8)` | 방향 상자 |
| `gaze` | `Gaze` | 라디안 단위의 피치 및 요 | 시선 추정 |
| `points` | `Points` | x, y, 클래스, 신뢰도로서 | 점 위치 지정 |
| `semantic_mask` | `SemanticMask` | `(H, W)` 클래스 ID | 시맨틱 분할 |
| `panoptic` | `PanopticSegmentation` | `(H, W)` 세그먼트 ID와 `segments_info` 포함 | 판옵틱 분할 |
| `depth_map` | `DepthMap` | `(H, W)` 부동 소수점 | 깊이 추정 |
| `normal_map` | `NormalMap` | `(H, W, 3)` 단위 벡터 | 표면 법선 |
| `edges` | `EdgeMap` | `(H, W)` `[0, 1]`의 부동 소수점 | 에지 검출 |
| `restored` | `RestoredImage` | `(H, W, 3)` uint8 RGB | 복원 및 초해상도 |
| `matte` | `Matte` | `(H, W)` `[0, 1]`의 부동 소수점 | 알파 매팅 및 배경 제거 |
| `ocr` | `OCRRegions` | `(N, 4, 2)` 다각형과 전사 | 텍스트 감지 및 인식 |
| `embeddings` | `Embeddings` | `(N, D)` L2 정규화된 행 | `embed` 작업 |
| `identities` | `Identities` | N 이름과 점수 | 갤러리가 있는 `embed` 작업 |
| `meshes` | `Meshes` | 신체 매개변수와 선택적 정점 | 신체 메시 복원 |

그들과 함께 모든 결과에 있는 필드: `orig_shape`를 `(height, width)`로, `path`(소스 경로 또는 메모리 내 입력의 경우 `None`), 클래스 ID를 클래스 이름에 매핑하는 `names`, 비디오와 라이브 프레임용 `frame_idx`, 추적 시 `track_id`, 그리고 복원 결과의 정수 업스케일 계수 `restore_scale`.

`result.normals`는 `result.normal_map`의 별칭입니다.

`result.speed`는 모든 결과에 존재하지만 [앙상블](/docs/predict/ensembling)에서만 채워지며, 그 키는 밀리초 단위로 `member_0`, `member_1` 및 `fusion`입니다. 단일 모델의 경우 빈 딕셔너리로 유지됩니다.

## 박스

<code-tabs name="basic" />

`Boxes` 좌표와 점수를 하나의 패킹된 텐서가 아닌 별도의 배열로 저장합니다.

| 속성 | 내용 |
|---|---|
| `xyxy` | `(N, 4)` 절대 픽셀, x1 y1 x2 y2 |
| `xywh` | `(N, 4)` 절대 픽셀, 중심 x, 중심 y, 너비, 높이 |
| `xyxyn`, `xywhn` | 이미지 너비와 높이로 나눈 동일 값 |
| `conf` | `(N,)` 신뢰도 |
| `cls` | `(N,)` 클래스 ID, float 형식으로 |
| `id` | `(N,)` 트랙 ID, 또는 `None` |
| `is_track` | `id`가 설정되어 있는지 여부 |
| `data` | 모든 것을 연결: 박스, 선택적 ID, 신뢰도, 클래스 |

`cls`는 float 배열이므로 `result.names[int(cls)]`로 사용하십시오.

`xyxyn`와 `xywhn`는 `orig_shape`가 필요하며, 이는 `Results`가 대신 채워줍니다.

## 조밀한 페이로드

전체 이미지를 아우르는 페이로드는 인스턴스별 페이로드와 다르게 동작하며, 슬라이싱할 때 중요합니다.

`SemanticMask`는 원본 캔버스에서 `(H, W)` 클래스 ID를 보유하며, `255`는 클래스에 포함되지 않는 무시값으로 예약됩니다. `classes`는 존재하는 ID를 나열하고 이를 제외하며; `class_mask(id)`는 불리언 `(H, W)`를 반환합니다.

`PanopticSegmentation`는 `(H, W)` 세그먼트 ID를 보유하며, `0`는 void ID이고, 최소한 `id`와 `category_id`를 포함하는 dict 리스트가 있습니다. `segment_ids`는 존재하는 ID를 나열하고, `segment_mask(id)`는 하나를 선택합니다.

`DepthMap`는 `(H, W)` 상대 역깊이를 보유하며: 값이 클수록 더 가까움을 의미하고, 값은 미터 단위가 아닙니다. 이것은 유한값 상에서 `min`, `max`, `mean`를 노출하며, `normalized()`는 `[0, 1]`로 재조정됩니다.

`NormalMap`는 OpenCV 카메라 프레임에서 `(H, W, 3)` 단위 벡터를 보유하며, `+x`는 오른쪽, `+y`는 아래, `+z`는 장면 안쪽을 향해 있으므로, 카메라를 향하는 표면은 `(0, 0, -1)`입니다. `assert_normalized()`는 모든 픽셀이 유한하고 단위 길이인지 확인합니다.

`EdgeMap`는 `[0, 1]`에서 `(H, W)` float32를 보유합니다. 연속적인 지도는 임계값 처리되지 않고 유지되므로, `binary(threshold=0.5)`는 컷오프를 선택하는 위치입니다.

`Matte`는 `[0, 1]`에서 `(H, W)` float32를 보유하며, `1`는 완전히 전경입니다. `array`는 이를 float32로 클리핑하여 반환합니다.

`RestoredImage`는 `(H, W, 3)` uint8 RGB를 보유하며, `array`는 원시 ndarray용이고 `save(path)`는 이를 쓰는 용도입니다.

`Probs`는 이미지에 대한 하나의 확률 벡터를 보유합니다. `top1`와 `top5`는 클래스 인덱스이고, `top1conf`와 `top5conf`는 일치 점수입니다.

`Embeddings`는 이미 L2 정규화된 `(N, D)` 행들을 보유하므로, 코사인 유사도는 점 곱(dot product)입니다. `similarity(other)`는 갤러리 대비 `(N, M)`를 반환하거나 단일 벡터 대비 `(N,)`를 반환하며, `verify(i, j, threshold=0.4)`는 두 행을 비교합니다.

`OCRRegions`는 읽기 순서대로 `(N, 4, 2)` 다각형을 보유하며, 모서리는 좌상, 우상, 우하, 좌하 순서로 정렬됩니다. 텍스트는 `texts`에 있으며, 인식 점수는 `conf`, 검출 점수는 `det_conf`에 있습니다. 이것들이 실제로 회전된 다각형이기 때문에 `boxes`를 채우지 않으며; 사각형이 필요할 때 `ocr.xyxy`는 축에 맞춘 외접 다각형을 제공합니다.

## 자르기 및 이동

`result[i]`는 하나의 인스턴스를 보유한 새로운 `Results`를 반환합니다. 인스턴스별 페이로드는 잘려 나가지만, 전체 이미지 페이로드는 그대로 전달되므로, 분류 결과를 자른다고 해서 확률 벡터를 단일 클래스만으로 잘라낼 수 없으며, 깊이 결과를 자른다고 해서 `(H, W)` 레이아웃이 손상되지 않습니다.

`len(result)`는 인스턴스를 계산합니다: 상자, 점, 임베딩, OCR 영역 또는 메시. 모든 조밀한 전체 이미지 페이로드는 `1`로 계산됩니다. 내용이 없는 결과는 `0`입니다.

`to()`, `cpu()`, `cuda()` 및 `numpy()`는 각각 모든 채워진 슬롯이 변환된 새로운 `Results`를 반환합니다. 원본은 수정하지 않습니다.

`update()`는 한 곳에서 변경하는 유일한 메서드로, 명명된 슬롯을 교체하고 동일한 객체를 반환합니다.

## JSON

<code-tabs name="json" />

`summary()`는 일반 딕셔너리 목록을 반환하며, `to_json()`는 해당 목록을 `json.dumps`를 통해 전달한 것입니다. 둘 다 동일한 세 가지 인수를 받습니다: `normalize=False`는 좌표를 `[0, 1]`로 전환하고, `decimals=5`는 반올림을 설정하며, `embeddings=False`는 임베딩 벡터 포함 여부를 제어합니다.

행 모양은 페이로드를 따릅니다. 감지 행은 `name`, `class`, `confidence` 및 `box` 사전을 포함하며, 마스크가 있을 때 `segments`를 선택하고, 지향 상자에는 `obb` 및 `corners`, 각도는 라디안과 도 단위 모두에서 `gaze`, 추적 중에는 `track_id`, 메시가 있을 때는 `mesh` 매개변수를 가져옵니다.

상자가 없는 경우 하나의 페이로드가 행을 결정합니다: OCR은 각 영역에 대해 하나의 행을 `text`로 방출하며, 포인트는 각 포인트에 대해 하나의 행, 파노라마는 각 세그먼트에 대해 `pixel_count` 및 `pixel_fraction`와 함께 하나의 행, 시맨틱은 존재하는 클래스마다 하나의 행, 분류는 상위 다섯 개 클래스를 방출합니다. 깊이, 노멀, 에지, 복원 및 매팅은 각 픽셀이 아닌 맵을 설명하는 단일 요약 행을 방출합니다.

두 개의 페이로드는 의도적으로 축약되었습니다. 임베딩 벡터는 `embedding_dim`로만 보고되며, 512-플롯 행이 얼굴당 약 2KB이기 때문입니다. 값을 포함하려면 `embeddings=True`를 전달하세요. 메시 정점은 절대 포함되지 않으며, 사람당 수만 개의 좌표가 필요합니다. 지오메트리를 원하면 `result.meshes.vertices`를 읽거나 `result.meshes.save_obj(path)`를 호출하세요.

## 그리기 및 저장

<code-tabs name="saving" />

`predict(save=True)`는 주석을 달고 저장하는 경로입니다. 채워진 슬롯에서 그리기 루틴을 선택하므로, 의미 론적 결과는 색상 마스크로, 깊이 결과는 깊이 시각화로, 파노프틱 결과는 그 세그먼트와 함께, 마스크는 투명 배경의 RGBA PNG로, 감지 결과는 하단 마스크가 있는 박스로 작성됩니다. 작성된 경로는 결과에 `result.saved_path`로 첨부됩니다.

`Results.plot()`는 이름이 암시하는 것보다 좁습니다. 이는 일반 맵과 엣지 맵에만 정의되며, 다른 것에는 `NotImplementedError`를 발생시킵니다. 다른 작업에는 `save=True`를 사용하십시오.

`Results.save(path)`도 마찬가지로 좁습니다: 투명 배경의 RGBA PNG 컷아웃으로 매트 결과를 기록하며, 그렇지 않으면 `NotImplementedError`를 발생시킵니다. `Results.cutout()`는 파일에 기록하지 않고 동일한 RGBA 배열을 반환합니다. 둘 다 소스 이미지가 필요하며, 이는 `result.path`에서 가져오거나 `image=`로 전달됩니다.

두 페이로드는 자체 작성기를 갖습니다: 복원된 이미지용 `result.restored.save(path)`와 메시용 `result.meshes.save_obj(path, index=0)`입니다.

파일이 위치하는 곳과 `output_path` 및 `output_file_format`가 어떻게 작동하는지에 대해서는 [예측 소스](/docs/predict/sources)를 참조하십시오.

## 내보낸 아티팩트는 동일한 객체를 반환

<code-tabs name="exported" />

`LibreYOLO()`는 파일 접미사에 따라 분기되므로, 내보낸 아티팩트는 `.pt` 체크포인트와 동일한 호출을 통해 로드되며 동일한 `Results`를 반환합니다. `.onnx`, `.engine`, `.pte` 및 `.mnn` 파일은 접미사로 인식되며, OpenVINO, Paddle 및 ncnn 디렉토리와 Triton 모델 URL도 마찬가지입니다. `result.boxes.xyxy`를 읽는 코드는 모델을 내보낸 빌드로 교체해도 변경되지 않습니다. 전체 형식 세트는 [Export](/docs/export)를 참조하십시오.

런타임 자체 API를 사용하는 것은 전처리, 후처리 및 클래스 이름을 직접 관리해야 함을 의미합니다.
