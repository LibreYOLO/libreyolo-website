---
title: 결과 유형
seo_title: LibreYOLO 결과 객체 참조
description: >-
  LibreYOLO 결과 객체가 담을 수 있는 모든 페이로드, 작업 형태별로 하나의 슬롯: 상자(boxes), 마스크(masks),
  키포인트(keypoints), 확률(probs), OBB, 깊이(depth), OCR, 임베딩(embeddings) 등 총 열 개 이상.
lead: >-
  Results는 모든 LibreYOLO 모델의 단일 이미지별 반환 유형입니다. 18개의 선택적 페이로드 슬롯을 가지며, 각 작업 형태별
  하나씩 있고 모델이 생성한 것만 채웁니다.
keywords:
  - libreyolo 결과 객체
  - Results.boxes
  - Results.masks
  - Results.probs
  - Results.depth_map
  - Results.summary
  - libreyolo 결과 to_json
last_verified: 1.5.0
verification: >-
  슬롯 이름, 형태, 속성 및 기본값은 v1.5.0의 libreyolo/utils/results.py에서 읽음. 의미는 페이로드 클래스
  docstring에서 인용.
snippets:
  usage:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9t.pt")
        result = model(SAMPLE_IMAGE)

        print(result.orig_shape, result.path)
        print(result.boxes.xyxy)
        print(result.boxes.conf)
        print(result.names[int(result.boxes.cls[0])])
  convert:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9t.pt")
        result = model(SAMPLE_IMAGE)

        # 모든 페이로드는 함께 이동합니다.
        result = result.cpu().numpy()

        # 행, 단순 딕셔너리로, 그 다음 JSON으로.
        print(result.summary()[:1])
        print(result.to_json())
source_hash: 16f654364ae6448a
---

## 결과 객체

하나의 `Results`는 하나의 이미지를 설명합니다. 단일 이미지 소스는 그 중 하나를 반환하고, 목록 소스나 디렉토리는 목록을 반환하며, `stream=True`는 그것들을 생성하는 제너레이터를 반환합니다.

| 속성 | 타입 | 의미 |
|---|---|---|
| `orig_shape` | `(int, int)` | 원본 이미지 높이와 너비 |
| `path` | `str` | 입력이 디스크에서 온 경우 소스 경로 |
| `names` | `dict[int, str]` | 클래스 인덱스에서 클래스 이름 |
| `speed` | `dict[str, float]` | 단계별 밀리초 |
| `track_id` | 텐서 | 결과가 `track()`에서 나왔을 때 트랙 ID |
| `frame_idx` | `int` | 비디오 및 스트림 소스를 위한 프레임 인덱스 |
| `restore_scale` | `int` | 복원 결과의 출력-입력 업스케일 계수; `1` 그 외 모든 곳 |

<code-tabs name="usage" />

## 페이로드 슬롯

각 슬롯은 `None` 모델이 생성하지 않은 경우. 계열이 채우는 슬롯은 작업에 의해 결정됨.

| 슬롯 | 클래스 | 작업 |
|---|---|---|
| `boxes` | `Boxes` | 감지 |
| `masks` | `Masks` | 분할 |
| `keypoints` | `Keypoints` | 자세 |
| `probs` | `Probs` | 분류 |
| `obb` | `OBB` | OBB |
| `gaze` | `Gaze` | 시선 |
| `points` | `Points` | 포인트 |
| `semantic_mask` | `SemanticMask` | 의미론적 |
| `panoptic` | `PanopticSegmentation` | 전경적 |
| `depth_map` | `DepthMap` | 깊이 |
| `normal_map` | `NormalMap` | 법선 |
| `edges` | `EdgeMap` | 가장자리 |
| `restored` | `RestoredImage` | 복원 |
| `matte` | `Matte` | 매트 |
| `ocr` | `OCRRegions` | 광학문자인식 |
| `embeddings` | `Embeddings` | 삽입 |
| `identities` | `Identities` | 갤러리가 있는 삽입 |
| `meshes` | `Meshes` | 메시 |

`result.normals`는 `result.normal_map`의 읽기-쓰기 별칭입니다.

한 번에 여러 슬롯을 설정할 수 있습니다. 세분화(segmentation) 모델은 `boxes`와 `masks`를 모두 채우며; 시선(gaze) 모델은 얼굴 박스로 `boxes`를 채우고 각도로 `gaze`를 채웁니다; 메시(mesh) 모델은 사람 박스로 `boxes`를 채우고 그에 맞춰 행 단위로 `meshes`를 채웁니다.

## 박스

한 이미지에 대한 감지 박스

| 항목 | 반환값 |
|---|---|
| `xyxy` | 원본 이미지 픽셀에서의 모서리 좌표 |
| `xywh` | 픽셀 단위 중심 및 크기 |
| `xyxyn` | `[0, 1]`에 정규화된 모서리 |
| `xywhn` | `[0, 1]`에 정규화된 중심 및 크기 |
| `conf` | 박스별 신뢰도 |
| `cls` | 박스별 클래스 인덱스 |
| `id` | 상자당 추적 ID, 또는 `None` |
| `is_track` | 추적 ID가 있는 경우 `True` |
| `data` | 포장된 텐서 |

및 `with_orig_shape(orig_shape)` 는 해당 필드가 교체된 새 `Boxes` 를 반환합니다.

## 마스크

한 이미지의 인스턴스 마스크. `data` 는 마스크 텐서이며; `xy` 는 픽셀 단위의 인스턴스별 윤곽을 반환하고 `xyn` 는 이를 정규화된 상태로 반환합니다.

## 키포인트

`boxes` 와 행 기준으로 정렬된 포즈 키포인트. `xy` 는 키포인트별 좌표 쌍, `xyn` 는 정규화된 쌍입니다. `conf` 는 데이터에 해당 채널이 존재할 때 세 번째 채널이며, 그렇지 않으면 `None` 입니다. `has_visible` 는 불리언 배열로, `conf > 0` 가 있는 위치는 true, 신뢰도 채널이 없을 경우 전체가 true 입니다.

## 포인트

한 이미지에 대한 점 위치 지정. `data`는 `(N, 4)` 모양을 가지며, 행 수는 `x, y, class, confidence`입니다. 좌표는 절대 픽셀 단위입니다; `xy`, `cls`, `conf`가 열을 나누고, `xyn`가 좌표를 정규화합니다.

## 확률

분류 점수. `top1`는 최고 인덱스이며, `top5`는 상위 다섯 개 인덱스, `top1conf`와 `top5conf`는 해당 점수입니다.

## OBB

방향 상자. `data`는 행당 7개 또는 8개의 값을 가집니다: `xywhr`, 선택적 트랙 ID, 그 다음 신뢰도와 클래스.

| 구성원 | 반환 |
|---|---|
| `xywhr` | 중심, 크기 및 라디안 단위 회전 |
| `xyxyxyxy` | 픽셀 단위의 네 모서리 |
| `xyxyxyxyn` | 정규화된 네 모서리 |
| `xyxy` | 픽셀 단위 축 정렬 외형 |
| `conf`, `cls`, `id`, `is_track` | `Boxes`와 동일 |

## 시선

페이스별 시선 각도(라디안 단위), 형태는 `(N, 2)`, 행은 `boxes`의 얼굴 박스와 정렬됨. 열 0은 피치, 열 1은 요이며, L2CS 규칙: 양의 요는 시선을 피험자의 왼쪽으로 회전시키고 양의 피치는 아래로 회전시킴. `pitch_deg`와 `yaw_deg`는 도 단위로 변환하며, `direction_3d`는 단위 방향 벡터를 반환함.

## 의미적 마스크

원본 이미지 캔버스상의 정수 클래스 ID로 구성된 촘촘한 의미 지도, 형상은 `(H, W)`. `255`는 무시 값으로, 절대 클래스(`SemanticMask.IGNORE_INDEX`)로 계산되지 않음. `classes`는 존재하는 클래스 ID를 나열하고, `class_mask(class_id)`는 한 클래스의 불린 마스크를 반환함.

## 판옵틱 분할

모든 픽셀은 정확히 하나의 겹치지 않는 세그먼트를 가지며, 사물 영역과 개체 인스턴스를 통합합니다. `data`은 `(H, W)` 정수 세그먼트-ID 맵입니다; 세그먼트 ID `0`는 레이블이 없습니다 (`PanopticSegmentation.IGNORE_INDEX`). `segments_info`는 각 세그먼트마다 하나씩 딕셔너리의 리스트이며, 각 딕셔너리에는 최소한 `{"id": int, "category_id": int}`가 포함되어 있으며, 여기서 `id`는 맵의 값과 일치하고 `category_id`는 `names`를 인덱싱합니다. `segment_ids`는 존재하는 ID를 나열하고, `segment_mask(segment_id)`는 하나의 세그먼트의 불리언 마스크를 반환합니다.

사물 대 배경(stuff)은 세그먼트가 아닌 카테고리의 속성입니다. 페이로드는 이를 각 세그먼트에 `"isthing": bool`로 비정규화할 수 있으며, 그렇게 할 경우 값은 카테고리 수준 맵과 일치해야 합니다.

## DepthMap

촘촘한 상대 역-깊이 맵, 원본 이미지 캔버스에서 동일한 형태 `(H, W)`의 부동 소수점 데이터. 값이 클수록 카메라에 더 가까움을 의미함. 값은 상대적이며, 실제 미터를 의미하지 않음. `min`, `max`, `mean`는 유한 값에 대해 계산되고, `normalized()`는 맵을 `[0, 1]`로 재조정함.

## 법선 맵

촘촘한 표면 법선 필드, 원본 이미지 캔버스에서 float32 `(H, W, 3)`, OpenCV 카메라 좌표계에서: `+x` 오른쪽, `+y` 아래, `+z` 장면 안쪽. 법선은 카메라를 향함, 따라서 정면 평행 표면은 `(0, 0, -1)`. 각 픽셀은 단위 벡터임. `assert_normalized(atol=1e-4)`가 이를 확인함.

## 에지 맵

밀집 가장자리 확률 맵, float32 `(H, W)` 원본 이미지 캔버스에 표시되며, `0`는 비가장자리이고 `1`는 가장자리입니다. 연속 맵이 유지되므로 임계값은 호출자가 선택할 수 있습니다: `binary(threshold=0.5)`가 하나를 적용하고 `array`가 numpy 뷰를 반환합니다.

## 복원된 이미지

복원된 RGB 이미지, `(H, W, 3)` uint8. 초해상도의 경우 캔버스는 입력의 `Results.restore_scale` 배입니다. `array`는 numpy 뷰를 반환하고 `save(path)`가 이미지를 씁니다.

## 마스크

부드러운 불투명도 마스크, float32 `(H, W)` 원본 이미지 캔버스의 `[0, 1]`에 표시됩니다. `1`는 완전히 전경이고 `0`는 완전히 배경입니다. 부드러운 마스크는 0.5에서 임계치가 적용된 하드 배경 제거 마스크를 포함하고, 이진 마스크가 버리는 안티앨리어싱 가장자리를 유지합니다. `array`가 numpy 뷰를 반환합니다.

매트 결과에서, `Results.cutout(image=None)`는 네 번째 채널이 매트인 RGBA `(H, W, 4)` uint8 배열을 반환하며, `Results.save(path, image=None)`는 해당 컷아웃을 투명 배경 PNG로 저장합니다. 둘 다 `image`가 주어지면 그 RGB를 사용하고, 그렇지 않으면 `Results.path`에서 다시 불러옵니다.

## OCRRegions

위치한 텍스트와 전사. `data`는 원본 이미지 픽셀 단위의 `(N, 4, 2)` float 폴리곤으로, 좌상단, 우상단, 우하단, 좌하단 순서로 정렬되며, 영역은 읽기 순서(위에서 아래로, 왼쪽에서 오른쪽으로)로 제공됩니다. `texts`는 N개의 전사 목록입니다. `conf`는 영역별 인식 점수이고 `det_conf`는 검출 점수로, 둘 다 `(N,)`입니다.

검출 사각형은 실제 폴리곤이므로 `Results.boxes`를 채우지 않습니다. `xyxy`는 축에 맞춘 외곽을 제공합니다.

## 임베딩

`embed` 작업에서 L2-정규화된 벡터, 항상 `(N, D)` 형태. 전체 이미지 결과는 한 행만 가지고 상자는 없음; 지역 임베딩은 `boxes`와 행 기준으로 정렬됨. 각 행이 정규화되어 있으므로, 코사인 유사도는 내적(dot product)임.

| 구성원 | 반환 |
|---|---|
| `dim` | `D` |
| `normalized` | 행들, 재정규화됨 |
| `similarity(other)` | 다른 `Embeddings` 또는 텐서에 대한 쌍별 코사인 유사도 |
| `verify(i, j, threshold=0.4)` | 행 `i`와 `j`가 일치할 때 `True` |

## 식별자

지명된 갤러리 매칭, `embeddings`와 행 단위 정렬. `Gallery`가 `embed` 예측에 전달될 때 생성됨. `name`는 항목이 매치 임계값 아래인 `None` 목록이며, 가장 가까운 임계값 이하의 이름은 절대 추측되지 않음. `score`는 매치 점수 배열이고 `data`는 그들을 쌍으로 만듦.

## 메시

매개변수화된 인간 신체 메시는 `boxes`의 사람 박스와 행 단위 정렬. 모든 것은 원본 이미지의 카메라 프레임에 있음. `transl`는 미터 단위이고 `+z`는 카메라에서 멀어지는 방향; `vertices`와 `joints3d`는 미터 단위이며 이미 `transl`를 포함; `joints2d`는 원본 이미지 캔버스의 픽셀 단위이며 네트워크가 본 크롭이 아님. 어떤 필드도 세계 또는 중력 프레임을 가지지 않음.

매개변수 레이아웃은 바디 모델마다 다르므로, 형태에 대한 내용은 하드코딩되어 있지 않습니다. `body_model`는 매개변수화를 명명하며, 카운트는 텐서에서 다시 읽어옵니다: `num_vertices`, `num_joints`, `num_betas` 및 `has_vertices`. `params`는 매개변수 사전을 반환하고, `save_obj(path, index=0)`는 하나의 메시를 작성합니다. 필드는 `global_orient`, `body_pose`, `betas`, `transl`, `vertices`, `faces`, `joints3d`, `joints2d`, `conf`, `focal_length` 및 `extras`입니다.

`body_model="mhr"`의 회전은 축-각(axis-angle) 대신 라디안 단위의 오일러 각이며, `body_pose`는 각 관절당 하나의 삼중항(triplet) 대신 평면(per-joint) 매개변수 벡터이고, `betas`는 항등 블렌드셰이프 계수(identity blendshape coefficients)입니다. 골격 스케일, 손 포즈 및 얼굴 표정은 `extras`에 위치합니다.

## 변환 및 선택

모든 페이로드는 `to(*args, **kwargs)`, `cpu()`, `cuda()` 및 `numpy()`를 포함하며, `Results`에서 그 중 하나를 호출하면 모든 채워진 슬롯에 동시에 적용됩니다.

<code-tabs name="convert" />

`result[idx]`는 행 정렬된 페이로드에서 행을 선택합니다. `len(result)`는 박스가 없을 경우 탐지 수 또는 포인트 수입니다. `result.update(...)`는 명명된 슬롯으로 교체된 복사본을 반환합니다; 모든 슬롯과 `track_id` 및 `restore_scale`를 허용합니다.

## summary 및 to_json

`summary(normalize=False, decimals=5, embeddings=False)`는 활성화된 슬롯에 따라 탐지, 세그먼트, 포인트 또는 영역별 한 행씩 평범한 dict 목록을 반환합니다. `to_json(**kwargs)`는 자신의 인수를 `summary`에 전달하고 JSON 문자열을 반환합니다.

`plot()`는 조밀한 일반 또는 엣지 결과를 표준 시각화 방식으로 렌더링합니다; 다른 결과 유형의 경우 예외를 발생시킵니다. 다른 작업에 대한 주석이 달린 이미지는 `predict(save=True)`에서 가져옵니다.
