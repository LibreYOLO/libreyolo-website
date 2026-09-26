---
title: 결과 유형
seo_title: LibreYOLO 결과 객체 참조
description: >-
  LibreYOLO Results 객체가 담을 수 있는 모든 페이로드, 작업 형태별 슬롯 하나: 박스, 마스크, 키포인트, 확률, OBB,
  깊이, OCR, 임베딩 및 추가 열 개.
lead: >-
  Results는 모든 LibreYOLO 모델의 단일 이미지별 반환 유형입니다. 그것은 18개의 선택적 페이로드 슬롯을 가지고 있으며, 각
  슬롯은 하나의 작업 형태에 해당하며, 모델이 생성한 것만 채웁니다.
keywords:
  - libreyolo 결과 객체
  - 결과.상자
  - 결과.마스크
  - 결과.확률
  - 결과.깊이_맵
  - 결과.요약
  - libreyolo 결과를 json으로
last_verified: 1.5.0
verification: >-
  슬롯 이름, 형태, 속성 및 기본값은 v1.5.0의 libreyolo/utils/results.py에서 읽었습니다. 의미는 페이로드 클래스
  도큐스트링에서 인용했습니다.
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

        # 모든 페이로드가 함께 이동합니다.
        result = result.cpu().numpy()

        # 행을 일반 딕셔너리로, 그 다음에는 JSON으로.
        print(result.summary()[:1])
        print(result.to_json())
source_hash: 16f654364ae6448a
---

## 결과 객체

하나의 `Results`는 하나의 이미지를 설명합니다. 단일 이미지 소스는 그 중 하나를 반환하고, 목록 소스나 디렉터리는 목록을 반환하며, `stream=True`는 그것들을 생성하는 제너레이터를 반환합니다.

| 속성 | 타입 | 의미 |
|---|---|---|
| `orig_shape` | `(int, int)` | 원본 이미지 높이와 너비 |
| `path` | `str` | 입력이 디스크에서 왔을 때의 소스 경로 |
| `names` | `dict[int, str]` | 클래스 인덱스를 클래스 이름으로 |
| `speed` | `dict[str, float]` | 단계별 밀리초 |
| `track_id` | 텐서 | 결과가 `track()`에서 나왔을 때 ID를 추적하십시오 |
| `frame_idx` | `int` | 비디오 및 스트림 소스를 위한 프레임 인덱스 |
| `restore_scale` | `int` | 복원 결과의 출력-입력 업스케일 비율; 다른 모든 곳에서는 `1` |

<code-tabs name="usage" />

## 페이로드 슬롯

각 슬롯은 모델이 생성하지 않는 한 `None`입니다. 계열이 채우는 슬롯은 그 계열의 작업에 따라 결정됩니다.

| 슬롯 | 수업 | 작업 |
|---|---|---|
| `boxes` | `Boxes` | 탐지하다 |
| `masks` | `Masks` | 세그먼트 |
| `keypoints` | `Keypoints` | 자세 |
| `probs` | `Probs` | 분류하다 |
| `obb` | `OBB` | obb |
| `gaze` | `Gaze` | 응시 |
| `points` | `Points` | 점 |
| `semantic_mask` | `SemanticMask` | 의미론의 |
| `panoptic` | `PanopticSegmentation` | 파노프틱 |
| `depth_map` | `DepthMap` | 깊이 |
| `normal_map` | `NormalMap` | 보통 |
| `edges` | `EdgeMap` | 모서리 |
| `restored` | `RestoredImage` | 복원하다 |
| `matte` | `Matte` | 매트 |
| `ocr` | `OCRRegions` | 광학 문자 인식 |
| `embeddings` | `Embeddings` | 삽입하다 |
| `identities` | `Identities` | 갤러리와 함께 삽입 |
| `meshes` | `Meshes` | 메시 |

`result.normals`는 `result.normal_map`의 읽기-쓰기 별칭입니다.

한 번에 하나 이상의 슬롯을 설정할 수 있습니다. 세분화 모델은 `boxes`와 `masks`를 모두 채우고, 시선 모델은 `boxes`를 얼굴 상자로, `gaze`를 각도로 채우며, 메시 모델은 `boxes`를 사람 상자로, `meshes`를 이와 행 정렬하여 채웁니다.

## 상자들

한 이미지에 대한 탐지 상자.

| 회원 | 반환 |
|---|---|
| `xyxy` | 원본 이미지 픽셀의 모서리 좌표 |
| `xywh` | 픽셀 단위의 중심과 크기 |
| `xyxyn` | 모서리 `[0, 1]`에 표준화됨 |
| `xywhn` | 중심과 크기가 `[0, 1]`에 맞게 정규화됨 |
| `conf` | 상자별 신뢰도 |
| `cls` | 상자별 클래스 인덱스 |
| `id` | 상자별 트랙 ID, 또는 `None` |
| `is_track` | 트랙 ID가 있을 때 `True` |
| `data` | 패킹된 텐서 |

`with_id(id)`와 `with_orig_shape(orig_shape)`는 해당 필드가 교체된 새로운 `Boxes`를 반환합니다.

## 마스크

한 이미지에 대한 인스턴스 마스크. `data`는 마스크 텐서이며, `xy`는 픽셀 단위의 인스턴스 윤곽선을 반환하고 `xyn`는 정규화된 윤곽선을 반환합니다.

## 핵심 포인트

포즈 키포인트는 `boxes`와 행 정렬되어 있습니다. `xy`는 각 키포인트별 좌표 쌍이며 `xyn`는 정규화된 쌍입니다. 데이터에 채널이 하나 있을 때 `conf`가 세 번째 채널이며, 그렇지 않으면 `None`입니다. `has_visible`는 불리언 배열로, `conf > 0`인 곳은 true이며, 신뢰도 채널이 없으면 모두 true입니다.

## 포인트

한 이미지에 대한 점 위치 지정. `data`의 모양은 `(N, 4)`이며 행 수는 `x, y, class, confidence`입니다. 좌표는 절대 픽셀 단위입니다; `xy`와 `cls`와 `conf`가 열을 나누고, `xyn`가 좌표를 정규화합니다.

## 아마

분류 점수. `top1`는 우승 지수이고, `top5`는 다섯 개의 최고 지수이며, `top1conf`와 `top5conf`는 그들의 점수입니다.

## OBB

정렬된 상자들. `data`는 행당 7개 또는 8개의 값을 가집니다: `xywhr`, 선택적 트랙 ID, 그 다음 신뢰도와 클래스.

| 회원 | 반환 |
|---|---|
| `xywhr` | 중심, 크기 및 라디안 단위 회전 |
| `xyxyxyxy` | 픽셀 단위의 네 모서리 |
| `xyxyxyxyn` | 네 모서리가 정규화되었습니다 |
| `xyxy` | 픽셀 단위의 축 정렬 외형 |
| `conf`, `cls`, `id`, `is_track` | `Boxes` 기준으로 |

## 응시

각 얼굴당 시선 각도(라디안 단위), 형상 `(N, 2)`, 행은 `boxes`의 얼굴 상자와 정렬됨. 열 0은 피치이고 열 1은 요우이며, L2CS 규칙에 따름: 양의 요우는 시선을 피험자의 왼쪽으로 회전시키고 양의 피치는 시선을 아래로 회전시킴. `pitch_deg`와 `yaw_deg`는 각도를 도 단위로 변환하며, `direction_3d`는 단위 방향 벡터를 반환함.

## 시맨틱마스크

원본 이미지 캔버스 상의 정수 클래스 ID의 조밀한 의미 지도. `255`는 무시 값이며 클래스(`SemanticMask.IGNORE_INDEX`)로 간주되지 않습니다. `classes`는 존재하는 클래스 ID를 나열하고, `class_mask(class_id)`는 한 클래스에 대한 불린 마스크를 반환합니다.

## 파노프틱 세그멘테이션

모든 픽셀은 정확히 하나의 겹치지 않는 세그먼트를 가지며, 사물 영역과 물체 인스턴스를 통합합니다. `data`는 `(H, W)` 정수 세그먼트-ID 맵입니다; 세그먼트 ID `0`는 레이블이 없습니다 (`PanopticSegmentation.IGNORE_INDEX`). `segments_info`는 세그먼트별로 하나씩 dict의 리스트이며, 각 dict에는 적어도 `{"id": int, "category_id": int}`가 있고, 여기서 `id`는 맵의 값과 일치하며 `category_id`는 `names`를 인덱싱합니다. `segment_ids`는 존재하는 ID들을 나열하고 `segment_mask(segment_id)`는 하나의 세그먼트에 대한 불리언 마스크를 반환합니다.

사물 대 물질(Thing-versus-stuff)은 세그먼트가 아닌 범주의 속성입니다. 페이로드는 이를 각 세그먼트에 `"isthing": bool`로 비정규화할 수 있으며, 그렇게 할 때 값은 범주 수준의 맵과 일치해야 합니다.

## 깊이맵

조밀한 상대 역-깊이 지도, 원본 이미지 캔버스에서 `(H, W)` 형태의 부동 소수점. 값이 높을수록 카메라에 더 가까움. 값은 상대적이며, 미터 단위가 아님. `min`, `max` 및 `mean`는 유한 값에 대해 계산되며, `normalized()`는 지도를 `[0, 1]`로 재조정함.

## 노말맵

조밀한 표면 법선 필드, 원본 이미지 캔버스에서 float32 `(H, W, 3)`, OpenCV 카메라 프레임에서: `+x` 오른쪽, `+y` 아래, `+z` 장면 안쪽. 법선은 카메라를 향하므로, 정면 평행 표면은 `(0, 0, -1)`. 모든 픽셀은 단위 벡터입니다. `assert_normalized(atol=1e-4)`가 그 불변성을 확인합니다.

## 엣지맵

조밀한 엣지-확률 맵, 원본 이미지 캔버스상의 float32 `(H, W)`, 여기서 `0`는 비엣지이고 `1`는 엣지입니다. 연속 맵은 유지되어 임계값이 호출자의 선택에 남도록 합니다: `binary(threshold=0.5)`가 하나를 적용하고 `array`가 numpy 뷰를 반환합니다.

## 복원된 이미지

복원된 RGB 이미지, `(H, W, 3)` uint8. 초해상도에서는 캔버스가 입력의 `Results.restore_scale` 배입니다. `array`는 numpy 뷰를 반환하고 `save(path)`는 이미지를 저장합니다.

## 무광

부드러운 불투명 매트, 원본 이미지 캔버스에서 float32 `(H, W)`를 `[0, 1]`에 둡니다. `1`는 완전히 전경이고 `0`는 완전히 배경입니다. 부드러운 매트는 0.5에서 임계값이 적용된 하드 배경 제거 마스크를 포함하며, 이진 마스크가 버리는 안티앨리어싱 가장자리를 유지합니다. `array`는 numpy 뷰를 반환합니다.

무광 결과에서, `Results.cutout(image=None)`는 네 번째 채널이 매트인 RGBA `(H, W, 4)` uint8 배열을 반환하고, `Results.save(path, image=None)`는 그 잘라낸 부분을 투명 배경 PNG로 저장합니다. 둘 다 `image`가 주어지면 RGB를 가져오고, 그렇지 않으면 `Results.path`에서 다시 로드합니다.

## OCR영역

원문 텍스트가 있는 위치 정보와 전사본이 포함되어 있습니다. `data`는 원본 이미지 픽셀 단위의 부동 소수점 다각형으로, 좌상, 우상, 우하, 좌하 순서이며, 영역은 읽는 순서대로 위에서 아래, 왼쪽에서 오른쪽으로 정렬됩니다. `texts`는 N개의 전사본 목록입니다. `conf`는 영역별 인식 점수이고 `det_conf`는 탐지 점수이며, 둘 다 `(N,)`입니다.

검출 쿼드는 실제 다각형이므로 `Results.boxes`를 채우지 않습니다. `xyxy`는 축에 정렬된 외피를 제공합니다.

## 임베딩

`embed` 작업에서 나온 L2 정규화 벡터는 항상 `(N, D)` 형태입니다. 전체 이미지 결과는 한 행을 가지며 상자는 없습니다; 영역 임베딩은 `boxes`와 행 기준으로 정렬됩니다. 각 행이 정규화되었기 때문에 코사인 유사도는 내적입니다.

| 회원 | 반환 |
|---|---|
| `dim` | `D` |
| `normalized` | 행, 재정규화됨 |
| `similarity(other)` | 다른 `Embeddings` 또는 텐서에 대한 쌍별 코사인 유사도 |
| `verify(i, j, threshold=0.4)` | 행 `i`와 `j`가 일치할 때 `True` |

## 정체성

명명된 갤러리는 `embeddings`와 행 정렬된 일치 항목입니다. `Gallery`가 `embed` 예측에 전달될 때 생성됩니다. `name`는 항목이 일치 임계값 아래인 `None`의 목록이며, 임계값 이하의 가장 가까운 이름은 절대 추측되지 않습니다. `score`는 일치 점수 배열이고 `data`는 이를 쌍으로 연결합니다.

## 메시

파라메트릭 인간 신체 메시는 `boxes`에서 사람 박스에 맞춰 행 단위로 정렬되어 있습니다. 모든 것은 원본 이미지의 카메라 프레임에 있습니다. `transl`는 미터 단위이며, `+z`는 카메라에서 멀어지는 방향을 가리킵니다; `vertices`와 `joints3d`는 미터 단위이며 이미 `transl`를 포함하고 있습니다; `joints2d`는 네트워크가 본 크롭이 아닌 원본 이미지 캔버스의 픽셀 단위입니다. 어떤 필드도 세계 또는 중력 프레임을 포함하지 않습니다.

파라미터 레이아웃은 바디 모델마다 다르므로 형태에 대해서는 하드코딩되어 있지 않습니다. `body_model`는 파라미터화를 지정하며, 카운트는 텐서에서 다시 읽습니다: `num_vertices`, `num_joints`, `num_betas`, `has_vertices`. `params`는 파라미터 딕트(parameter dict)를 반환하고 `save_obj(path, index=0)`는 하나의 메시를 작성합니다. 필드는 `global_orient`, `body_pose`, `betas`, `transl`, `vertices`, `faces`, `joints3d`, `joints2d`, `conf`, `focal_length`, `extras`입니다.

`body_model="mhr"`의 회전은 축-각이 아니라 라디안 단위의 오일러 각이고, `body_pose`는 관절마다 하나의 삼중항이 아니라 평면의 관절별 매개변수 벡터이며, `betas`는 항등 블렌드셰이프 계수입니다. 스켈레톤 크기, 손 포즈, 얼굴 표정은 `extras`에 있습니다.

## 변환 및 선택

모든 페이로드는 `to(*args, **kwargs)`, `cpu()`, `cuda()` 및 `numpy()`를 가지고 있으며, `Results`에서 그 중 하나를 호출하면 모든 채워진 슬롯에 한 번에 적용됩니다.

<code-tabs name="convert" />

`result[idx]`는 행 정렬된 페이로드 전체에서 행을 선택합니다. `len(result)`는 박스가 없을 때 탐지 수 또는 포인트 수입니다. `result.update(...)`는 명명된 슬롯이 교체된 복사본을 반환합니다; 모든 슬롯과 `track_id` 및 `restore_scale`를 허용합니다.

## 요약 및 to_json

`summary(normalize=False, decimals=5, embeddings=False)`는 설정된 슬롯에 따라 탐지, 세그먼트, 포인트 또는 영역당 한 행씩의 일반 딕셔너리 목록을 반환합니다. `to_json(**kwargs)`는 자신의 인수를 `summary`에 전달하고 JSON 문자열을 반환합니다.

`plot()`는 표준 시각화에서 조밀한 정상 또는 엣지 결과를 렌더링하며, 다른 결과 유형에서는 증가합니다. 다른 작업에 대한 주석된 이미지는 `predict(save=True)`에서 가져옵니다.
