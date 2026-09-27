---
title: 데이터셋 형식
seo_title: 모든 작업을 위한 LibreYOLO 데이터셋 형식
description: '정규 작업별 데이터셋 파일 계약: YAML 키, 폴더 구성, 레이블 행, 마스크 및 맵 규칙, 그리고 각각을 읽는 로더.'
lead: >-
  이 페이지는 라이브러리 자체 docs/dataset_schema.md.에 있는 데이터셋-파일 계약을 반영합니다. 여기에는 각 표준 작업이
  기대하는 YAML 키와 디스크 상의 레이아웃이 포함됩니다.
keywords:
  - libreyolo 데이터셋 형식
  - yolo 레이블 형식
  - data.yaml
  - 분할 마스크 데이터셋
  - 코코 파노프틱 형식
  - 깊이 데이터셋
  - 포즈 kpt_모양
last_verified: 1.5.0
verification: >-
  v1.5.0에서 libreyolo 저장소의 docs/dataset_schema.md를 미러링하며, 로더 이름은
  libreyolo/data/.와 교차 검증됨
snippets:
  usage:
    - label: 하나의 탐지 레이블 행을 파싱하다
      language: python
      code: >
        from libreyolo.data import parse_yolo_label_line


        # 클래스_id cx cy w h, [0, 1]로 정규화됨

        row = parse_yolo_label_line("0 0.5 0.5 0.25 0.5", 640, 480,
        num_classes=80)


        # (class_id, x1, y1, x2, y2, area) 픽셀 단위로

        print(row)
source_hash: a8282c079624044d
---

## 일반 YAML

`detect`, `segment`, `pose` 및 `obb`에 적용됩니다.

| 열쇠 | 필수 | 의미 |
|---|---|---|
| `path` | | 데이터셋 루트 |
| `train` | 학습을 위해 | 학습 이미지 |
| `val` | 검증을 위해 | 검증 이미지 |
| `test` | | 테스트 이미지 |
| `names` | 예 | 클래스 목록 또는 정수 키 매핑 |
| `nc` | | 클래스 수; 존재할 경우 `names`와 일치해야 함 |
| `download` | | 다운로드 지침; Python 스크립트는 명시적인 옵트인이 필요합니다 |
| `annotations` | | 탐지, 분할 및 OBB를 위해 네이티브 COCO JSON 파일로 분할 |

`train`, `val` 및 `test`는 이미지 디렉토리, 이미지 목록 `.txt` 파일 또는 그 목록일 수 있습니다. 레이블 경로는 하나의 치환을 따릅니다:

```text
images/.../image.jpg -> labels/.../image.txt
```

원시 COCO JSON 데이터셋의 경우, `annotations`는 스플릿을 해당 JSON 파일에 매핑하고 스플릿 경로는 이미지 루트를 제공합니다:

```yaml
path: dataset
train: images/train
val: images/val
annotations:
  train: annotations/train.json
  val: annotations/val.json
```

`names`가 있는 경우, 원래 COCO JSON 카테고리 이름은 YAML 클래스 이름과 일치해야 하며, 이러한 이름이 모델 레이블 ID를 정의합니다. `names`가 없는 경우, COCO 카테고리 ID는 정렬되어 `0..N-1`에 밀집하게 매핑됩니다.

데이터세트 YAML에는 `task` 키가 없습니다. 명시적인 모델 및 작업 선택이 우선합니다.

모든 텍스트 레이블 파일에 공통되는 규칙:

- 이미지당 하나의 `.txt` 레이블 파일;
- 레이블 파일이 없거나 비어 있으면 객체가 없습니다;
- `class_id`는 `0..nc-1`에 있는 정수입니다;
- 좌표는 `[0, 1]`에서 유한한 정규화된 부동소수점입니다.
- 좌표는 원본 이미지의 너비와 높이를 기준으로 합니다;
- 행은 신뢰도나 추적 ID를 가지지 않습니다.

<code-tabs name="usage" />

## 탐지

한 줄당 정확히 다섯 개의 필드:

```text
<class_id> <cx> <cy> <w> <h>
```

`cx cy w h`는 정규화된 축 정렬 바운딩 박스이며, `w`와 `h`는 양수여야 합니다.

## 세그먼트

다각형 행:

```text
<class_id> <x1> <y1> ... <xN> <yN>
```

`N`는 최소 3이어야 하며, `class_id` 이후의 좌표 수는 짝수여야 하고, 다각형은 퇴화하지 않아야 합니다. 또한 다섯 칸 탐지 행도 허용되며 직사각형 구간을 나타냅니다.

## 자세

YAML은 필수인 `kpt_shape`를 추가하며, 이는 `[K, 2]` 또는 `[K, 3]`이고, 선택 사항인 `flip_idx`도 추가되며, 이는 `0..K-1`의 정수 순열입니다.

```text
<class_id> <cx> <cy> <w> <h> <k1x> <k1y> [<k1v>] ... <kKx> <kKy> [<kKv>]
```

필드 수는 정확히 `5 + K * D`이며, 여기서 `D`는 두 번째 `kpt_shape` 값입니다. 키포인트 좌표는 정규화되어 있습니다. 가시성 `v`는 존재할 경우 `0`, `1` 또는 `2`입니다.

## obb

정확히 아홉 개의 필드:

```text
<class_id> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4>
```

네 점은 `[0, 1]`에서 정규화된 이미지 좌표이며, 비퇴화된 방향이 있는 사각형을 형성합니다. 레이블 파일에는 각도가 저장되지 않습니다.

정규 파서는 기본적으로 엄격하며 범위를 벗어난 좌표를 거부합니다. 데이터셋 및 검증 수집은 그렇지 않으면 유효한 크롭 경계 레이블의 좌표를 `[0, 1]`로 잘라낼 수 있으며, 그럼에도 불구하고 퇴화된 상자는 여전히 거부됩니다. 파싱은 작업 인식적입니다: 아홉 개 필드는 `obb` 모드에서만 `obb`를 의미하며, `segment` 모드에서는 네 점 다각형일 수 있습니다.

내부적으로, 정규화된 코너는 표준 `xywhr`로 변환되며, 라디안 단위의 각도는 박스 중심을 기준으로 한 너비 측의 회전을 나타냅니다. 공개된 결과는 `xywhr, conf, cls` 행으로 OBB 탐지를 보여줍니다.

네이티브 COCO JSON OBB 로딩은 다음 우선 순위로 주석을 허용합니다: 8개의 픽셀 공간 코너로서 `obb`; 라디안 단위의 각도를 포함한 `[cx, cy, w, h, angle]`로서 `obb`; 최소 면적 직사각형에 맞춰 조정된 COCO `segmentation` 폴리곤 또는 RLE; 그리고 COCO `bbox`로, 축 정렬되고 정규화된 것으로 읽습니다.

코너를 인식하는 OBB 증강이 존재할 때까지 OBB 학습에서는 모자이크와 믹스업이 비활성화됩니다.

표준 행 파서는 `libreyolo.data.parse_yolo_obb_label_line`입니다.

## 의미론의

각 이미지는 일반적으로 `.txt` 파일 대신 손실 없는 형식, 보통 PNG로 된 단일 채널 마스크와 짝을 이룹니다:

```text
images/.../image.jpg -> <masks_dir>/.../image.png
```

마스크는 단일 채널이며, 팔레트 모드 PNG는 팔레트 인덱스로 읽힙니다. 각 픽셀 값은 `0..nc-1`의 클래스 ID이며, 픽셀 값 `255`는 무시를 의미하고 손실 및 측정치에서 제외됩니다. 마스크 해상도는 이미지 해상도와 동일해야 합니다.

두 개의 선택적 YAML 키가 공통 계약 상단에 위치합니다. `masks_dir`는 각 이미지 경로에서 `images` 대신 사용되는 마스크 디렉토리 이름이며, 기본값은 `masks`입니다. `label_mapping`는 불러오기 시 마스크 픽셀 값에 적용되는 `{source_id: train_id}` 재매핑으로, 매핑되지 않은 소스 값은 무시 처리되며, 학습 ID는 `0..nc-1` 범위 내에 있어야 합니다.

`masks_dir`가 생략되면, 마스크는 `images`에서 `labels` 규칙을 통해 해결된 `segment` 폴리곤 레이블에서 로드 시 래스터화되며, 객체 클래스 뒤에 `background` 클래스가 추가되어 `nc`가 하나 증가합니다.

정식 로더: `libreyolo.data.SemanticDataset`.

## 파노프틱

LibreYOLO는 COCO-파노프틱 형식을 문자 그대로 채택합니다(Kirillov 등, CVPR 2019). LibreYOLO 전용 파노프틱 형식은 없습니다.

각 이미지 해상도에서 하나의 RGB PNG는 각 픽셀의 세그먼트 ID를 색상으로 인코딩합니다:

```text
segment_id = R + 256 * G + 256 * 256 * B
```

모든 픽셀은 정확히 하나의 세그먼트에 속하며 세그먼트는 결코 겹치지 않습니다. 세그먼트 ID `0`, RGB 검정,은 공백입니다: 메트릭에서 제외된 레이블 없는 픽셀.

```json
{
  "images":      [{"id": 139, "file_name": "000000000139.jpg"}],
  "annotations": [{"image_id": 139, "file_name": "000000000139.png",
                   "segments_info": [
                     {"id": 3226956, "category_id": 1, "area": 2840,
                      "bbox": [413, 158, 53, 138], "iscrowd": 0}]}],
  "categories":  [{"id": 1, "name": "person", "isthing": 1, "supercategory": "person"}]
}
```

`annotations[].file_name`는 `panoptic_dir` 안의 세그먼트-ID PNG의 이름을 지정하고, `segments_info[].id`는 해당 PNG의 값과 일치합니다. `iscrowd`는 그룹 영역을 표시합니다: 이들은 결코 거짓 부정이 아니며, 하나를 대부분 포함하는 예측은 거짓 긍정이 아닙니다.

사물 대 물질은 범주별 특성입니다. `isthing`는 `categories`에 살고, `segments_info`에는 절대 살지 않습니다.

COCO-panoptic `category_id` 값은 데이터셋의 원시 ID이며 일반적으로 연속적이지 않습니다. 모델은 연속적인 `0..nc-1`를 예측하므로, 원시 ID는 YAML `names`를 통해 카테고리 이름으로 재매핑됩니다. 이는 기존 COCO JSON 탐지 로더가 따르는 규칙과 동일합니다. `names`에 없는 JSON 카테고리는 무시되는 것이 아니라 오류로 간주되며, 그렇지 않으면 영구적인 false negative로 점수가 매겨지게 됩니다.

```yaml
path: coco
val: images/val2017
annotations:
  val: annotations/panoptic_val2017.json
panoptic_dir:
  val: annotations/panoptic_val2017
names: {0: person, 1: bicycle, 132: rug-merged}
```

`annotations`와 `panoptic_dir`는 단일 경로 또는 분할별 매핑을 허용합니다.

검증은 파노프틱 품질을 보고하며, 이는 실제 정답 해상도에서 계산되고 나타나는 카테고리별로 평균내어지며, 이후 `PQ_things`와 `PQ_stuff`로 나뉩니다. 매칭은 고유하며, IoU가 0.5 이상일 때 동일한 카테고리의 예측 세그먼트와 실제 정답 세그먼트가 매칭됩니다.

정식 로더: `libreyolo.data.PanopticDataset`.

## 깊이

각 이미지는 밀집된 단일 채널 깊이 맵과 쌍을 이룹니다:

```text
images/.../image.jpg -> <depths_dir>/.../image.png
```

지도는 이미지 해상도의 단일 채널 PNG 또는 TIF 파일이거나 `.npy` 파일입니다. 값은 데이터셋 일관 단위의 일반 깊이입니다. 0, 음수, NaN 및 무한 값은 유효하지 않은 픽셀을 표시하며 손실 및 지표에서 제외됩니다.

| 열쇠 | 기본값 | 의미 |
|---|---|---|
| `depths_dir` | `depths` | 깊이 디렉토리가 `images`로 대체되었습니다 |
| `depth_stem_suffix` | | 이미지 줄기 뒤에 접미사가 붙습니다. 생략하면 같은 줄기와 `_depth` 접미사가 모두 시도됩니다. |
| `depth_mask_suffix` | `_mask` | 유효성 마스크의 접미사; 0 이하, NaN 및 무한대 값은 깊이 픽셀을 무효화합니다 |
| `depth_scale` | `256.0` | 정수형 깊이 맵의 나눗셈 값, 일반적인 16비트 PNG 규칙 |

float `.npy` 지도는 있는 그대로 사용되며 `depth_scale`는 적용되지 않습니다.

정식 로더: `libreyolo.data.DepthDataset`.

## 모서리

각 RGB 이미지는 동일한 줄기(single-stem)를 가진 단일 채널 무손실 맵과 선택적 유효성 마스크와 쌍을 이룹니다:

```text
images/val/scene.jpg -> edges/val/scene.png
                     -> masks/val/scene.png
```

맵은 이미지 해상도에서 RGB 시각화가 아닌 단일 채널 PNG 또는 TIF입니다. 정수형 맵은 해당 dtype의 최대값으로 나누고, 부동 소수점 맵은 이미 유한해야 하며 `[0, 1]`에 있어야 합니다. `0`는 비엣지를 의미하고 `1`는 엣지를 의미합니다. 선택적 마스크 픽셀은 0이 아닐 때 유효합니다. 리사이징은 타깃과 마스크에 대해 최근접 이웃 보간법을 사용하며, 패딩된 픽셀은 유효하지 않고 검증에 기여하지 않습니다.

| 열쇠 | 기본값 | 의미 |
|---|---|---|
| `edges_dir` | `edges` | `images` 대신에 엣지 맵 디렉토리 사용 |
| `edge_stem_suffix` | | 이미지 어근에 붙는 접미사 |
| `edge_extension` | `.png` | 무손실 대상 확장 |
| `edge_invert` | | 소스 맵이 흰색 위에 검은 가장자리를 저장할 때 true로 설정 |
| `masks_dir` | `masks` | 선택적 유효성 마스크 디렉터리 |

```yaml
path: edge-dataset
train: images/train
val: images/val
edges_dir: edges
masks_dir: masks
nc: 1
names: {0: edge}
```

검증은 네 방향 그래디언트 비최대 억제와 함께 연속적인 예측을 얇게 처리하고 구성 가능한 임계값 스윕에 대해 ODS 및 OIS F-측정값을 보고합니다. 예측된 픽셀과 실제 픽셀은 `edge_max_dist * image_diagonal` 내에서 일대일로 매칭되며, 기본 정규화 허용 오차는 `0.0075`입니다.

정식 로더: `libreyolo.data.EdgeDataset`. 이 로더는 형식 전용이며, 벤치마크 데이터를 다운로드하거나 재배포하지 않습니다.

## 보통

각 이미지는 동일한 이름의 3채널 16비트 PNG와 페어링되며, 선택적으로 동일한 이름의 유효성 마스크를 포함할 수 있습니다:

```text
images/val/room.jpg -> normals/val/room.png
                    -> masks/val/room.png
```

PNG는 이미지 해상도에서 채널을 RGB로 저장한 정확히 3채널 `uint16`입니다. `n = png / 65535 * 2 - 1`로 디코딩한 다음 각 벡터를 재정규화합니다. 디코딩된 벡터는 OpenCV 카메라 프레임을 사용하며, `+x`는 오른쪽, `+y`는 아래쪽, `+z`는 장면 안쪽을 향하며 카메라를 바라봅니다. 선택적 마스크는 0이 아닌 값이 유효함을 의미하는 단일 채널 PNG입니다. 마스크가 없으면 모든 유한하고 0이 아닌 디코딩된 벡터가 유효합니다. 잘못되었거나 패딩된 대상 픽셀은 내부적으로 `(0, 0, 0)`로 나타냅니다. 크기 조정은 세 가지 구성 요소를 양선형으로 보간한 후 재정규화하며, 유효성 마스크는 최근접 이웃 보간 방식을 사용하고, 수평 뒤집기는 x 구성 요소도 반전시킵니다.

| 열쇠 | 기본값 | 의미 |
|---|---|---|
| `normals_dir` | `normals` | Normal-map 디렉토리가 `images`로 대체되었습니다 |
| `masks_dir` | `masks` | 선택적 유효성 마스크 디렉터리 |

검증 보고서는 도 단위의 평균 및 중앙값 각도 오차와 11.25도, 22.5도 및 30도 내 유효 픽셀의 비율을 나타냅니다.

정식 로더: `libreyolo.data.NormalDataset`.

## 복원

각 열화된 입력 이미지는 깨끗한 RGB 타겟과 짝을 이룹니다:

```text
inputs/.../image.jpg -> targets/.../image.jpg
```

입력과 대상은 RGB 호환 이미지 파일이어야 하며 해상도가 정확히 일치해야 합니다. 검증 시에는 원본 해상도를 유지하며 배치를 쌓기 위해 필요한 만큼만 패딩을 적용하고, 지표는 원본 이미지 캔버스에서 계산됩니다. 학습 시에는 입력과 대상 쌍에 대해 함께 자르기와 수평 뒤집기를 적용합니다.

| 열쇠 | 기본값 | 의미 |
|---|---|---|
| `input_dir` | `inputs` | 분할 경로에서 사용된 저하된 입력 디렉터리 |
| `target_dir` | `targets` | `input_dir` 대신 클린-타겟 디렉토리 사용 |
| `target_stem_suffix` | | 대상 조회 전에 입력 어간에 붙는 접미사 |
| `target_stem_suffixes` | | `target_stem_suffix`의 목록 형태 |
| `degradation` | | `deblur` 또는 `denoise`와 같은 메타데이터 레이블 |
| `dataset` | | 데이터셋 또는 출처 레이블 |

클래스와 유사한 YAML 필드는 스키마 자리 표시자입니다: `nc: 1` 및 `names: {0: image}`를 사용하십시오. 복원 모델은 `Results.restored`를 노출하며, 검출은 노출하지 않습니다.

정식 로더: `libreyolo.data.RestoreDataset`.

## 매트

각 RGB 이미지는 동일한 스템을 공유하는 단일 채널 실제 매트와 짝을 이루며, 여기서 0은 배경이고 255는 전경입니다:

```text
images/subject.jpg -> mattes/subject.png
```

두 가지 레이아웃이 허용됩니다. `images/`를 포함하는 디렉터리 루트와, `mattes/`, `matte/`, `gt/`, `masks/`, `mask/` 및 `alpha/` 중 자동 탐지되는 매트 디렉터리가 `data=`로 전달됩니다. 또는 `path`가 포함된 YAML과 각 분할별 `val_images` 및 `val_mattes`, 선택적으로 `train_images` 및 `train_mattes`를 포함하며, 각 항목은 `path`를 기준으로 하거나 절대 경로입니다.

매트는 그레이스케일이며 `[0, 1]`에서 불투명도로 읽히며, 형태가 다를 경우 예측 캔버스로 양선형 보간법으로 크기가 조정됩니다. 지표는 원본 이미지 캔버스에서 MAE와 S-메저(Fan et al., ICCV 2017)이며, S-메저가 최적 체크포인트 적합도로 사용됩니다.

클래스 같은 YAML 필드는 스키마 자리 표시자입니다: `nc: 1`와 `names: {0: matte}`를 사용하십시오. 매트 모델은 `Results.matte`를 노출합니다.

검증은 이 버전에서 추론 전용입니다. 정준 쌍 해결기: `libreyolo.data.matte_dataset.resolve_matte_pairs`.

## 광학 문자 인식

레이블은 분할당 하나의 JSONL 파일, 이미지당 하나의 JSON 객체입니다:

```text
images/val/receipt.jpg -> labels/val.jsonl
```

```json
{"image": "receipt.jpg", "regions": [{"polygon": [[10, 12], [118, 14], [117, 40], [9, 38]], "text": "TOTAL 12.50"}]}
```

`polygon`은 절대 픽셀 좌표의 4점 사각형으로, 좌측 상단, 우측 상단, 우측 하단, 좌측 하단 순서입니다. 판독 불가능한 텍스트가 있는 영역에는 `"text": "###"`을 사용하며, 이는 ICDAR에서 정의한 무시(do-not-care) 규칙입니다: 해당 영역은 인식 평가에서 제외되며, 이 영역과 겹치는 예측은 검출 매칭 시 페널티를 받지 않고 무시됩니다.

측정 지표는 IoU 0.5 이상에서의 1:1 다각형 매칭을 통한 검출 hmean, IoU 0.5 이상 및 NFKC 정규화와 공백 제거 후 정확한 전사와 대소문자 구분을 모두 요구하는 end-to-end F1, 그리고 매칭된 쌍에서의 1-NED입니다. 최적 체크포인트 적합도는 end-to-end F1입니다.

두 가지 레이아웃이 허용됩니다: `images/<split>/`와 `labels/<split>.jsonl`를 포함하는 디렉토리 루트로, `data=`로 전달되거나, `path`와 선택적으로 `images` 및 `labels` 디렉토리 이름을 포함하는 YAML 파일입니다.

클래스 같은 YAML 필드는 스키마 자리 표시자입니다: `nc: 1`와 `names: {0: text}`를 사용하십시오. OCR 모델은 `Results.ocr`를 노출합니다.

검증은 이 버전에서 추론 전용입니다. 표준 샘플 해결사: `libreyolo.data.ocr_dataset.resolve_ocr_samples`.

## 분류

레이블 파일이 아닌 ImageFolder 스타일의 디렉토리 트리:

```text
dataset_root/
  train/
    class_a/*.jpg
    class_b/*.jpg
  val/
    class_a/*.jpg
    class_b/*.jpg
```

`train/`는 학습에 필요하며 폴더 이름을 정렬하여 클래스-인덱스 매핑을 정의합니다. `val/`는 검증에 필요합니다. `test/`가 있을 수 있지만 기본 학습 및 검증 명령에서는 사용되지 않습니다. 학습 외의 분할에는 예상되는 학습 또는 체크포인트 클래스 집합과 동일한 클래스 폴더 이름이 포함되어야 합니다. 지원되는 이미지 확장자는 `libreyolo.data.classify_dataset.IMAGE_EXTENSIONS`에 정의되어 있습니다.

## 응시하고 가리키다

`gaze`에는 학습 또는 검증 데이터셋 파일 계약이 구현되지 않았습니다.

`point`는 데이터셋-레이블 스키마라기보다는 모델 출력 작업입니다. 포인트 계열은 내부적으로 기존 레이블을 조정할 수 있는데, 예를 들어 박스 행에서 객체 중심을 도출하는 방식이 있을 수 있지만, 포인트 전용 텍스트 레이블 형식은 정의되어 있지 않습니다.
