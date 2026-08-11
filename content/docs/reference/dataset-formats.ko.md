---
title: 데이터셋 형식
seo_title: 모든 작업을 위한 LibreYOLO 데이터셋 형식
description: '표준 작업별 데이터셋 파일 계약: YAML 키, 폴더 구조, 레이블 행, 마스크 및 맵 규칙, 각 파일을 읽는 로더'
lead: >-
  이 페이지는 라이브러리 자체의 docs/dataset_schema.md.의 데이터셋 파일 계약을 미러링합니다. 여기에는 각 표준 작업에서
  기대하는 YAML 키와 디스크 상의 레이아웃이 포함됩니다.
keywords:
  - libreyolo 데이터셋 형식
  - yolo 레이블 형식
  - data.yaml
  - 분할 마스크 데이터셋
  - coco 파노라마 형식
  - 깊이 데이터셋
  - 자세 kpt_shape
last_verified: 1.5.0
verification: >-
  v1.5.0에서 libreyolo 저장소의 docs/dataset_schema.md를 미러링하며, 로더 이름은
  libreyolo/data/.와 교차 확인됨
snippets:
  usage:
    - label: 하나의 객체 탐지 레이블 행 파싱
      language: python
      code: >
        from libreyolo.data import parse_yolo_label_line


        # class_id cx cy w h, [0, 1]로 정규화됨

        row = parse_yolo_label_line("0 0.5 0.5 0.25 0.5", 640, 480,
        num_classes=80)


        # (class_id, x1, y1, x2, y2, area) 픽셀 단위

        print(row)
source_hash: a8282c079624044d
---

## 공통 YAML

`detect`, `segment`, `pose` 및 `obb`에 적용됩니다.

| 키 | 필수 | 의미 |
|---|---|---|
| `path` | | 데이터셋 루트 |
| `train` | 학습용 | 학습 이미지 |
| `val` | 검증용 | 검증 이미지 |
| `test` | | 테스트 이미지 |
| `names` | 예 | 클래스 목록 또는 정수 키 매핑 |
| `nc` | | 클래스 수; 존재할 경우 `names`와 일치해야 함 |
| `download` | | 다운로드 지침; Python 스크립트는 명시적 옵트인 필요 |
| `annotations` | | detect, segment 및 obb용 네이티브 COCO JSON 파일로 분할 |

`train`, `val` 및 `test`는 이미지 디렉토리, 이미지 목록 `.txt` 파일 또는 이들의 목록일 수 있습니다. 레이블 경로는 하나의 치환을 따릅니다:

```text
images/.../image.jpg -> labels/.../image.txt
```

네이티브 COCO JSON 데이터셋의 경우, `annotations`는 분할을 그 JSON 파일에 매핑하며, 분할 경로는 이미지 루트를 제공합니다:

```yaml
path: dataset
train: images/train
val: images/val
annotations:
  train: annotations/train.json
  val: annotations/val.json
```

`names`가 존재하면, 네이티브 COCO JSON 카테고리 이름은 YAML 클래스 이름과 일치해야 하며, 그 이름들은 모델 레이블 ID를 정의합니다. `names`가 없으면, COCO 카테고리 ID는 정렬되어 `0..N-1`에 밀집하게 매핑됩니다.

데이터셋 YAML에는 `task` 키가 포함되어 있지 않습니다. 명시적인 모델 및 작업 선택이 우선합니다.

모든 텍스트 레이블 파일에 공통적인 규칙:

- 이미지당 하나의 `.txt` 레이블 파일;
- 레이블 파일이 없거나 비어 있으면 객체 없음;
- `class_id`는 `0..nc-1` 내의 정수;
- 좌표는 `[0, 1]`에서 유한한 정규화된 부동 소수점입니다;
- 좌표는 원본 이미지의 너비와 높이에 상대적입니다;
- 행에는 신뢰도나 추적 ID가 없습니다.

<code-tabs name="usage" />

## 감지

행마다 정확히 다섯 개의 필드가 있습니다:

```text
<class_id> <cx> <cy> <w> <h>
```

`cx cy w h`는 정규화된 축에 정렬된 상자이며, `w`와 `h`는 양수여야 합니다.

## 분할

다각형 행:

```text
<class_id> <x1> <y1> ... <xN> <yN>
```

`N`는 최소 3이어야 하며, `class_id` 이후의 좌표 수는 짝수여야 하고, 다각형은 퇴화하지 않아야 합니다. 다섯 필드의 감지 행도 허용되며 직사각형 세그먼트를 나타냅니다.

## 포즈

YAML은 필수인 `kpt_shape`를 추가하며, 이것은 `[K, 2]` 또는 `[K, 3]`이며, 선택적 `flip_idx`는 `0..K-1`의 정수 순열입니다.

```text
<class_id> <cx> <cy> <w> <h> <k1x> <k1y> [<k1v>] ... <kKx> <kKy> [<kKv>]
```

필드 수는 정확히 `5 + K * D`이며, 여기서 `D`는 두 번째 `kpt_shape` 값입니다. 키포인트 좌표는 정규화되어 있습니다. 가시성 `v`가 있는 경우 `0`, `1` 또는 `2`입니다.

## obb

정확히 아홉 개의 필드:

```text
<class_id> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4>
```

네 점은 `[0, 1]`에서 정규화된 이미지 좌표이며, 비퇴화 지향 직사각형을 형성합니다. 레이블 파일에는 각도가 저장되지 않습니다.

표준 파서는 기본적으로 엄격하며 범위를 벗어난 좌표를 거부합니다. 데이터셋 및 검증 수집은 그렇지 않으면 유효한 자르기 경계 레이블의 좌표를 `[0, 1]`로 클리핑할 수 있지만, 퇴화된 박스는 여전히 거부합니다. 파싱은 작업 인식적입니다: 아홉 개의 필드는 `obb` 모드에서만 `obb`를 의미하며, `segment` 모드에서는 네 점 다각형일 수 있습니다.

내부적으로, 정규화된 코너는 표준 `xywhr`로 변환되며, 라디안 단위의 각도는 박스 중심을 기준으로 너비 방향의 회전을 나타냅니다. 공개 결과는 OBB 감지를 `xywhr, conf, cls` 행으로 노출합니다.

네이티브 COCO JSON OBB 로딩은 주석을 다음 우선순위로 허용합니다: `obb`로서 8개의 픽셀 공간 코너; 라디안 단위 각도를 가진 `[cx, cy, w, h, angle]`로서 `obb`; 최소 면적 사각형으로 다시 맞춰진 COCO `segmentation` 폴리곤 또는 RLE; 그리고 축에 정렬되고 표준화된 것으로 읽는 COCO `bbox`.

코너 인식 OBB 증강이 존재할 때까지 모자이크 및 믹스업은 OBB 학습용으로 비활성화됩니다.

표준 행 파서는 `libreyolo.data.parse_yolo_obb_label_line`입니다.

## 의미적

각 이미지는 일반적으로 PNG 형식의 무손실 포맷으로, `.txt` 파일 대신 단일 채널 마스크와 쌍을 이룹니다:

```text
images/.../image.jpg -> <masks_dir>/.../image.png
```

마스크는 단일 채널이며, 팔레트 모드 PNG는 팔레트 인덱스로 읽힙니다. 각 픽셀 값은 `0..nc-1`의 클래스 ID를 나타내며, 픽셀 값 `255`는 무시됨을 의미하고 손실 및 지표에서 제외됩니다. 마스크 해상도는 이미지 해상도와 같아야 합니다.

두 개의 선택적 YAML 키가 공통 계약 위에 위치합니다. `masks_dir`는 각 이미지 경로에서 `images` 대신 치환되는 마스크 디렉토리 이름으로, 기본값은 `masks`입니다. `label_mapping`는 로드 시 마스크 픽셀 값에 적용되는 `{source_id: train_id}` 리맵으로, 매핑되지 않은 원본 값은 무시되고 학습 ID는 `0..nc-1` 범위 내에 있어야 합니다.

`masks_dir`가 생략되면, 마스크는 `segment` 폴리곤 레이블에서 로드 시 래스터화되며, `images`에서 `labels` 규칙을 통해 해결됩니다. 객체 클래스 뒤에 `background` 클래스가 추가되므로 `nc`는 하나 증가합니다.

정식 로더: `libreyolo.data.SemanticDataset`.

## 전경-배경(panoptic)

LibreYOLO는 COCO-panoptic 형식을 그대로 채택합니다(Kirillov et al., CVPR 2019). LibreYOLO 전용 panoptic 형식은 없습니다.

이미지 해상도의 각 이미지마다 하나의 RGB PNG가 있으며, 각 픽셀의 세그먼트 ID를 색상으로 인코딩합니다:

```text
segment_id = R + 256 * G + 256 * 256 * B
```

모든 픽셀은 정확히 하나의 세그먼트에 속하며 세그먼트는 겹치지 않습니다. 세그먼트 ID `0`, RGB 검정색,은 void로 간주됩니다: 평가지표에서 제외된 레이블 없는 픽셀입니다.

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

`annotations[].file_name`는 `panoptic_dir` 안의 세그먼트-ID PNG의 이름이며, `segments_info[].id`는 해당 PNG에서 값을 일치시킵니다. `iscrowd`는 그룹 영역을 표시합니다: 이들은 절대 false negative가 아니며, 예측이 대부분을 덮는 경우 false positive가 아닙니다.

Thing-versus-stuff는 카테고리별 속성입니다. `isthing`는 `categories`에 존재하며, `segments_info`에는 존재하지 않습니다.

COCO-panoptic `category_id` 값은 데이터셋의 원시 ID이며 일반적으로 연속적이지 않습니다. 모델은 연속적인 `0..nc-1`를 예측하므로, 원시 ID는 YAML `names`를 통해 카테고리 이름으로 재매핑됩니다. 이는 기본 COCO JSON detect 로더가 따르는 규칙과 동일합니다. `names`에 없는 JSON 카테고리는 조용히 무시되는 것이 아니라 오류입니다. 그렇지 않으면 영구적인 False Negative로 점수가 계산될 수 있기 때문입니다.

```yaml
path: coco
val: images/val2017
annotations:
  val: annotations/panoptic_val2017.json
panoptic_dir:
  val: annotations/panoptic_val2017
names: {0: person, 1: bicycle, 132: rug-merged}
```

`annotations`와 `panoptic_dir`는 단일 경로 또는 데이터 분할별 매핑을 받을 수 있습니다.

검증 보고서는 생성된 진짜 해상도에서 계산된 Panoptic Quality를 나타내며, 나타난 카테고리별로 평균을 내고 `PQ_things`와 `PQ_stuff`로 나눕니다. 매칭은 고유하며, 예측된 세그먼트와 동일 카테고리의 진짜 세그먼트는 IoU가 0.5 이상일 때 매칭됩니다.

표준 로더: `libreyolo.data.PanopticDataset`.

## 깊이

각 이미지에는 밀집 단일 채널 깊이 맵이 쌍으로 제공됩니다:

```text
images/.../image.jpg -> <depths_dir>/.../image.png
```

이 맵은 이미지 해상도의 단일 채널 PNG 또는 TIF, 또는 `.npy` 파일입니다. 값은 데이터셋 일관 단위로의 순수 깊이입니다. 0, 음수, NaN 및 무한 값은 유효하지 않은 픽셀을 나타내며 손실 및 메트릭에서 제외됩니다.

| 키 | 기본 | 의미 |
|---|---|---|
| `depths_dir` | `depths` | `images` 대신 사용된 깊이 디렉터리 |
| `depth_stem_suffix` | | 이미지 스템에 추가되는 접미사; 생략 시 동일한 스템과 `_depth` 접미사 모두 시도됨 |
| `depth_mask_suffix` | `_mask` | 유효성 마스크용 접미사; 마스크 값이 0 이하, NaN 및 무한인 경우 깊이 픽셀 무효 |
| `depth_scale` | `256.0` | 정수형 깊이 맵용 나눗셈기, 일반적인 16비트 PNG 규칙 |

부동 소수점 `.npy` 맵은 그대로 사용되며 `depth_scale`를 적용하지 않습니다.

표준 로더: `libreyolo.data.DepthDataset`.

## 가장자리

각 RGB 이미지는 동일한 접두사를 가진 단일 채널 무손실 맵 및 선택적 유효성 마스크와 쌍을 이룹니다:

```text
images/val/scene.jpg -> edges/val/scene.png
                     -> masks/val/scene.png
```

맵은 이미지 해상도의 단일 채널 PNG 또는 TIF이며, RGB 시각화가 아닙니다. 정수형 맵은 dtype 최대값으로 나눕니다; 부동 소수점 맵은 이미 유한하며 `[0, 1]`에 있어야 합니다. `0`는 비-가장자리를 의미하고 `1`는 가장자리를 의미합니다. 선택적 마스크 픽셀은 0이 아닐 때 유효합니다. 크기 조정은 대상 및 마스크에 대해 최근접 이웃 보간법을 사용하며, 패딩된 픽셀은 유효하지 않으며 검증에 기여하지 않습니다.

| 키 | 기본값 | 의미 |
|---|---|---|
| `edges_dir` | `edges` | `images` 대신 가장자리 맵 디렉토리 |
| `edge_stem_suffix` | | 이미지 스템에 접미사 추가 |
| `edge_extension` | `.png` | 무손실 대상 확장자 |
| `edge_invert` | | 소스 맵이 흰색 위에 검은색 가장자리를 저장할 때 true 설정 |
| `masks_dir` | `masks` | 선택적 유효성 마스크 디렉토리 |

```yaml
path: edge-dataset
train: images/train
val: images/val
edges_dir: edges
masks_dir: masks
nc: 1
names: {0: edge}
```

Validation은 4방향 그래디언트 비최대 억제와 함께 연속 예측을 얇게 하고, 구성 가능한 임계값 범위에 대해 ODS 및 OIS F-측정값을 보고합니다. 예측 픽셀과 실제 픽셀은 `edge_max_dist * image_diagonal` 내에서 일대일로 매치되며, 기본 정규화 허용오차는 `0.0075`입니다.

표준 로더: `libreyolo.data.EdgeDataset`. 로더는 형식 전용이며, 벤치마크 데이터를 다운로드하거나 재배포하지 않습니다.

## 정상

각 이미지에는 동일한 이름의 3채널 16비트 PNG가 쌍으로 있으며, 선택적으로 동일한 이름의 유효성 마스크가 포함될 수 있습니다:

```text
images/val/room.jpg -> normals/val/room.png
                    -> masks/val/room.png
```

PNG는 정확히 3채널 `uint16`이며, 채널은 RGB로 저장되고 이미지 해상도로 제공됩니다. `n = png / 65535 * 2 - 1`로 디코딩한 후 각 벡터를 다시 정규화하세요. 디코딩된 벡터는 OpenCV 카메라 좌표계를 사용하며, `+x`는 오른쪽, `+y`는 아래 방향, `+z`는 장면 안쪽이며 카메라를 향합니다. 선택적 마스크는 단일 채널 PNG로, 0이 아닌 값이 유효함을 의미합니다; 마스크가 없으면 모든 유한하고 0이 아닌 디코딩 벡터가 유효합니다. 유효하지 않거나 패딩된 대상 픽셀은 내부적으로 `(0, 0, 0)`로 표시됩니다. 크기 조정은 세 가지 구성 요소를 선형 보간한 후 다시 정규화하며, 유효성 마스크는 최근접 이웃 보간을 사용하고, 수평 뒤집기는 x 구성 요소를 반전시킵니다.

| 키 | 기본값 | 의미 |
|---|---|---|
| `normals_dir` | `normals` | Normal-map 디렉토리가 `images` 대신 사용됨 |
| `masks_dir` | `masks` | 선택적 유효성 마스크 디렉토리 |

검증 보고서는 평균 및 중앙 각오차를 도 단위로 나타내며, 11.25도, 22.5도, 30도 내의 유효 픽셀 비율을 표시합니다.

표준 로더: `libreyolo.data.NormalDataset`.

## 복원

각 열화된 입력 이미지는 깨끗한 RGB 대상과 쌍을 이룹니다:

```text
inputs/.../image.jpg -> targets/.../image.jpg
```

입력과 대상은 RGB 호환 이미지 파일이며 해상도가 정확히 일치해야 합니다. 검증은 원본 해상도를 유지하고 배치를 쌓기 위해 필요한 만큼만 패딩하며, 지표는 원본 이미지 캔버스에서 계산됩니다. 학습은 입력과 대상 쌍에 결합된 크롭 및 수평 뒤집기를 적용합니다.

| 키 | 기본값 | 의미 |
|---|---|---|
| `input_dir` | `inputs` | 분할 경로에서 사용된 저하된 입력 디렉토리 |
| `target_dir` | `targets` | `input_dir` 대신 사용된 클린 대상 디렉토리 |
| `target_stem_suffix` | | 대상 조회 전에 입력 스템에 추가된 접미사 |
| `target_stem_suffixes` | | `target_stem_suffix`의 목록 형태 |
| `degradation` | | `deblur` 또는 `denoise`와 같은 메타데이터 레이블 |
| `dataset` | | 데이터셋 또는 출처 레이블 |

클래스와 같은 YAML 필드는 스키마 자리표시자입니다: `nc: 1` 및 `names: {0: image}`를 사용하십시오. 복원 모델은 감지를 보여주지 않고 `Results.restored`를 노출합니다.

표준 로더: `libreyolo.data.RestoreDataset`.

## 매트

각 RGB 이미지에는 동일한 스템을 공유하는 단일 채널 그라운드-트루스 매트가 짝지어져 있으며, 0은 배경, 255는 전경을 나타냅니다:

```text
images/subject.jpg -> mattes/subject.png
```

두 가지 레이아웃이 허용됩니다. `images/`와 매트 디렉토리를 포함하는 디렉토리 루트, `mattes/`, `matte/`, `gt/`, `masks/`, `mask/` 및 `alpha/` 중 자동으로 감지되어 `data=`로 전달됩니다. 또는 `path`와 각 스플릿 `val_images` 및 `val_mattes`, 선택적으로 `train_images` 및 `train_mattes`를 포함하는 YAML을 사용할 수 있으며, 각각 `path`를 기준으로 상대 경로나 절대 경로를 가집니다.

매트는 그레이스케일이며 `[0, 1]`에서 불투명도로 읽히며, 형상이 다를 경우 예측 캔버스로 이중 선형 보간법으로 크기 조정됩니다. 메트릭은 원본 이미지 캔버스에서 MAE와 S-측정(Fan et al., ICCV 2017)을 사용하며, S-측정은 최적 체크포인트 적합도로 사용됩니다.

클래스와 유사한 YAML 필드는 스키마 자리 표시자입니다: `nc: 1`와 `names: {0: matte}`를 사용하십시오. Matte 모델은 `Results.matte`를 노출합니다.

이 버전에서는 검증이 추론 전용입니다. 표준 쌍 해석기: `libreyolo.data.matte_dataset.resolve_matte_pairs`.

## OCR

레이블은 분할당 하나의 JSONL 파일, 이미지당 하나의 JSON 객체입니다:

```text
images/val/receipt.jpg -> labels/val.jsonl
```

```json
{"image": "receipt.jpg", "regions": [{"polygon": [[10, 12], [118, 14], [117, 40], [9, 38]], "text": "TOTAL 12.50"}]}
```

`polygon`는 절대 픽셀 좌표의 4점 쿼드로, 순서는 좌상, 우상, 우하, 좌하입니다. 읽을 수 없는 텍스트가 있는 영역은 `"text": "###"`를 사용하며, ICDAR ‘관심 없음(do-not-care)’ 규칙을 따릅니다: 인식 스코어링에서 제외되고, 해당 영역과 겹치는 예측은 감지 매칭에서 패널티를 받지 않고 무시됩니다.

메트릭은 IoU 0.5 이상에서 1:1 폴리곤 매칭을 사용한 detection hmean, IoU가 0.5 이상이고 NFKC 정규화 및 공백 제거 후 정확한 전사를 요구하는 end-to-end F1(대소문자 구분), 그리고 매칭된 쌍에서 1-NED입니다. 가장 좋은 체크포인트 적합도는 end-to-end F1입니다.

두 가지 레이아웃이 허용됩니다: `data=`로 전달된 `images/<split>/`와 `labels/<split>.jsonl`를 포함하는 디렉토리 루트, 또는 선택적으로 `images`와 `labels` 디렉토리 이름을 포함하는 `path` YAML.

클래스와 유사한 YAML 필드는 스키마 자리 표시자입니다: `nc: 1`와 `names: {0: text}`를 사용하십시오. OCR 모델은 `Results.ocr`를 노출합니다.

이 버전의 검증은 추론 전용입니다. 표준 샘플 해석기: `libreyolo.data.ocr_dataset.resolve_ocr_samples`.

## 분류

레이블 파일이 아닌 ImageFolder 스타일 디렉토리 트리:

```text
dataset_root/
  train/
    class_a/*.jpg
    class_b/*.jpg
  val/
    class_a/*.jpg
    class_b/*.jpg
```

`train/`는 학습에 필요하며 정렬된 폴더 이름에 따른 클래스-인덱스 매핑을 정의합니다. `val/`는 검증에 필요합니다. `test/`는 존재할 수 있지만 기본 학습 및 검증 명령어에서는 사용되지 않습니다. 학습이 아닌 분할 데이터셋은 예상 학습 또는 체크포인트 클래스 세트와 동일한 클래스 폴더 이름을 포함해야 합니다. 지원되는 이미지 확장자는 `libreyolo.data.classify_dataset.IMAGE_EXTENSIONS`에 정의되어 있습니다.

## 시선(gaze) 및 포인트(point)

`gaze`에 대해 학습이나 검증 데이터셋-파일 계약은 구현되어 있지 않습니다.

`point`는 데이터셋-레이블 스키마보다는 모델 출력 작업입니다. 포인트 패밀리는 기존 레이블을 내부적으로 조정할 수 있으며, 예를 들어 박스 행에서 객체 중심을 유도할 수 있지만, 포인트 전용 텍스트 레이블 형식은 정의되어 있지 않습니다.
