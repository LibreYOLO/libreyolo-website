---
title: 체크포인트 스키마
seo_title: LibreYOLO 체크포인트 메타데이터 스키마 v1.0
description: >-
  모든 LibreYOLO .pt 체크포인트가 포함하는 메타데이터: 필수 키, 작업별 추가 항목, 내보내기 런타임 키, 양자화 매니페스트 및
  학습 필드.
lead: >-
  LibreYOLO .pt 파일은 torch.save로 저장된 평면 딕셔너리입니다. model 키는 상태 딕셔너리를 보유하며, 나머지 최상위
  키들은 파일 이름 파싱이나 상태 딕셔너리 확인 없이 체크포인트를 식별하는 메타데이터입니다.
keywords:
  - libreyolo 체크포인트 스키마
  - 스키마_버전 1.0
  - 모델_계열
  - libreyolo 체크포인트 메타데이터
  - 양적 현시
  - wrap_libreyolo_체크포인트
last_verified: 1.5.0
verification: >-
  v1.5.0에서 libreyolo 저장소의 docs/checkpoint_schema.md를 반영하며,
  libreyolo/utils/serialization.py 및 BaseModel.save와 교차 확인됨.
snippets:
  usage:
    - label: 체크포인트에서 메타데이터를 읽으십시오
      language: python
      code: >
        from libreyolo import LibreYOLO

        from libreyolo.utils.serialization import unwrap_libreyolo_checkpoint

        import torch


        # 체크포인트를 다운로드한 후 로컬 경로가 존재하도록 다시 저장하십시오.

        LibreYOLO("LibreYOLO9t.pt").save("roundtrip.pt")


        loaded = torch.load("roundtrip.pt", map_location="cpu",
        weights_only=False)

        state_dict, metadata = unwrap_libreyolo_checkpoint(loaded)


        print(metadata["schema_version"], metadata["model_family"])

        print(metadata["size"], metadata["task"], metadata["nc"],
        metadata["imgsz"])

        print(len(state_dict), "tensors")
source_hash: ce760f1bed97bfd0
---

## 스키마 v1.0

모든 공식 LibreYOLO `.pt` 체크포인트에는 다음이 포함되어 있습니다:

```python
{
    "model": state_dict,
    "schema_version": "1.0",
    "libreyolo_version": "0.x.y",
    "model_family": "yolo9",
    "size": "t",
    "task": "detect",
    "nc": 80,
    "names": {0: "cat", 1: "dog"},
    "imgsz": 640,
}
```

| 열쇠 | 타입 | 의미 |
|---|---|---|
| `model` | 상태 사전 | 모델 가중치 |
| `schema_version` | 문자열 | 메타데이터 계약 버전; v1.0은 문자열 `"1.0"`를 사용합니다 |
| `libreyolo_version` | 문자열 | 체크포인트를 생성한 버전 |
| `model_family` | 문자열 | 등록된 계열, 예를 들어 `yolo9`, `rfdetr`, `dfine`, `ec` |
| `size` | 문자열 | 계열 내 변형, 예를 들어 `t`, `s`, `r18`, `atto` |
| `task` | 문자열 | 정식 작업 이름 |
| `nc` | 정수 | 양성 클래스 수 |
| `names` | 사전 | `dict[int, str]`를 `0..nc-1`의 키와 함께 |
| `imgsz` | 정수 | 양의 제곱 입력 해상도, 또는 직사각형 계약에 대한 기존 스칼라 |

`task`는 `detect`, `segment`, `semantic`, `panoptic`, `pose`, `classify`, `gaze`, `obb`, `point`, `depth`, `edge`, `normal`, `restore`, `matte`, `ocr`, `embed` 또는 `mesh` 중 하나입니다.

공식 검문소는 모든 `names` 키를 기록합니다. 리더기는 레거시 희소 매핑을 위해 누락된 키를 `class_i` 레이블로 채울 수 있지만, 범위를 벗어난 키는 유효하지 않습니다.

직사각형 체크포인트는 레거시 리더를 위해 스칼라 `imgsz`를 유지하며, `max(imgsz_h, imgsz_w)`로 설정하고 추가로 실제 치수와 함께 `imgsz_h`와 `imgsz_w`를 기록합니다. 직사각형 필드를 이해하는 리더는 스칼라보다 이를 선호해야 합니다. HRNet 포즈와 같은 고정 직사각형 계약을 가진 계열는 호환되지 않는 런타임 크기를 거부합니다.

스키마는 의도적으로 평면이며, `model`는 의도적으로 상태 딕트입니다.

<code-tabs name="usage" />

## 포즈 추가

포즈는 보통 단일 클래스이며, `nc: 1`는 `person`와 함께 사용되지만, YOLO-NAS 포즈 헤드는 하나의 공유 키포인트 스켈레톤으로 다중 클래스 포즈도 지원하며, 이 경우 `nc`와 `names`가 탐지에서처럼 클래스를 설명합니다. 런타임 포즈 내보내기는 `scores`를 `[batch, anchors, nc]` 형태로 생성합니다.

| 열쇠 | 의미 |
|---|---|
| `num_keypoints` | 포즈 헤드에서 사용되는 양성 키포인트 수 |
| `keypoint_dim` | `2`는 `x,y` 레이블용, `3`는 `x,y,visibility` 레이블용; 모델 출력은 항상 `x,y,visibility`를 노출합니다 |
| `oks_sigmas` | 선택적 키포인트별 OKS 시그마; 부재 시 `num_keypoints`의 작업 기본값이 사용됩니다 |
| `num_keypoints_per_class` | 클래스별 키포인트 수를 선택적으로 지정할 수 있으며, 키포인트 텐서가 클래스별로 패딩된 GroupPose 스타일 헤드에 해당; 키포인트가 없는 클래스에는 `0` |

## 메시 추가

메시 체크포인트는 `task: "mesh"`, `nc: 1` 및 `names: {0: "person"}`를 사용합니다. 매개변수 배치는 바디 모델마다 다르기 때문에 치수를 가정하지 않고 기록합니다.

| 열쇠 | 의미 |
|---|---|
| `body_model` | 아래의 모든 필드를 해석하는 데 필요하고 사용되는 `mhr`와 같은 매개변수화 |
| `num_betas` | 정체성과 형태 계수 수; MHR의 경우 45 |
| `num_body_pose` | 몸 자세 매개변수 블록의 너비; MHR의 경우 130. 각 관절마다 하나의 삼중항이 아니라 평면 벡터, 왜냐하면 리그 관절은 서로 다른 자유도를 가지기 때문 |
| `num_vertices` | 디코더가 내보내는 정점 수; MHR의 경우 18439 |
| `num_joints` | 디코더가 출력하는 관절 수; MHR의 경우 127 |
| `rotation_format` | 회전이 어떻게 인코딩되는지, 예를 들어 MHR의 경우 `euler_zyx` 또는 `axis_angle`처럼. 텐서 형태에서 추론하지 말 것, 3-벡터는 모호하기 때문 |

## 밀집 작업 자리 표시자

몇 가지 작업은 클래스를 예측하기보다는 밀집 지도를 예측하므로, 클래스와 같은 슬롯은 스키마 호환성을 위해서만 존재합니다.

| 작업 | `nc` | `names` |
|---|---|---|
| `depth` | 1 | `{0: "depth"}` |
| `edge` | 1 | `{0: "edge"}` |
| `restore` | 1 | `{0: "image"}` |
| `ocr` | 1 | `{0: "text"}` |

엣지 예측은 `[0, 1]`에서의 밀집 float32 확률 맵입니다.

체크포인트 복원은 `degradation`, `deblur`, `denoise` 또는 `super-resolution`와 같은 짧은 손상 레이블인 `degradation`; `dataset`, `GoPro` 또는 `SIDD`와 같은 출처 레이블; 그리고 `scale`, 예를 들어 x4 초해상도 모델의 경우 `4`와 같은 양의 정수 출력-입력 업스케일 계수를 추가할 수 있습니다. 없거나 `1`인 경우 복원된 이미지는 입력 해상도를 유지합니다. 런타임은 또한 계열과 크기에서 스케일을 도출하므로 `scale`는 로드 시 요구 사항이 아닌 출처 메타데이터입니다.

## OCR 추가

`ppocr` 계열은 각 계층마다 하나의 복합 체크포인트를 제공하며, 이 체크포인트의 `model` 상태 딕셔너리에는 `det.*` 및 `rec.*` 키 네임스페이스 아래에 두 개의 서브모델이 포함되어 있습니다.

| 열쇠 | 의미 |
|---|---|
| `charset` | 출력 인덱스 순서의 전체 CTC 알파벳: 인덱스 0은 CTC 블랭크이고, 그 다음은 인식 사전, 그 다음은 공백 문자입니다. 로더는 체크포인트에서 반드시 읽어야 하며, 별도의 파일에서 읽으면 안 됩니다. |
| `pipeline` | 변환 시점에 내장된 파이프라인 기본값: `det_limit_side_len`, `det_db_thresh`, `det_db_box_thresh`, `det_db_unclip_ratio`, `rec_image_shape`. 런타임 인수는 호출마다 이들을 재정의할 수 있습니다 |
| `components` | 문서 방향, 왜곡 보정 및 텍스트 라인 회전과 같은 선택적 파이프라인 단계를 위해 예약됨. v1에서는 비어 있음 |

## 런타임 메타데이터 내보내기

내보낸 아티팩트는 동일한 직사각형 이중 쓰기 규약을 사용합니다: `imgsz_h`와 `imgsz_w`는 기존 스칼라 `imgsz` 옆에 쓰이며, 직사각형 필드를 이해하지 못하는 리더는 스칼라를 사각형 계약으로 조용히 취급해서는 안 됩니다.

직사각형 런타임 지원은 계열 범위와 형식 범위에 속합니다. YOLO9-family, HRNet, NAFNet 및 Real-ESRGAN 내보내기는 지원되는 형식에서 비정사각형 `imgsz_h` 및 `imgsz_w`를 사용할 수 있습니다. 명시적인 직사각형 지원이 없는 계열이나 형식은 해당 아티팩트를 정사각형으로 전처리하는 대신 메타데이터를 거부합니다. HRNet 내보내기는 고정된, 배치 1, FP32 사람 크롭 헤드이며, W32는 256x192를, W48은 384x288을 수용하고, 사람 탐지기는 그래프에 내장되어 있지 않습니다.

임베디드-NMS 내보내기는 이러한 평면 키를 추가할 수 있습니다:

| 열쇠 | 의미 |
|---|---|
| `nms` | 문자열 불리언; `"true"`는 그래프가 포함된 후처리 출력을 포함하고 있음을 의미합니다 |
| `nms_conf` | 임베디드 출력에 내장된 신뢰도 임계값 |
| `nms_iou` | 임베디드 출력에 내장된 IoU 임계값 |
| `max_det` | 임베디드 출력이 내보내는 최대 NMS 이후 탐지 행 |
| `nms_raw_output` | 문자열 boolean; `"true"`는 그래프가 보조 원시 탐지기 출력을 또한 노출한다는 것을 의미한다 |

`nms=true`와 함께 ONNX YOLO9 탐지 내보내기의 경우, 출력 `0`(이름 `output`)는 내보내기 시점의 임계값에서 독립적인 post-NMS 텐서입니다. `nms_raw_output=true`인 경우, 출력 `1`(이름 `raw`)는 LibreYOLO 백엔드가 원래 캔버스 클리핑 및 런타임 `predict(conf=..., iou=..., max_det=...)` 의미론을 적용할 수 있도록 예약됩니다. 타사 사용자는 첫 번째 출력을 사용해야 합니다.

포즈 내보내기는 `num_keypoints`; `keypoint_dim`를 추가할 수 있으며, GroupPose 스타일의 원시 내보내기는 텐서에 정밀도 또는 클래스 로짓 필드가 포함된 경우 `8`와 같은 더 큰 값을 사용할 수 있습니다; JSON으로 인코딩된 목록인 `num_keypoints_per_class`이며, 제로 키포인트 클래스 슬롯은 스키마를 정의하므로 유지해야 합니다; 그리고 `pose_input`, 여기서 `"person_crop"`는 그래프가 이미 추출된 하나의 크롭을 사용하며 디텍터를 포함하지 않음을 의미합니다. HRNet 런타임 내보내기는 해당 값을 필요로 합니다.

분류 내보내기는 `crop_pct`를 추가할 수 있으며, 이는 사전 자르기 리사이즈 목표가 `round(imgsz / crop_pct)`인 float 중심 자르기 비율이며, 없을 경우 기본값은 `0.875`입니다. 또한 `interpolation`, `"bilinear"` 또는 `"bicubic"`를 추가할 수 있으며, 기본값은 `"bilinear"`입니다.

ExecuTorch 내보내기는 평면 메타데이터를 필수 `<program>.pte.json` 사이드카에 작성합니다. v1 계약은 CPU, FP32, 배치 1 및 고정 입력 캔버스를 사용하며, 추가적으로 `executorch_version`, `executorch_delegate`가 `"xnnpack"`와 같고, 양수 `executorch_delegate_partitions`가 필요합니다. 로더는 다른 대리인, 동적 형태, 또는 비 FP32 정밀도를 주장하는 사이드카를 거부합니다.

MNN 내보내기는 평면 메타데이터를 필수 `<model>.mnn.json` 사이드카에 기록합니다. v1 계약은 CPU, FP32, 탐지 전용이며 고정된 NCHW 입력 형태를 가지며, 추가로 `mnn_version`, `mnn_backend`가 `"cpu"`와 같아야 하고, 순서가 있는 비어 있지 않은 `mnn_input_names`와 `mnn_output_names`, `mnn_input_shape`는 `[batch, channels, height, width]` 순서로 된 네 개의 양의 정수여야 하며, `mnn_batch`는 `mnn_input_shape[0]`와 같아야 합니다. 로더는 동적이거나 FP32가 아니거나, 탐지 기능이 없거나, 지원되지 않는 계열이거나, 형태 메타데이터가 일치하지 않으면 거부합니다.

`.pte`와 `.mnn`는 백엔드별 특수 파일로, PyTorch 체크포인트가 아닙니다.

## 양자화된 체크포인트

양자화된 모델은 선택적 평면 키 `quant`를 추가하며, 이 키는 `schema`, `recipe`, `keep_high_precision`, `execution`, 보정 출처, `module_count` 및 `state`가 포함된 매니페스트 딕트(manifest dict)를 보유합니다. FP8 매니페스트는 또한 `fp8_tensorwise_weights`, 가중치 스케일이 출력 채널별이 아니라 텐서별인 `QuantLinear` 모듈 이름의 정확한 목록을 포함할 수 있습니다. `quant`를 확인한 로더는 `load_state_dict` 전에 양자화된 모듈 구조와 스케일링 정책을 재구성합니다.

`state`는 두 가지 유물 형태를 구별합니다.

`"prepared"`, 기본값은 FP32 마스터 가중치와 `_q_*` 스케일 버퍼를 포함하며 학습이 가능합니다. 양자화 지원이 없는 리더는 `quant` 키를 무시하고 마스터를 float 모델로 로드할 수 있습니다.

`"finalized"`는 `export(format="pt")`가 작성한 배포 형식입니다. 마스터는 제거되고 각 양자화된 모듈은 대신 패킹된 가중치를 가지고 있습니다:

| 레시피 | 패킹된 텐서 | 비양자화 |
|---|---|---|
| int8 | 원래 가중치 형태의 `weight_packed` int8, 채널별 `_q_w_scale` FP32 | `weight_packed * scale` |
| fp8 | 원래 형태의 `weight_packed` float8_e4m3fn, 출력 채널당 하나의 항목인 `_q_w_scale` FP32 | `weight_packed * scale` |
| w4a16, w4a8 | `weight_packed` uint8, 바이트당 2개의 4비트 코드, 낮은 니블 우선, 코드 `q + 8`; `_q_w_gscale` FP32 `[out, ngroups]`, in_features를 따라 128 그룹 | 그룹별 척도 |
| int2 | 바이트당 4개의 2비트 코드, 코드 `q + 2`, 그룹 64 | 그룹별 척도 |
| nvfp4 | `weight_packed` uint8 `[out, ceil(in/16)*8]`, 코드 `sign<<3 "| E2M1 레벨`; `weight_block_scale` float8_e4m3fn `[out, ceil(in/16)]`; `_q_w_amax` FP32 텐서당 | `block_scale * amax / (448 * 6)` |
| mxfp4 | nvfp4와 같지만 32-요소 블록, plus `weight_block_exp` int8 `[out, ceil(in/32)]` | `2 ** exponent` |

활성화 범위 버퍼 `_q_act_lo`, `_q_act_hi` 및 `_q_calibrated`는 int8에 대해 유지됩니다. 매니페스트에는 정량화되지 않은 텐서에 대해 `remainder`, `"fp16"` 또는 `"fp32"`가 기록됩니다. 언패킹은 시뮬레이션을 비트 단위로 재현하므로, 최종 추론은 최종 장치에서 준비된 추론과 정확히 일치합니다. 이 레이아웃은 외부 내보내기업체와 런타임을 위한 안정적인 계약입니다.

## 학습 체크포인트

트레이너 체크포인트는 동일한 필수 메타데이터 코어를 사용하며, 평면 학습 및 재개 필드를 추가할 수 있습니다:

```python
{
    "model": state_dict,
    "epoch": 42,
    "optimizer": optimizer_state_dict,
    "config": {},
    "loss": 1.23,
    "best_metric_key": "metrics/mAP50-95",
    "best_metric_value": 0.51,
    "best_epoch": 39,
    "is_ema_weights": True,
    "train_model": raw_state_dict,
    "ema": ema_state_dict,
    "ema_updates": 12345,
}
```

`is_ema_weights`는 최상위 `model`가 EMA로 스무딩되었는지 여부를 선언합니다. EMA가 활성화되면 `train_model`, `ema` 및 `ema_updates`는 재개 상태를 유지합니다. 공개된 추론 가중치는 경량이어야 하며, 의도적으로 학습 체크포인트로 배포되지 않는 한 옵티마이저, 에포크, 구성, 손실 또는 EMA 재개 상태를 포함해서는 안 됩니다.

릴리스 호환성을 위해, 독자들은 레거시 베스트 메트릭 별칭 `best_mAP50_95`, `best_mAP50`, `best_metric` 및 `best_metric_name`를 수용합니다.

## 외부 스냅샷

이 스키마는 LibreYOLO에서 작성한 `.pt` 파일을 관리합니다. 별도의 모델 계층에서 사용하는 다중 파일 업스트림 스냅샷을 이름 변경하거나 래핑하지 않습니다.

LibreMODUS 크기 `14b-a7b`는 명시적인 예외입니다: 별칭은 `LibreVLM(...)`를 통해 고정된 업스트림 파일 디렉토리로 해결되며, LibreYOLO는 여기에 v1.0 메타데이터를 추가하지도 않고 `.pt`로 다시 게시하지도 않습니다.

## 전통 및 외국 무게

새로운 작성자는 엄격하게 검증하며 v1.0 메타데이터를 반드시 출력해야 합니다. 메타데이터가 없거나 불완전한 경우, 레거시 LibreYOLO 스타일의 체크포인트는 경고와 변환 지침과 함께 호환성 경로를 통해 로드되며, 외부 업스트림 체크포인트는 자동 변환으로 라우팅됩니다. [업스트림 체크포인트](/docs/reference/upstream-checkpoints)를 참조하십시오.

## 도움이 되는 사람들

스키마 도우미는 `libreyolo.utils.serialization`에 있습니다:

```python
wrap_libreyolo_checkpoint(
    state_dict,
    *,
    model_family,
    size,
    task,
    nc,
    names=None,
    imgsz=None,
    libreyolo_version=None,
    schema_version="1.0",
    **extra_metadata,
) -> dict

validate_checkpoint_metadata(checkpoint, *, strict=False) -> list[str]

unwrap_libreyolo_checkpoint(loaded, *, strict=False) -> tuple[dict, dict]
```

`validate_checkpoint_metadata`는 변이를 일으키지 않으며 오류 목록을 반환합니다; `strict=True`를 사용하면 대신 `CheckpointMetadataError`를 발생시킵니다. `model.save(path)`는 준수하는 체크포인트를 작성하는 지원되는 방법입니다.
