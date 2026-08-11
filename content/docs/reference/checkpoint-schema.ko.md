---
title: 체크포인트 스키마
seo_title: LibreYOLO 체크포인트 메타데이터 스키마 v1.0
description: >-
  모든 LibreYOLO .pt 체크포인트가 포함하는 메타데이터: 필수 키, 태스크별 추가, 내보내기 런타임 키, 양자화 매니페스트 및 학습
  필드.
lead: >-
  LibreYOLO .pt 파일은 torch.save로 저장된 평면 사전입니다. 모델 키는 상태 사전을 포함하고; 다른 최상위 키는 파일 이름
  파싱이나 상태 사전 탐지 없이 체크포인트를 식별하는 메타데이터입니다.
keywords:
  - libreyolo 체크포인트 스키마
  - schema_version 1.0
  - model_family
  - libreyolo 체크포인트 메타데이터
  - 양자 매니페스트
  - libreyolo 체크포인트 래핑
last_verified: 1.5.0
verification: >-
  v1.5.0의 libreyolo 저장소에서 docs/checkpoint_schema.md를 미러링하며,
  libreyolo/utils/serialization.py 및 BaseModel.save와 교차 확인됨.
snippets:
  usage:
    - label: 체크포인트에서 메타데이터 읽기
      language: python
      code: >
        from libreyolo import LibreYOLO

        from libreyolo.utils.serialization import unwrap_libreyolo_checkpoint

        import torch


        # 체크포인트를 다운로드한 후 로컬 경로가 존재하도록 다시 저장합니다.

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

모든 공식 LibreYOLO `.pt` 체크포인트는 다음을 포함합니다:

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

| 키 | 유형 | 의미 |
|---|---|---|
| `model` | 상태 딕셔너리 | 모델 가중치 |
| `schema_version` | 문자열 | 메타데이터 계약 버전; v1.0은 문자열 `"1.0"`를 사용합니다 |
| `libreyolo_version` | 문자열 | 체크포인트를 생성한 버전 |
| `model_family` | 문자열 | 등록된 패밀리, 예: `yolo9`, `rfdetr`, `dfine`, `ec` |
| `size` | 문자열 | 패밀리 내 변형, 예: `t`, `s`, `r18`, `atto` |
| `task` | 문자열 | 표준 작업 이름 |
| `nc` | 정수 | 양성 클래스 수 |
| `names` | 사전 | `dict[int, str]` 키가 `0..nc-1`에 있음 |
| `imgsz` | 정수 | 양의 정사각형 입력 해상도 또는 직사각형 계약용 이전 스칼라 |

`task`는 `detect`, `segment`, `semantic`, `panoptic`, `pose`, `classify`, `gaze`, `obb`, `point`, `depth`, `edge`, `normal`, `restore`, `matte`, `ocr`, `embed` 또는 `mesh` 중 하나입니다.

공식 체크포인트는 모든 `names` 키를 씁니다. 판독기는 이전 희소 매핑을 위해 누락된 키를 `class_i` 레이블로 채울 수 있지만, 범위를 벗어난 키는 잘못된 것입니다.

직사각형 체크포인트는 레거시 리더를 위해 스칼라 `imgsz`를 유지하며, `max(imgsz_h, imgsz_w)`로 설정되고, 추가적으로 실제 크기를 `imgsz_h`와 `imgsz_w`에 기록합니다. 직사각형 필드를 이해하는 리더는 스칼라보다 이를 우선해야 합니다. HRNet 포즈와 같이 고정 직사각형 계약을 가진 패밀리는 호환되지 않는 런타임 크기를 거부합니다.

스키마는 의도적으로 평면이며, `model`는 의도적으로 상태 사전입니다.

<code-tabs name="usage" />

## 포즈 추가

포즈는 일반적으로 단일 클래스이며, `nc: 1`와 `person`와 함께하지만, YOLO-NAS 포즈 헤드는 하나의 공유 키포인트 스켈레톤으로 다중 클래스 포즈도 지원하며, 이 경우 `nc`와 `names`가 감지 시와 같이 클래스를 설명합니다. 런타임 포즈 내보내기는 `scores`를 `[batch, anchors, nc]` 형상으로 내보냅니다.

| 키 | 의미 |
|---|---|
| `num_keypoints` | 포즈 헤드에서 사용되는 양성 키포인트 수 |
| `keypoint_dim` | `2`는 `x,y` 레이블용 또는 `3`는 `x,y,visibility` 레이블용; 모델 출력은 항상 `x,y,visibility`를 노출함 |
| `oks_sigmas` | 선택적 키포인트별 OKS 시그마; `num_keypoints`는 없을 경우 작업 기본값 사용 |
| `num_keypoints_per_class` | 클래스별 키포인트 수를 선택적으로 지정 (GroupPose 스타일 헤드용, 키포인트 텐서가 클래스별로 패딩된 경우); 키포인트가 없는 클래스는 `0` |

## 메시 추가

메시 체크포인트는 `task: "mesh"`, `nc: 1` 및 `names: {0: "person"}` 사용. 파라미터 레이아웃은 바디 모델별로 다르므로 차원을 가정하지 않고 기록함

| 키 | 의미 |
|---|---|
| `body_model` | 매개변수화, 예: `mhr`; 필수이며 아래의 각 필드를 해석하는 데 사용됨 |
| `num_betas` | 신원 및 형태 계수 수; MHR의 경우 45 |
| `num_body_pose` | 본체-포즈 매개변수 블록의 너비; MHR의 경우 130. 각 관절마다 하나의 3벡터가 아니라 평면 벡터, 관절마다 자유도가 다르기 때문 |
| `num_vertices` | 디코더가 출력하는 정점 수; MHR의 경우 18439 |
| `num_joints` | 디코더가 출력하는 관절 수; MHR의 경우 127 |
| `rotation_format` | 회전이 인코딩되는 방식, 예: MHR의 경우 `euler_zyx` 또는 `axis_angle`. 텐서 형태에서 유추하지 않음, 3벡터는 모호하기 때문 |

## 밀집 작업 자리 표시자

몇 가지 작업은 클래스를 예측하는 대신 밀집 맵을 예측하므로, 클래스와 유사한 슬롯은 스키마 호환성을 위해서만 존재함.

| 작업 | `nc` | `names` |
|---|---|---|
| `depth` | 1 | `{0: "depth"}` |
| `edge` | 1 | `{0: "edge"}` |
| `restore` | 1 | `{0: "image"}` |
| `ocr` | 1 | `{0: "text"}` |

엣지 예측은 `[0, 1]`에서 밀집 float32 확률 맵입니다.

복원 체크포인트는 `degradation`, `deblur`, `denoise` 또는 `super-resolution`와 같은 짧은 손상 레이블; `GoPro` 또는 `SIDD`와 같은 출처 레이블; 및 `scale`, 예를 들어 x4 초해상도 모델의 경우 `4`와 같은 양의 정수 출력-입력 업스케일 계수를 추가할 수 있습니다. 부재하거나 `1`이면 복원된 이미지는 입력 해상도를 유지합니다. 런타임은 또한 패밀리와 크기에서 스케일을 유도하므로 `scale`는 로드 시 요구사항이 아닌 출처 메타데이터입니다.

## OCR 추가 사항

`ppocr` 계열은 각 계층마다 하나의 복합 체크포인트를 배송하며, 그 `model` 상태 사전에는 `det.*` 및 `rec.*` 키 네임스페이스 아래 두 개의 하위 모델이 포함되어 있습니다.

| 키 | 의미 |
|---|---|
| `charset` | 출력 인덱스 순서의 전체 CTC 알파벳: 인덱스 0은 CTC 공백이며, 그다음이 인식 사전, 그다음이 공백 문자입니다. 로더는 체크포인트에서 읽어야 하며, 별도 파일에서 읽으면 안 됩니다. |
| `pipeline` | 변환 시 내장된 파이프라인 기본값: `det_limit_side_len`, `det_db_thresh`, `det_db_box_thresh`, `det_db_unclip_ratio`, `rec_image_shape`. 런타임 인자는 호출별로 이를 재정의할 수 있습니다. |
| `components` | 문서 방향, 왜곡 제거 및 텍스트 라인 회전과 같은 선택적 파이프라인 단계에 예약됨. v1에서는 비어 있음. |

## 런타임 메타데이터 내보내기

내보낸 아티팩트는 동일한 직사각형 이중 쓰기 규칙을 사용합니다: `imgsz_h`와 `imgsz_w`는 기존 스칼라 `imgsz` 옆에 기록되며, 직사각형 필드를 이해하지 못하는 리더는 스칼라를 사각형 계약으로 조용히 처리해서는 안 됩니다.

직사각형 런타임 지원은 패밀리 범위와 형식 범위로 제한됩니다. YOLO9 패밀리, HRNet, NAFNet 및 Real-ESRGAN 내보내기는 지원되는 형식에서 비정사각형 `imgsz_h`와 `imgsz_w`를 사용할 수 있으며; 직사각형 지원이 명시적으로 없는 패밀리나 형식은 해당 메타데이터를 사각형으로 전처리하지 않고 거부합니다. HRNet 내보내기는 고정된, 배치 1, FP32 사람 크롭 헤드로, W32는 256x192를, W48는 384x288을 수용하며, 사람 탐지기는 그래프에 포함되지 않습니다.

내장 NMS 내보내기는 이러한 평면 키를 추가할 수 있습니다:

| 키 | 의미 |
|---|---|
| `nms` | 문자열 불리언; `"true"`는 그래프가 포함된 후처리 출력을 포함함을 의미합니다 |
| `nms_conf` | 내장 출력에 포함된 신뢰도 임계값 |
| `nms_iou` | 내장 출력에 포함된 IoU 임계값 |
| `max_det` | 내장 출력이 내보내는 최대 후-NMS 감지 행 수 |
| `nms_raw_output` | 문자열 불리언; `"true"`는 그래프가 보조 원시 탐지기 출력을 또한 노출함을 의미합니다 |

ONNX YOLO9 감지 내보내기를 `nms=true`와 함께 사용할 경우, 출력 `0`(이름 `output`)는 내보내기 시점의 임계값에서 독립적인 NMS 후 텐서입니다. `nms_raw_output=true`일 때, 출력 `1`(이름 `raw`)는 LibreYOLO 백엔드용으로 예약되어 있어, 원래 캔버스 클리핑 및 런타임 `predict(conf=..., iou=..., max_det=...)` 의미를 적용할 수 있습니다. 서드파티 사용자는 첫 번째 출력을 사용해야 합니다.

포즈 내보내기는 `num_keypoints`; `keypoint_dim`를 추가할 수 있으며, GroupPose 스타일의 원시 내보내기는 텐서에 정밀도 또는 클래스 로짓 필드가 포함될 경우 `8`와 같은 더 큰 값을 사용할 수 있습니다; `num_keypoints_per_class`를 JSON 인코딩된 리스트로 사용하며, 제로-키포인트 클래스 슬롯은 스키마를 정의하므로 보존해야 합니다; `pose_input`에서는 `"person_crop"`가 그래프가 이미 추출된 크롭을 하나 소비하며 탐지기를 포함하지 않음을 의미합니다. HRNet 런타임 내보내기는 해당 값을 필요로 합니다.

분류 내보내기는 `crop_pct`를 추가할 수 있으며, 이는 사전 자르기 크기 조정 대상이 `round(imgsz / crop_pct)`인 부동 소수점 센터 크롭 비율이며, 없을 경우 기본값은 `0.875`입니다. 또한 `interpolation`, `"bilinear"` 또는 `"bicubic"`를 포함할 수 있으며, 기본값은 `"bilinear"`입니다.

ExecuTorch 내보내기는 평탄화된 메타데이터를 필수 `<program>.pte.json` 사이드카에 기록합니다. v1 계약은 CPU, FP32, 배치 1 및 고정 입력 캔버스를 사용하며, 추가로 `executorch_version`와 `executorch_delegate`가 `"xnnpack"`와 같아야 하고, 양의 `executorch_delegate_partitions`가 필요합니다. 로더는 다른 대리자, 동적 형상 또는 FP32가 아닌 정밀도를 주장하는 사이드카를 거부합니다.

MNN은 플랫 메타데이터를 필요한 `<model>.mnn.json` 사이드카에 기록합니다. v1 계약은 CPU, FP32, 감지 전용이며 고정된 NCHW 입력 형태를 가지며, 추가로 `mnn_version`, `mnn_backend`가 `"cpu"`와 같아야 하고, 비어 있지 않은 순서가 있는 `mnn_input_names`와 `mnn_output_names`, `mnn_input_shape`는 `[batch, channels, height, width]` 순서의 네 개 양수 정수여야 하며, `mnn_batch`는 `mnn_input_shape[0]`와 같아야 합니다. 로더는 동적, 비-FP32, 비감지, 지원되지 않는 패밀리 또는 일관되지 않은 형태의 메타데이터를 거부합니다.

`.pte`와 `.mnn`는 PyTorch 체크포인트가 아닌 백엔드 특정 아티팩트입니다.

## 양자화된 체크포인트

양자화 모델은 선택적 평면 키 `quant`를 하나 추가하며, 이 키는 `schema`, `recipe`, `keep_high_precision`, `execution`, 교정 출처, `module_count` 및 `state`를 포함한 매니페스트 사전을 보유합니다. FP8 매니페스트는 또한 `fp8_tensorwise_weights`를 포함할 수 있으며, 이는 가중치 스케일이 출력 채널별이 아닌 텐서별인 `QuantLinear` 모듈 이름의 정확한 목록입니다. `quant`를 감지한 로더는 `load_state_dict` 전에 양자화된 모듈 구조와 스케일링 정책을 재구성합니다.

`state`는 두 가지 아티팩트 형식을 구별합니다.

기본값인 `"prepared"`는 FP32 마스터 가중치와 `_q_*` 스케일 버퍼를 보유하며 학습 가능 합니다. 양자화를 지원하지 않는 리더는 `quant` 키를 무시하고 마스터를 플로트 모델로 로드할 수 있습니다.

`"finalized"`는 `export(format="pt")`가 작성한 배포 형식입니다. 마스터는 제거되고, 각 양자화 모듈은 대신 패킹된 가중치를 가집니다:

| 레시피 | 패킹된 텐서 | 역양자화 |
|---|---|---|
| int8 | `weight_packed` 원래 가중치 형상의 int8, `_q_w_scale` 채널별 FP32 | `weight_packed * scale` |
| fp8 | `weight_packed` 원래 형상의 float8_e4m3fn, `_q_w_scale` 출력 채널 당 FP32 한 항목 | `weight_packed * scale` |
| w4a16, w4a8 | `weight_packed` uint8, 바이트당 두 개의 4비트 코드, 먼저 낮은 니블, 코드 `q + 8`; `_q_w_gscale` FP32 `[out, ngroups]`, in_features를 따라 128 그룹 | 그룹별 스케일 |
| int2 | 바이트당 네 개의 2비트 코드, 코드 `q + 2`, 64 그룹 | 그룹별 스케일 |
| nvfp4 | `weight_packed` uint8 `[out, ceil(in/16)*8]`, 코드 'sign<<3 \| E2M1 level`; `weight_block_scale` float8_e4m3fn `[out, ceil(in/16)]`; `_q_w_amax' 텐서당 FP32 | `block_scale * amax / (448 * 6)` |
| MXFP4 | nvfp4와 32요소 블록, 그리고 `weight_block_exp` int8 `[out, ceil(in/32)]` | `2 ** exponent` |

활성화 범위 버퍼는 int8을 위해 `_q_act_lo`, `_q_act_hi`, `_q_calibrated` 유지됩니다. 매니페스트는 비양자화된 텐서에 대해 `remainder`, `"fp16"` 또는 `"fp32"`을 기록합니다. 언패킹은 시뮬레이션 비트 단위로 재현하므로, 최종 추론은 최종 추론 장치에서 준비된 추론과 정확히 일치합니다. 이 레이아웃은 외부 익스포터와 런타임의 안정적인 계약입니다.

## 학습 체크포인트

트레이너 체크포인트는 동일한 필수 메타데이터 코어를 사용하며, 플랫 트레이닝 및 재개(resume) 필드를 추가할 수 있습니다:

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

`is_ema_weights`는 최상위 `model`가 EMA-스무딩되는지 여부를 선언합니다. EMA가 활성화되면 `train_model`, `ema` 및 `ema_updates`는 재개 상태를 보존합니다. 게시된 추론 가중치는 가볍게 유지되어야 하며, 의도적으로 트레이닝 체크포인트로 배포되지 않는 한 옵티마이저, 에폭, 구성, 손실 또는 EMA 재개 상태를 포함해서는 안 됩니다.

릴리스 호환성을 위해 리더는 기존의 최고 메트릭 별칭인 `best_mAP50_95`, `best_mAP50`, `best_metric` 및 `best_metric_name`를 허용합니다.

## 외부 스냅샷

이 스키마는 LibreYOLO가 작성한 `.pt` 파일을 관리합니다. 별도의 모델 계층에서 사용되는 다중 파일 업스트림 스냅샷의 이름을 바꾸거나 래핑하지 않습니다.

LibreMODUS 크기 `14b-a7b`는 명시적인 예외입니다: 별칭은 `LibreVLM(...)`를 통해 고정된 업스트림 파일 디렉토리로 해결되며, LibreYOLO는 여기에 v1.0 메타데이터를 추가하지도 않고 `.pt`로 다시 게시하지도 않습니다.

## 레거시 및 외국 가중치

새로운 작성자는 철저히 검증하고 v1.0 메타데이터를 반드시 발행해야 합니다. 메타데이터가 없거나 불완전한 경우, 레거시 LibreYOLO 스타일의 체크포인트는 경고와 변환 지침과 함께 호환성 경로로 로드되며, 외국 업스트림 체크포인트는 자동 변환으로 라우팅됩니다. [업스트림 체크포인트](/docs/reference/upstream-checkpoints)를 참조하십시오.

## 헬퍼

스키마 헬퍼는 `libreyolo.utils.serialization`에 있습니다:

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

`validate_checkpoint_metadata`는 비변형이며 오류 목록을 반환합니다; `strict=True`와 함께 사용하면 대신 `CheckpointMetadataError`를 발생시킵니다. `model.save(path)`는 준수하는 체크포인트를 작성하는 지원되는 방식입니다.
