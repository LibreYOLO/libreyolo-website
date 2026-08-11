---
title: libreyolo export
seo_title: libreyolo export 명령 레퍼런스
description: '체크포인트를 배포 형식으로 내보냅니다: 모든 인자와 기본값, 결과물이 저장되는 위치, 그리고 명령이 거부하는 조합.'
lead: >-
  체크포인트 하나를 배포 형식 하나로 변환하고 결과물을 weights/ 아래에 씁니다. 아래 인자 중 어떤 것이 적용되는지는 형식이
  결정합니다.
keywords:
  - libreyolo export cli
  - libreyolo export 명령어
  - yolo onnx 변환 cli
  - tensorrt 내보내기 명령어
  - libreyolo export 인자
last_verified: 1.5.0
meta:
  - label: 명령
    value: libreyolo export
    mono: true
  - label: 필수
    value: model
    mono: true
  - label: 출력
    value: 'weights/<checkpoint-stem>[_fp16|_int8]<format-suffix>'
    mono: true
snippets:
  examples:
    - label: 기본
      language: bash
      code: |
        # weights/LibreYOLO9s.onnx 파일을 씁니다
        libreyolo export model=LibreYOLO9s.pt format=onnx imgsz=640
    - label: 그래프 안의 NMS
      language: bash
      code: |
        libreyolo export model=LibreYOLO9s.pt format=onnx \
          nms=true conf=0.25 iou=0.45 max_det=300
    - label: 결과물 실행하기
      language: bash
      code: |
        libreyolo export model=LibreYOLO9s.pt format=onnx imgsz=640

        # 팩토리는 파일 접미사를 보고 분기하므로 내보낸 파일이 체크포인트처럼 로드됩니다.
        libreyolo predict model=weights/LibreYOLO9s.onnx \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
source_hash: ef2ca20af3814109
---

## 요약

```bash
libreyolo export model=<name|path> [format=<format>] [key=value ...]
```

인자는 `key=value` 쌍이며 POSIX 형식도 동작하므로 `format=onnx`와
`--format onnx`는 같은 인자입니다.

## 인자

| 인자 | 기본값 | 의미 |
|---|---|---|
| `model` | | 모델 가중치 `.pt`. 필수 |
| `format` | `onnx` | 내보내기 형식: `onnx`, `torchscript`, `executorch`, `tensorrt`, `openvino`, `paddle`, `mnn`, `rknn`, `ncnn`, `tflite`, `coreml`, `coreai` |
| `name` | | RKNN 대상 플랫폼, 현재는 `rk3588`만 지원합니다. 다른 형식과 함께 쓰면 거부됩니다 |
| `imgsz` | | 입력 이미지 크기: `640` 또는 `480x640` (HxW). `480,640` 형식도 허용됩니다. 설정하지 않으면 모델 자체의 크기 |
| `batch` | `1` | 내보내기 배치 크기 |
| `half` | `false` | FP16 정밀도 |
| `int8` | `false` | INT8 양자화 |
| `dynamic` | `false` | 동적 입력 형상 (ONNX) |
| `simplify` | `true` | ONNX 그래프 단순화 |
| `nms` | `false` | 모델에 NMS를 내장합니다. ONNX와 CoreML만 해당 |
| `conf` | `0.25` | 내장 NMS의 신뢰도 임계값 |
| `iou` | `0.45` | 내장 NMS의 IoU 임계값 |
| `max_det` | `300` | ONNX 내장 NMS의 최대 탐지 수 |
| `opset` | | ONNX opset 버전. 설정하지 않으면 자동으로 선택됩니다 |
| `data` | | INT8용 보정 데이터 |
| `fraction` | `1.0` | 사용할 보정 데이터의 비율 |
| `device` | `auto` | 트레이싱에 사용할 장치 |
| `allow_download_scripts` | `false` | 데이터셋 YAML 다운로드 블록에 내장된 Python 허용 |
| `json` | `false` | stdout에 JSON 출력 |
| `quiet` | `false` | stderr 억제 |
| `verbose` | `false` | 상세 내보내기 로깅 |
| `verify` | `false` | RKNN Toolkit2 PC 시뮬레이터를 실행해 ONNX Runtime과 비교합니다. RKNN만 해당 |
| `help_json` | `false` | 명령 스키마를 JSON으로 출력하고 종료합니다 |

`engine`은 `tensorrt`의 별칭이고 `litert`는 `tflite`의 별칭입니다. 둘 다 무언가
기록되기 전에 정규 이름으로 해석되므로 JSON 출력과 로그 줄에는 항상
`tensorrt` 또는 `tflite`가 표시됩니다.

## 예제

<code-tabs name="examples" />

## 참고

### 파일이 저장되는 위치

이 명령은 출력 경로를 받지 않습니다. 결과물은 `weights/`에 기록되며, 이름은 원본
체크포인트에서 확장자를 뺀 이름에 형식의 접미사를 붙인 것이고, 두 정밀도 중
하나를 요청했다면 `_fp16` 또는 `_int8`이 중간에 삽입됩니다. `LibreYOLO9s.pt`를
FP16으로 ONNX로 내보내면 `weights/LibreYOLO9s_fp16.onnx`가 됩니다. JSON 결과에는
확정된 `output_path`, MB 단위 파일 크기, 그리고 `[batch, 3, height, width]` 형태의
입력 형상이 담깁니다.

### 거부되는 조합

`nms=true`는 ONNX와 CoreML에서 허용되고 다른 모든 형식에서는
`nms_unsupported_format`으로 거부됩니다. ONNX에서는 내장 그래프가 배치 1로
고정되기 때문에 `dynamic`을 강제로 끄고 그 사실을 stderr에 알립니다. CoreML에서는
`conf`와 `iou`는 받지만 `max_det`은 받지 않으므로, `format=coreml nms=true`와 함께
기본값이 아닌 `max_det`을 넘기면 `config_unsupported`로 종료합니다.

`half=true`와 `int8=true`를 함께 쓰는 것은 오류가 아닙니다. INT8이 우선하고
`half`는 무시되며 경고가 stderr로 나갑니다.

`name`과 `verify`는 현재 RKNN 옵션입니다. 둘 중 하나를 다른 형식과 함께 넘기면
무시되지 않고 `config_unsupported`로 종료합니다.

### 모델 계열이 지원하는 형식

지원 여부는 전역이 아니라 모델 계열별, 작업별로 정해집니다.
`libreyolo formats family=<family> task=<task>`는 그 조합에서 각 형식의 등급을
이유 및 관련 제약과 함께 출력합니다. 인자는
[`libreyolo formats`](/docs/cli/utilities) 문서를 참고하십시오.

일부 형식은 선택적 설치가 필요하고 일부는 툴체인이 필요합니다. Python 의존성이
빠져 있으면 `export_dep_missing`으로 종료하고, 형식이 만들어 낼 수 없는 정밀도를
요청하면 `format_precision_unsupported`로 종료합니다.

### 내보낸 결과물 실행

내보낸 결과물은 체크포인트와 동일한 모델 팩토리를 거쳐 파일 접미사를 키로 삼아
로드되므로 `libreyolo predict model=weights/LibreYOLO9s.onnx`는 추가 변환 없이
동작합니다. 예측 옵션 세 가지는 예외이며 런타임 백엔드에서는 거부됩니다:
`tiling`, `overlap_ratio`, `output_file_format`.

배포 대상 두 가지에는 별도의 문서 페이지가 있습니다:
[NVIDIA DeepStream](/docs/export/deepstream)과
[NVIDIA Jetson](/docs/export/jetson).

### 출력과 종료 코드

결과는 stdout으로, 진행 상황은 stderr로 나갑니다. 종료 코드는 성공 시 `0`,
사용법이나 설정 오류는 `2`, 모델을 로드할 수 없을 때는 `4`, 알 수 없는 형식이나
내보내기 의존성 누락, 지원하지 않는 정밀도, 거부된 내장 NMS 요청은 `5`, 그 외
런타임 실패는 `1`입니다.

관련 문서: [`libreyolo quantize`](/docs/cli/quantize)는 PyTorch에 머물면서 배포
결과물이 아니라 체크포인트를 씁니다.
