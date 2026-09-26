---
title: libreyolo predict
seo_title: libreyolo predict 명령 레퍼런스
description: '명령줄에서 추론을 실행합니다: 모든 인자, CLI 정의에서 읽어온 기본값, 그리고 stdout에 무엇이 출력될지 바꾸는 플래그.'
lead: >-
  불러온 모델을 하나의 소스에 대해 실행하고 예측 결과를 출력합니다. 소스는 이미지, 디렉터리, 비디오, URL 또는 실시간 스트림일 수
  있고, 모델은 체크포인트일 수도 내보낸 산출물일 수도 있습니다.
keywords:
  - libreyolo predict cli
  - libreyolo 추론 명령어
  - yolo cli 예측
  - libreyolo predict 인자
  - libreyolo json 출력
last_verified: 1.5.0
meta:
  - label: 명령
    value: libreyolo predict
    mono: true
  - label: 필수
    value: source
    mono: true
  - label: 출력
    value: 예측 결과는 stdout으로. save=true이면 어노테이션된 파일이 runs/detect/predict 아래에
snippets:
  examples:
    - label: 기본
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: 어노테이션된 이미지 저장
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt save=true \
          project=runs/detect name=parkour exist_ok=true \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: '클래스 필터링, stdout으로 JSON 출력'
      language: bash
      code: >
        # 클래스 0은 체크포인트에 포함된 COCO 클래스 목록에서 person입니다.

        libreyolo predict model=LibreYOLO9s.pt classes="[0]" conf=0.4 max_det=50
        \
          json=true quiet=true \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
source_hash: 7e46c7ed7dd9e6c4
---

## 개요

```bash
libreyolo predict source=<path|url|index> [model=<name|path>] [key=value ...]
```

인자는 `key=value` 쌍입니다. 같은 명령은 POSIX 형식도 받으므로 `conf=0.4`와
`--conf 0.4`는 서로 바꿔 쓸 수 있고, `save=true`로 쓴 불리언은 `--save`가
됩니다. 밑줄이 들어간 이름은 두 표기를 모두 받습니다: `max_det=50`과
`--max-det 50`은 같은 옵션에 도달합니다.

`libreyolo detect predict ...`도 받아들여지고 동작은 동일하며, 태스크 단어는
파싱 전에 제거됩니다.

## 인자

| 인자 | 기본값 | 의미 |
|---|---|---|
| `source` | | 이미지 경로, 디렉터리 또는 URL. 필수 |
| `model` | `yolox-s` | 모델 이름 또는 경로 |
| `conf` | `0.25` | 신뢰도 임계값 |
| `iou` | `0.45` | NMS IoU 임계값 |
| `imgsz` | | 입력 이미지 크기: `640`(정사각형) 또는 `480x640`(HxW). 설정하지 않으면 모델 자체의 입력 크기 |
| `classes` | | 클래스 ID로 필터링, 예: `[0,2,5]`. 단일 정수도 받습니다 |
| `max_det` | `300` | 이미지당 최대 탐지 수 |
| `half` | `false` | FP16 추론(CUDA 전용, 모델 지원 필요) |
| `save` | `false` | 어노테이션된 이미지 저장 |
| `batch` | `1` | 디렉터리 소스에서 순전파 한 번에 처리할 이미지 수. 1보다 크면 이를 지원하는 모델에서 실제 배치 추론을 실행합니다 |
| `stream` | `false` | 결과를 점진적으로 내보냅니다. 웹캠과 실시간 스트림에서는 자동으로 켜집니다 |
| `stream_buffer` | `false` | 가장 최신 프레임만 유지하지 않고 모든 실시간 프레임을 버퍼링합니다 |
| `vid_stride` | `1` | N번째 비디오 또는 실시간 프레임마다 처리합니다 |
| `show` | `false` | 비디오와 실시간 결과를 표시하고, `q`로 중지합니다 |
| `tiling` | `false` | 큰 이미지에 대한 타일 추론 |
| `overlap_ratio` | `0.2` | 타일 겹침 비율 |
| `output_path` | | 명시적 출력 경로. 그 외에는 `save=true`일 때 `project/name` |
| `color_format` | `auto` | 입력 색상: `auto`, `rgb`, `bgr` |
| `output_file_format` | | 출력 형식: `jpg`, `png`, `webp` |
| `device` | `auto` | 장치: `0`, `cpu`, `mps`, `auto` |
| `face_detector` | | 얼굴 탐지 모델(경로 또는 CLI 이름). 시선 추정 모델에는 필수 |
| `gallery` | | 얼굴을 대조할 `libreyolo enroll` 산출 얼굴 갤러리 `.npz`. 얼굴 임베딩 모델 전용 |
| `gallery_threshold` | `0.4` | 갤러리 신원 일치에 쓰는 코사인 임계값 |
| `project` | `runs/detect` | 출력 디렉터리 루트 |
| `name` | `predict` | 실험 이름 |
| `exist_ok` | `false` | 기존 출력 디렉터리 재사용 |
| `json` | `false` | stdout으로 JSON 출력 |
| `quiet` | `false` | stderr 억제 |
| `verbose` | `false` | 상세한 stderr 출력 |
| `help_json` | `false` | 명령 스키마를 JSON으로 출력하고 종료 |

## 예제

<code-tabs name="examples" />

## 참고

내보낸 산출물은 체크포인트와 같은 방식으로 로드되므로
`model=weights/LibreYOLO9s.onnx`와 `model=weights/LibreYOLO9s.engine`은
`model`의 유효한 값입니다. 그런 런타임에서는 세 가지 옵션이 무시되는 대신
거부됩니다: `tiling`, `overlap_ratio`, `output_file_format`은 런타임 백엔드가
이를 처리할 수 없을 때 `config_unsupported`로 종료합니다.

`half`는 반대로 동작합니다. 내보낸 런타임은 이를 받아 FP16으로 실행하고,
네이티브 PyTorch 추론은 무시되었다고 기록한 뒤 FP32로 계속합니다.

시선 추정 모델은 2단계이고 자체 탐지기가 없으므로 그런 모델에는
`face_detector`가 필요합니다. `gallery`는 태스크가 `embed`인 모델에만
적용되며, 다른 모델에 전달하면 `config_unsupported`로 종료합니다.

stdout에는 결과만 실리고 그 외에는 아무것도 실리지 않으며, 진행 상황과 경고,
오류는 stderr로 갑니다. `json=true`는 호출당 JSON 객체 하나를, 스트리밍
중에는 프레임당 하나를 출력하고 각각 `schema_version`을 담습니다.
`quiet=true`는 stderr를 침묵시킵니다. 둘을 함께 쓰면 기계 판독기가 깨끗한
stdout 스트림을 얻습니다.

종료 코드는 성공 시 `0`, 사용법이나 설정 오류는 `2`, 소스를 찾을 수 없으면
`3`, 모델을 로드할 수 없으면 `4`, 그 외 런타임 실패는 `1`입니다.

`help_json=true`는 아무것도 실행하지 않고 명령의 파라미터, 타입, 기본값,
플래그를 JSON으로 출력하며, 설치된 버전에서 이 표를 다시 읽어내는 확실한
방법입니다.

관련: 데이터셋에서 측정한 지표는 [`libreyolo val`](/docs/cli/val), 위에서
언급한 런타임 산출물 생성은 [`libreyolo export`](/docs/cli/export).
