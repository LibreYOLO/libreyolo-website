---
title: libreyolo 유틸리티
seo_title: libreyolo CLI 유틸리티 명령 레퍼런스
description: >-
  LibreYOLO의 작은 명령들: version, checks, models, formats, cfg, info, metadata,
  enroll과 compare를 각각의 인자와 기본값과 함께 정리합니다.
lead: >-
  계산하기보다 보고하고 점검하는 아홉 개의 명령입니다. 환경 정보, 모델과 형식 목록, 해석된 기본값, 체크포인트 세부 정보를 출력하고, 얼굴
  갤러리를 만들고 조회합니다.
keywords:
  - libreyolo version
  - libreyolo checks
  - libreyolo 모델 목록
  - libreyolo 내보내기 형식 확인
  - yolo 체크포인트 메타데이터 보기
  - libreyolo 기본 설정 확인
  - libreyolo 얼굴 등록
  - libreyolo 얼굴 비교
last_verified: 1.5.0
meta:
  - label: 명령
    value: 'version, checks, models, formats, cfg, info, metadata, enroll, compare'
    mono: true
  - label: 출력
    value: 'stdout, 텍스트 형식 또는 json=true일 때 schema_version을 담은 객체 하나'
snippets:
  examples:
    - label: 환경
      language: bash
      code: |
        libreyolo version
        libreyolo checks
    - label: 사용 가능한 항목
      language: bash
      code: |
        libreyolo models
        libreyolo formats family=yolo9 task=detect
    - label: 체크포인트 확인
      language: bash
      code: |
        libreyolo info model=LibreYOLO9s.pt
        libreyolo metadata path=weights/LibreYOLO9s.pt
source_hash: 7b5b53c46df00c06
---

## 개요

```bash
libreyolo <command> [key=value ...]
```

인자는 `key=value` 쌍이며 POSIX 형식도 동작하므로 `model=x`와 `--model x`는 같은
인자입니다. 여기의 모든 명령은 결과를 stdout에 쓰고 `json=true`와 `quiet=true`를
받습니다.

루트 명령에는 자체 플래그가 하나 있는데, `libreyolo --version`은 버전 문자열을
출력하고 종료합니다. 이는 아래의 `version` 명령보다 작은 출력입니다.

## version

LibreYOLO 버전과 함께, 실행 기반이 되는 Python, torch, CUDA 버전을 출력합니다.

```bash
libreyolo version
```

| 인자 | 기본값 | 의미 |
|---|---|---|
| `json` | `false` | stdout으로 JSON 출력 |
| `quiet` | `false` | stderr 억제 |

## checks

환경을 더 자세히 출력합니다: Python, torch, CUDA, cuDNN, 감지된 모든 GPU의 이름과
메모리, 그리고 내보내기 경로가 사용하는 각 선택적 패키지의 설치된 버전입니다.

```bash
libreyolo checks
```

| 인자 | 기본값 | 의미 |
|---|---|---|
| `json` | `false` | stdout으로 JSON 출력 |
| `quiet` | `false` | stderr 억제 |

패키지 목록은 `onnx`, `onnxruntime`, `tensorrt`, `openvino`, `paddlepaddle`,
`x2paddle`, `mnn`, `ncnn`, `onnx2tf`, `ai-edge-litert`, `transformers`,
`scipy`를 다룹니다. 설치되지 않은 패키지는 목록에서 빠지는 대신 설치되지 않았다고
표시되므로, 실패한 내보내기를 이 명령 하나로 누락된 의존성까지 추적할 수 있습니다.

## models

모든 모델 계열을 그 작업, 크기, 해당 체크포인트로 해석되는 CLI 이름, 각 크기의
입력 해상도와 함께 나열합니다.

```bash
libreyolo models
```

| 인자 | 기본값 | 의미 |
|---|---|---|
| `json` | `false` | stdout으로 JSON 출력 |
| `quiet` | `false` | stderr 억제 |

선택적 의존성이 설치되지 않은 계열은 사용할 수 없음으로 표시되며, 사용 가능하게
만들어 주는 `pip install` 줄이 함께 나옵니다. CLI 이름은 `model=`이 축약형으로
받는 값입니다: `yolox-s`는 `LibreYOLOXs.pt`로 해석되고, 탐지가 아닌 작업에는 해당
작업 접미사가 붙습니다.

## formats

설치된 환경이 만들어 낼 수 있는 내보내기 형식을, 각 형식의 파일 확장자와 FP16 및
INT8 지원 여부와 함께 나열합니다.

```bash
libreyolo formats [family=<family>] [task=<task>]
```

| 인자 | 기본값 | 의미 |
|---|---|---|
| `family` | | 한 모델 계열의 지원 등급 표시. `model=`도 같은 옵션으로 받습니다 |
| `task` | | 정식 모델 작업. 설정하지 않으면 계열의 기본 작업 |
| `json` | `false` | stdout으로 JSON 출력 |
| `quiet` | `false` | stderr 억제 |

`family` 없이 실행하면 출력은 형식 목록뿐입니다. 함께 지정하면 각 형식에 그 계열과
작업의 지원 등급, 그 등급의 이유, 그리고 등급에 붙은 제약이 추가됩니다. 알 수 없는
계열이나 계열이 지원하지 않는 작업은 사용 오류입니다.

형식 별칭은 정식 이름 옆에 표시됩니다: `tensorrt`에는 `engine`, `tflite`에는
`litert`입니다.

## cfg

해석된 기본 설정을 출력합니다: 학습 기본값, 검증 기본값, 예측 기본값, 그리고
계열별 재정의입니다.

```bash
libreyolo cfg
```

| 인자 | 기본값 | 의미 |
|---|---|---|
| `json` | `false` | stdout으로 JSON 출력 |
| `quiet` | `false` | stderr 억제 |

값은 사본이 아니라 설정 데이터클래스에서 읽으므로, 인자를 넘기지 않았을 때 학습
실행이 무엇을 사용할지에 대해서는 이 출력이 기준입니다. `family_overrides`는 어떤
계열이 요청하지 않은 설정으로 학습된 이유를 알려 주는 섹션입니다. 그 재정의가
적용되는 방식은 [`libreyolo train`](/docs/cli/train)을 참고하십시오.

## info

모델을 CPU에 로드하고 그 계열, 크기, 파라미터 수, 클래스, 각 형식의 내보내기
등급을 보고합니다.

```bash
libreyolo info model=<name|path>
```

| 인자 | 기본값 | 의미 |
|---|---|---|
| `model` | | 모델 이름 또는 가중치 경로. 필수 |
| `detailed` | `false` | 파라미터별 세부 정보 포함 |
| `json` | `false` | stdout으로 JSON 출력 |
| `quiet` | `false` | stderr 억제 |

## metadata

모델을 구성하지 않고 체크포인트의 메타데이터를 읽어, LibreYOLO 체크포인트 스키마에
맞는지 검증합니다.

```bash
libreyolo metadata path=<checkpoint.pt>
```

| 인자 | 기본값 | 의미 |
|---|---|---|
| `path` | | `.pt` 체크포인트 경로. 필수 |
| `json` | `false` | stdout으로 JSON 출력 |
| `quiet` | `false` | stderr 억제 |

텐서를 담은 큰 항목은 그대로 출력하지 않고 요약하므로, 전체 학습 체크포인트에서도
출력이 읽을 만한 상태로 유지됩니다. 존재하지 않는 체크포인트는
`checkpoint_not_found`로 종료하고, 메타데이터 검증에 실패한 체크포인트는 오류를
출력하고 `1`로 종료합니다.

## enroll

사람마다 폴더가 하나씩인 트리에서 얼굴 갤러리를 만들어, 이후 예측이 찾아낸 얼굴에
이름을 붙일 수 있게 합니다.

```bash
libreyolo enroll model=<embedder> source=<people-dir> gallery=<gallery.npz>
```

| 인자 | 기본값 | 의미 |
|---|---|---|
| `model` | | 얼굴 임베딩 모델, 경로 또는 이름. 필수 |
| `source` | | 사람마다 폴더가 하나씩인 트리, `source/<identity>/*.jpg`. 필수 |
| `gallery` | | 출력 갤러리 파일 `.npz`. 이미 있으면 그 자리에서 확장됩니다. 필수 |
| `face_detector` | | 얼굴 탐지기: YuNet `.onnx` 또는 LibreYOLO 탐지기. 설정하지 않으면 계열의 기본 탐지기 |
| `device` | `auto` | 장치: `0`, `cpu`, `mps`, `auto` |
| `json` | `false` | stdout으로 JSON 출력 |
| `quiet` | `false` | stderr 억제 |

```bash
# people/ 아래에는 신원마다 폴더가 하나씩 있으며, 폴더 이름이 신원이 됩니다.
libreyolo enroll model=librefacerec-l.onnx source=people/ gallery=people.npz
```

하위 폴더 이름이 신원입니다. 얼굴을 탐지할 수 없는 참조 이미지는 stderr에 한 줄을
남기고 건너뛰며 나머지는 계속 진행됩니다; 신원 하위 폴더가 없는 소스나 얼굴이
하나도 발견되지 않은 소스는 오류입니다.

만들어진 파일을 `gallery=people.npz`로
[`libreyolo predict`](/docs/cli/predict)에 넘기면 탐지 결과에 신원과 일치 점수가
함께 담깁니다.

## compare

두 얼굴 이미지 사이의 코사인 유사도와, 그 값이 동일 신원 임계값을 넘는지를
보고합니다.

```bash
libreyolo compare model=<embedder> source=<a.jpg> source2=<b.jpg>
```

| 인자 | 기본값 | 의미 |
|---|---|---|
| `model` | | 얼굴 임베딩 모델, 경로 또는 이름. 필수 |
| `source` | | 첫 번째 이미지. 필수 |
| `source2` | | 비교할 두 번째 이미지. 필수 |
| `face_detector` | | 얼굴 탐지기: YuNet `.onnx` 또는 LibreYOLO 탐지기 |
| `threshold` | `0.4` | 동일 신원 판정에 쓰는 코사인 유사도 임계값 |
| `device` | `auto` | 장치: `0`, `cpu`, `mps`, `auto` |
| `json` | `false` | stdout으로 JSON 출력 |
| `quiet` | `false` | stderr 억제 |

```bash
libreyolo compare model=librefacerec-l.onnx source=a.jpg source2=b.jpg
```

`libreyolo verify`는 이 명령의 두 번째 이름으로 등록되어 있으며 같은 인자를
받습니다.

`compare`와 `enroll`은 모두 작업이 얼굴 임베딩인 모델이 필요합니다. 그 외의
모델은 `config_unsupported`로 종료합니다. 소스로는 로컬 이미지 경로와 `http`
또는 `https` URL을 모두 받습니다.

## 예제

<code-tabs name="examples" />

## 참고

결과는 stdout으로 나갑니다; 진행 상황과 경고는 stderr로 갑니다. `json=true`는
`schema_version`이 담긴 객체 하나를 출력하며, 스크립트에서 읽을 때는 이 형식을
씁니다. 텍스트 출력이 기본이며 사람이 읽도록 만들어진 것입니다.

종료 코드는 CLI의 나머지와 같은 표를 따릅니다: 성공은 `0`, 사용 또는 설정 오류는
`2`, 소스를 찾을 수 없을 때는 `3`, 모델이나 체크포인트를 로드할 수 없을 때는 `4`,
그 외 런타임 실패는 `1`입니다.

관련 문서: 데이터셋 쪽 점검 명령인 [`libreyolo doctor`](/docs/cli/doctor), 그리고
성능 쪽 명령인 [`libreyolo profile`](/docs/cli/profile)입니다.
