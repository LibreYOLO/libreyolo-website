---
title: libreyolo quantize
seo_title: libreyolo quantize 명령 레퍼런스
description: 'PyTorch 체크포인트를 명령줄에서 양자화합니다: 레시피, 캘리브레이션 인자, 기본값, 그리고 각 레시피가 받는 모델 계열.'
lead: >-
  모델의 float 모듈을 양자화된 모듈로 교체하고, 레시피에 통계값이 필요한 경우 레이블 없는 이미지로 캘리브레이션한 뒤, 결과를
  PyTorch 체크포인트로 저장합니다.
keywords:
  - libreyolo quantize cli
  - int8 양자화 명령어
  - fp8 양자화
  - 학습 후 양자화 yolo
  - libreyolo quantize 인자
last_verified: 1.5.0
meta:
  - label: 명령
    value: libreyolo quantize
    mono: true
  - label: 필수
    value: model
    mono: true
  - label: 출력
    value: '접미사 앞에 -<recipe>가 붙은 원본 경로, 예: LibreYOLO9s-int8.pt'
    mono: true
snippets:
  examples:
    - label: 기본
      language: bash
      code: |
        # coco128에서 캘리브레이션하고 LibreYOLO9s-int8.pt를 씁니다
        libreyolo quantize model=LibreYOLO9s.pt recipe=int8
    - label: '캐스팅만, 캘리브레이션 없음'
      language: bash
      code: |
        libreyolo quantize model=LibreYOLO9s.pt recipe=fp16 calib=none \
          out=weights/LibreYOLO9s-fp16.pt
    - label: 캘리브레이션 확대 후 정확도 회복
      language: bash
      code: >
        libreyolo quantize model=LibreYOLO9s.pt recipe=int8 \
          calib=coco128.yaml samples=256 batch=16 algorithm=minmax

        # 양자화된 체크포인트에 양자화 인식 학습을 적용하면 정확도가 회복됩니다.

        libreyolo train model=LibreYOLO9s-int8.pt data=coco8.yaml epochs=10
        lr0=0.001
source_hash: 7ae663e9f117826e
---

## 개요

```bash
libreyolo quantize model=<name|path> [recipe=<recipe>] [key=value ...]
```

인자는 `key=value` 쌍이며 POSIX 형식도 동작하므로 `recipe=int8`과
`--recipe int8`은 같은 인자입니다.

## 인자

| 인자 | 기본값 | 의미 |
|---|---|---|
| `model` | | 모델 가중치 `.pt`. 필수 |
| `recipe` | `int8` | 양자화 레시피: `fp16`, `bf16`, `fp8`, `int8`, `w4a16`, `w4a8`, `nvfp4`, `mxfp4`, `int2` |
| `calib` | `coco128.yaml` | 캘리브레이션 이미지: 데이터 YAML 또는 내장 데이터셋 이름. 레이블 없이 순전파에만 사용합니다. `none`은 캘리브레이션을 건너뜁니다 |
| `samples` | `128` | 최대 캘리브레이션 이미지 수 |
| `batch` | `8` | 캘리브레이션 배치 크기 |
| `algorithm` | `auto` | 활성화 범위 추정: minmax를 선택하는 `auto`, 또는 `minmax`, 또는 `percentile` |
| `out` | | 출력 체크포인트 경로. 기본값은 접미사 앞에 `-<recipe>`가 붙은 원본 경로입니다 |
| `device` | `auto` | 장치 |
| `allow_download_scripts` | `false` | 데이터셋 YAML의 download 블록에 내장된 Python 허용 |
| `json` | `false` | stdout으로 JSON 출력 |
| `quiet` | `false` | stderr 억제 |
| `help_json` | `false` | 명령 스키마를 JSON으로 출력하고 종료 |

## 예제

<code-tabs name="examples" />

## 참고

### 지원하는 모델 계열

양자화는 네 개 계열을 지원합니다: `yolo9`, `rfdetr`, `birefnet`,
`feynobg`. 그 외의 계열은 이 목록을 담은 `quantize_failed`로 종료됩니다.

### 각 레시피가 건드리는 부분

`fp16`과 `bf16`은 캐스팅입니다. dtype만 바꾸고 캘리브레이션이 필요 없으므로
`calib=none`이 알맞은 설정입니다.

`int8`과 `fp8`은 `Conv2d`와 `Linear` 모듈을 양자화하며, 그래서 합성곱 계열에
적합합니다.

`w4a16`, `w4a8`, `nvfp4`, `mxfp4`, `int2`는 `nn.Linear`만 양자화하므로
트랜스포머 계열을 대상으로 합니다. `yolo9`에 이 중 하나를 요청하면, 그 계열에서
8비트 미만 가속은 GEMM에만 적용되고 합성곱은 더 높은 정밀도로 남기 때문에,
양자화되지 않은 모델을 조용히 내놓는 대신 설명과 함께 거부됩니다.

`int8`, `fp8`, `w4a8`, `int2`는 활성화에 대한 캘리브레이션 통계값이 필요합니다.
`int2`는 이후 정확도를 회복하기 위한 학습도 필요하므로, 트레이너가 없는
`birefnet`과 `feynobg`에서는 거부됩니다.

각 계열은 레시피와 무관하게 일부 모듈을 float으로 유지합니다: 첫 레이어,
예측 헤드, 그리고 YOLOv9에서는 양자화하면 안 되는 고정 적분 기댓값 연산자인
DFL 합성곱입니다.

### 캘리브레이션 데이터는 학습 데이터가 아닙니다

`calib`은 활성화 범위를 구하기 위해 순전파에만 사용하는 작고 레이블 없는 이미지
집합을 가리킵니다. 이 데이터로 평가하지 않으며 레이블은 전혀 읽지 않습니다.
기본값인 `coco128.yaml`은 처음 사용할 때 URL에서 내려받으므로 추가 권한이 필요
없지만, Python 다운로드 스크립트가 내장된 YAML은 `allow_download_scripts=true`가
필요합니다.

`algorithm=percentile`을 사용할 수 있지만 트랜스포머 계열에서 정확도를 떨어뜨릴
수 있고, 그래서 `auto`는 minmax를 선택합니다.

### 정확도 회복

출력은 일반적인 PyTorch 체크포인트이므로
[`libreyolo train`](/docs/cli/train)이 그대로 받습니다. 양자화된 체크포인트를
학습하는 것이 양자화 인식 학습이며, `distill_model=<teacher>`를 추가하면 양자화
인식 증류가 됩니다.

### 출력과 종료 코드

결과에는 저장된 경로, 레시피, 실행 모드, 캘리브레이션 수행 여부, 종류별로 교체된
모듈 수가 출력됩니다. 종료 코드는 성공 시 `0`, 모델을 불러올 수 없을 때 `4`,
양자화나 저장이 실패할 때 `5`, 그 외 런타임 실패는 `1`입니다.

관련 항목: PyTorch를 벗어나 배포용 산출물을 대신 쓰는
[`libreyolo export`](/docs/cli/export).
