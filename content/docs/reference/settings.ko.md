---
title: 설정
seo_title: LibreYOLO 환경 변수 및 디렉토리
description: 'LibreYOLO가 읽는 모든 환경 변수, 그가 쓰는 디렉토리, 필요한 토큰 및 어떤 코드 경로가 실행되는지를 바꾸는 토글.'
lead: >-
  LibreYOLO에는 구성 파일이 없습니다. 함수 인수가 아닌 동작은 환경 변수와 소수의 관례적인 디렉토리로 제어되며, 여기 모두 나열되어
  있습니다.
keywords:
  - LIBREYOLO_DATASETS_DIR
  - LIBREYOLO_KERNELS
  - LIBREYOLO_FASTER_COCO_EVAL
  - HF_TOKEN
  - libreyolo 가중치 디렉토리
  - libreyolo 캐시
last_verified: 1.5.0
verification: >-
  v1.5.0에서 os.environ과 os.getenv를 검색하여 찾은 변수; 각 사용 시점에서 의미 읽기. 디렉토리 관례는
  libreyolo/data/utils.py, libreyolo/utils/download.py,
  libreyolo/export/exporter.py, libreyolo/models/base/model.py 및
  libreyolo/models/sam3dbody/mhr_body.py.에서 읽음
snippets:
  usage:
    - label: 데이터셋 루트를 다른 곳으로 지정하세요
      language: bash
      code: |
        export LIBREYOLO_DATASETS_DIR=/data/datasets
        python -c "from libreyolo.data import DATASETS_DIR; print(DATASETS_DIR)"
    - label: Python에서 해결된 값을 읽습니다
      language: python
      code: |
        from libreyolo.data import DATASETS_DIR

        # 기본값은 ~/datasets입니다; LIBREYOLO_DATASETS_DIR이 가져오기 시점에 이를 덮어씁니다.
        print(DATASETS_DIR)
source_hash: 462f1288582225ce
---

## 환경 변수

| 변수 | 기본값 | 효과 |
|---|---|---|
| `LIBREYOLO_DATASETS_DIR` | `~/datasets` | 데이터셋 루트. 가져오기 시 한 번 읽어 `libreyolo.data.DATASETS_DIR`에 저장 |
| `LIBREYOLO_FASTER_COCO_EVAL` | 설정되지 않음 | `faster_coco_eval` 검증 플래그를 덮어씁니다. `1`, `true`, `yes` 또는 `on`는 더 빠른 백엔드를 강제로 켭니다, 다른 값은 강제로 끄며, 설정되지 않으면 구성 플래그를 따릅니다 |
| `LIBREYOLO_KERNELS` | 설정되지 않음 | 커널 선택. `off` 또는 `reference`는 참조 구현을 강제로 사용; 다른 값은 해당 이름으로 등록된 구현만 선택 |
| `LIBREYOLO_QUANT_KERNELS` | 설정되지 않음 | `LIBREYOLO_KERNELS`의 이전 별칭으로, 해당 값이 설정되지 않은 경우에만 읽음 |
| `LIBREYOLO_HUB_KERNELS` | 설정되지 않음 | `0`, `false`, `off` 또는 `no`는 Hugging Face Hub 커널 로딩을 비활성화합니다. 설정되지 않은 값을 포함한 다른 모든 값은 활성 상태로 둡니다 |
| `LIBREYOLO_MHR_PATH` | `~/.cache/libreyolo/mhr/mhr_model.pt` | `mesh` 작업에서 사용하는 MHR 바디 모델의 위치 |
| `LIBRELABEL_ENABLE_LOCATE` | 설정되지 않음 | 레이블링 도구에서 LocateAnything 어시스턴트를 노출하려면 정확히 `1`, `true`, `yes` 또는 `on`이어야 합니다. 다른 값은 비활성 상태로 둡니다 |
| `SAM_3D_BODY_PATH` | 설정되지 않음 | 생성자에게 전달되지 않은 경우, 메시 패밀리용 SAM 3D Body 패키지 경로 |
| `HF_TOKEN` | 설정되지 않음 | Hugging Face 액세스 토큰, 제한된 저장소에 사용됨 |

<code-tabs name="usage" />

`LIBREYOLO_DATASETS_DIR`는 가져오기 시점에 읽히므로 `libreyolo.data`를 가져온 후 설정해도 `DATASETS_DIR`에는 영향이 없음.

Hub 커널은 두 부분으로 된 선택적 참여(opt-in)입니다. 런타임 가져오기는 선택적 `kernels` 패키지가 설치된 경우에만 발생하므로 `libreyolo[hub-kernels]` 설치는 선택적 참여이며 `LIBREYOLO_HUB_KERNELS=0`는 참여 거부(opt-out)입니다. 추가 패키지 없이 설치할 경우에는 영향이 없음.

커널 선택은 가져오기(import)도 단축시킵니다: `LIBREYOLO_KERNELS`가 `off` 또는 `reference`를 강제로 설정하면 트리 내 가속 제공자는 전혀 가져오지 않습니다. 이 세 변수가 제어하는 레지스트리는 [커널](/docs/reference/kernels) 문서에 있음.

## 라이브러리가 설정하는 변수들

이 변수들은 읽는 것이 아니라 쓰기 때문에, 수동으로 설정하는 것은 지원되는 방법이 아님.

| 변수 | 설정자 |
|---|---|
| `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT` | DDP 스폰 헬퍼, 각 워커 프로세스마다 하나의 값 |
| `CUDA_VISIBLE_DEVICES` | 분산 설정 동안 일시적으로 좁혀졌다가 그 후 복원됨 |
| `PYTORCH_ENABLE_MPS_FALLBACK` | EC 트레이너에 의해 `1`로 설정되며, `setdefault`가 있어서 기존 값이 우선됨 |
| `MOMENTUM_ENABLED` | 메시 패밀리 로더에 의해 `setdefault`로 설정됨 |

`LOCAL_RANK`는 분산 모드 신호 역할도 함: 환경에 존재함으로써 학습 코드가 DDP 하에서 실행 중임을 감지함

## 로거 변수

선택적인 학습 로거는 프로젝트 이름에 대해 환경 기본값을 사용함

| 변수 | 기본값 | 사용처 |
|---|---|---|
| `WANDB_PROJECT` | `libreyolo` | 프로젝트가 전달되지 않았을 때 Weights and Biases 로거 |
| `COMET_PROJECT_NAME` | `libreyolo` | 프로젝트가 전달되지 않았을 때 Comet 로거 |

이러한 서비스에 대한 인증은 LibreYOLO가 아닌 각 서비스의 도구를 따릅니다.

## 토큰

`HF_TOKEN`는 Hugging Face 접근 토큰입니다. 설정되지 않은 경우, 토큰은 Hugging Face CLI 로그인 시 작성되는 `~/.cache/huggingface/token`에서 읽습니다. 어느 경로든 작동합니다.

토큰은 공개되지 않은 저장소에서만 필요합니다. SAM 3은 제공된 예제입니다: 그 가중치는 사용자 정의 라이선스 하에 공개되지 않은 저장소에서 다운로드되므로, 저장소 페이지에서 조건을 수락하고 세션을 인증해야 합니다.

## 디렉토리

| 경로 | 내용 |
|---|---|
| `weights/` | 체크포인트 다운로드, Hugging Face 스냅샷 다운로드 및 내보낸 아티팩트 |
| `~/datasets` | 데이터셋 루트, 단 `LIBREYOLO_DATASETS_DIR`에서 달리 명시하지 않는 경우 |
| `~/.cache/huggingface/token` | Hugging Face 토큰, `HF_TOKEN`에 없는 경우 |
| `~/.cache/libreyolo/mhr/mhr_model.pt` | MHR 본체 모델, 단 `LIBREYOLO_MHR_PATH`에서 달리 명시하지 않는 경우 |
| `runs/track/` | `model.track(save=True)`의 기본 출력 |

`weights/`는 작업 디렉토리를 기준으로 합니다. 단순 파일 이름은 이를 통해 확인되므로 `LibreYOLO("LibreYOLO9t.pt")`는 `weights/LibreYOLO9t.pt`를 찾고 없으면 해당 위치에 다운로드합니다. `model.export()`는 `output_path`가 주어지지 않으면 동일한 디렉토리에 작성합니다. 형제 계층은 멀티 파일 스냅샷을 `weights/<Prefix><size>/`로 다운로드합니다.

## 다운로드 동작

가중치 다운로드는 세 번 재시도되며, 백오프(backoff)가 적용되고, 부분 파일에서 이어받기(resume)가 가능하며, 두 프로세스가 동시에 같은 체크포인트를 가져오지 않도록 잠금 파일로 보호됩니다. 제3자 호스트에서 가져오는 패밀리는 체크섬을 고정(pin)하고 불일치 시 실패(fail closed)할 수 있습니다.

일부 다운로드는 시작 전에 라이선스 공지를 출력합니다. 이러한 공지는 다운로드 경로의 일부이며 설정을 통해 억제할 수 없습니다.

## 검증 백엔드

`model.val()`는 기본적으로 `faster_coco_eval=True`를 허용하며 패키지가 설치되어 있지 않은 경우 pycocotools로 대체(fallback)되며 한 번 경고를 출력합니다. `LIBREYOLO_FASTER_COCO_EVAL`를 설정하면 호출별 플래그를 무시하고 덮어쓸 수 있으며, 이는 각 실행 설정을 건드릴 수 없는 벤치마크 하니스가 사용해야 하는 방법입니다. 실제로 실행된 백엔드는 `model.last_eval_backend`에 보고됩니다.

## 데이터셋 다운로드 스크립트

데이터셋 YAML은 Python을 포함할 수 있는 `download` 필드를 가질 수 있습니다. 이는 `allow_download_scripts=True`가 이를 읽는 호출에 전달되지 않는 한 실행되지 않으며, 이는 환경 변수가 아니라 `val()`와 `export()`의 함수 인수입니다.
