---
title: 설정
seo_title: LibreYOLO 환경 변수 및 디렉토리
description: 'LibreYOLO가 읽는 모든 환경 변수, 작성하는 디렉토리, 필요한 토큰, 그리고 어떤 코드 경로를 실행할지 변경하는 토글.'
lead: >-
  LibreYOLO에는 구성 파일이 없습니다. 함수 인수가 아닌 동작은 환경 변수와 소수의 관례적인 디렉토리에 의해 제어되며, 이들은 모두
  여기 나열되어 있습니다.
keywords:
  - LIBREYOLO_데이터셋_디렉토리
  - 리브레욜로_커널
  - LIBREYOLO_빠른_COCO_평가
  - HF_토큰
  - libreyolo 가중치 디렉토리
  - libreyolo 캐시
last_verified: 1.5.0
verification: >-
  v1.5.0에서 libreyolo/**/*.py를 검색하여 os.environ 및 os.getenv로 위치한 변수; 각 사용 지점에서 의미
  읽기. 디렉토리 규칙은 libreyolo/data/utils.py, libreyolo/utils/download.py,
  libreyolo/export/exporter.py, libreyolo/models/base/model.py 및
  libreyolo/models/sam3dbody/mhr_body.py.에서 읽기
snippets:
  usage:
    - label: 데이터셋 루트를 다른 곳으로 지정하십시오
      language: bash
      code: |
        export LIBREYOLO_DATASETS_DIR=/data/datasets
        python -c "from libreyolo.data import DATASETS_DIR; print(DATASETS_DIR)"
    - label: Python에서 해결된 값을 읽으십시오
      language: python
      code: |
        from libreyolo.data import DATASETS_DIR

        # 기본값은 ~/datasets이며; LIBREYOLO_DATASETS_DIR가 가져오기 시점에 이를 덮어씁니다.
        print(DATASETS_DIR)
source_hash: 462f1288582225ce
---

## 환경 변수

| 변수 | 기본값 | 효과 |
|---|---|---|
| `LIBREYOLO_DATASETS_DIR` | `~/datasets` | 데이터셋 루트. 가져올 때 한 번 읽어 `libreyolo.data.DATASETS_DIR` 안으로 들어감 |
| `LIBREYOLO_FASTER_COCO_EVAL` | 설정 해제 | `faster_coco_eval` 검증 플래그를 무시합니다. `1`, `true`, `yes` 또는 `on`는 더 빠른 백엔드를 강제로 켜고, 다른 값은 강제로 끄며, 설정되지 않은 경우 구성 플래그를 따릅니다. |
| `LIBREYOLO_KERNELS` | 설정 해제 | 커널 선택. `off` 또는 `reference`는 참조 구현을 강제하며; 다른 값은 해당 이름으로 등록된 구현만 선택합니다 |
| `LIBREYOLO_QUANT_KERNELS` | 설정 해제 | `LIBREYOLO_KERNELS`의 레거시 별칭으로, 해당 항목이 설정되지 않은 경우에만 읽습니다 |
| `LIBREYOLO_HUB_KERNELS` | 설정 해제 | `0`, `false`, `off` 또는 `no`는 Hugging Face Hub 커널 로딩을 비활성화합니다. 설정되지 않은 것을 포함한 다른 값은 이를 활성 상태로 둡니다. |
| `LIBREYOLO_MHR_PATH` | `~/.cache/libreyolo/mhr/mhr_model.pt` | `mesh` 작업에 사용된 MHR 본체 모델의 위치 |
| `LIBRELABEL_ENABLE_LOCATE` | 설정 해제 | 레이블링 도구에서 LocateAnything 어시스턴트를 표시하려면 정확히 `1`, `true`, `yes` 또는 `on`이어야 합니다. 다른 값은 표시하지 않습니다. |
| `SAM_3D_BODY_PATH` | 설정 해제 | 생성자에 전달되지 않았을 때, 메시 계열를 위한 SAM 3D 본체 패키지 경로 |
| `HF_TOKEN` | 설정 해제 | 게이트된 저장소에 사용되는 Hugging Face 액세스 토큰 |

<code-tabs name="usage" />

`LIBREYOLO_DATASETS_DIR`는 가져올 때 읽히므로, `libreyolo.data`를 가져온 후에 설정해도 `DATASETS_DIR`에는 영향을 미치지 않습니다.

허브 커널은 두 부분으로 구성된 선택적 참여(opt-in)입니다. 런타임 가져오기(fetch)는 선택적 `kernels` 패키지가 설치될 때만 발생하며, 따라서 `libreyolo[hub-kernels]` 설치가 선택적 참여이며 `LIBREYOLO_HUB_KERNELS=0`가 선택적 제외(opt-out)입니다. 추가 패키지 없이 설치하는 경우에는 어떠한 방식으로도 영향을 받지 않습니다.

커널 선택은 또한 가져오기를 단축시킵니다: `LIBREYOLO_KERNELS`가 `off` 또는 `reference`를 강제로 사용할 때, 트리 내 가속 제공자는 전혀 가져오지 않습니다. 이 세 변수로 제어되는 레지스트리는 [kernels](/docs/reference/kernels)에 문서화되어 있습니다.

## 라이브러리가 설정하는 변수

이것들은 읽는 것이 아니라 쓰는 것이므로, 수동으로 설정하는 것은 지원되는 방법이 아닙니다.

| 변수 | 설정한 |
|---|---|
| `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT` | DDP 스폰 도우미, 워커 프로세스당 하나의 값 |
| `CUDA_VISIBLE_DEVICES` | 분산 설정 중 일시적으로 좁아졌다가 이후 복원됨 |
| `PYTORCH_ENABLE_MPS_FALLBACK` | EC 트레이너에 의해 `1`로 설정되고, `setdefault`와 함께 설정되어 기존 값이 우선합니다 |
| `MOMENTUM_ENABLED` | 메시 계열 로더로 `setdefault`와 함께 설정 |

`LOCAL_RANK`는 분산 모드 신호로도 사용됩니다: 환경에서 이것이 존재하는 것은 학습 코드가 DDP 하에서 실행 중임을 탐지하는 방법입니다.

## 로거 변수

선택적 학습 로거는 프로젝트 이름에 대해 환경 기본값을 사용합니다.

| 변수 | 기본값 | 사용됨 |
|---|---|---|
| `WANDB_PROJECT` | `libreyolo` | 프로젝트가 전달되지 않았을 때의 Weights and Biases 로거 |
| `COMET_PROJECT_NAME` | `libreyolo` | 프로젝트가 전달되지 않을 때 코멧 로거 |

해당 서비스들의 인증은 LibreYOLO가 아닌 자체 도구를 따릅니다.

## 토큰

`HF_TOKEN`는 Hugging Face 액세스 토큰입니다. 설정되지 않은 경우, 토큰은 Hugging Face CLI 로그인이 기록하는 `~/.cache/huggingface/token`에서 읽습니다. 어느 경로든 작동합니다.

토큰은 제한된 저장소에만 필요합니다. SAM 3이 제공된 예시입니다: 그 가중치는 사용자 지정 라이선스가 적용된 제한된 저장소에서 다운로드되므로, 저장소 페이지에서 조건을 수락하고 세션을 인증해야 합니다.

## 디렉토리

| 경로 | 내용 |
|---|---|
| `weights/` | 다운로드한 체크포인트, 다운로드한 Hugging Face 스냅샷, 내보낸 아티팩트 |
| `~/datasets` | 데이터셋 루트, `LIBREYOLO_DATASETS_DIR`가 달리 말하지 않는 한 |
| `~/.cache/huggingface/token` | Hugging Face 토큰, `HF_TOKEN`에 없을 때 |
| `~/.cache/libreyolo/mhr/mhr_model.pt` | `LIBREYOLO_MHR_PATH`가 달리 말하지 않는 한 MHR 바디 모델 |
| `runs/track/` | `model.track(save=True)`의 기본 출력 |

`weights/`는 작업 디렉토리를 기준으로 합니다. 일반 파일명은 이를 통해 해결되므로, `LibreYOLO("LibreYOLO9t.pt")`는 `weights/LibreYOLO9t.pt`를 찾고 없으면 거기에 다운로드합니다. `model.export()`는 `output_path`가 제공되지 않으면 동일한 디렉토리에 씁니다. 형제 계층은 다중 파일 스냅샷을 `weights/<Prefix><size>/`로 다운로드합니다.

## 다운로드 동작

가중치 다운로드는 백오프로 세 번 재시도되며, 부분 파일에서 재개되고, 두 프로세스가 동시에 같은 체크포인트를 가져오지 않도록 잠금 파일로 보호됩니다. 타사 호스트에서 가져오는 계열는 체크섬을 고정하고 불일치 시 실패할 수 있습니다.

일부 다운로드는 시작하기 전에 라이선스 공지를 출력합니다. 이러한 공지는 다운로드 경로의 일부이며 설정을 통해 숨길 수 없습니다.

## 검증 백엔드

`model.val()`는 기본적으로 `faster_coco_eval=True`를 수락하며, 패키지가 설치되어 있지 않은 경우 pycocotools로 대체하고 한 번 경고를 표시합니다. `LIBREYOLO_FASTER_COCO_EVAL`를 설정하면 호출당 플래그가 무시되며, 이는 실행당 설정을 건드릴 수 없는 벤치마크 하니스가 사용해야 하는 것입니다. 실제로 실행된 백엔드는 `model.last_eval_backend`에 보고됩니다.

## 데이터셋 다운로드 스크립트

데이터셋 YAML은 Python을 포함하는 `download` 필드를 가질 수 있습니다. 이는 `val()`와 `export()`에 있는 함수 인수인 호출에 `allow_download_scripts=True`가 전달되지 않는 한 실행되지 않습니다. 환경 변수는 아닙니다.
