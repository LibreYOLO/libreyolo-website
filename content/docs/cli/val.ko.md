---
title: libreyolo val
seo_title: libreyolo val 명령 레퍼런스
description: '명령줄에서 데이터셋 분할(split)에 대해 체크포인트를 평가합니다: 모든 인자와 기본값, 그리고 태스크별로 반환되는 지표 키.'
lead: >-
  모델 하나를 데이터셋 분할 하나에 대해 평가하고 지표를 출력합니다. 지표 집합은 모델의 태스크에 따라 달라지며, 이 숫자들이 벤치마크 표의
  한 행을 이루는 값입니다.
keywords:
  - libreyolo val 명령어
  - libreyolo 검증 cli
  - yolo 모델 성능 평가 명령어
  - mAP50-95 측정
  - libreyolo val 옵션
last_verified: 1.5.0
meta:
  - label: 명령
    value: libreyolo val
    mono: true
  - label: 필수
    value: 'model, data'
    mono: true
  - label: 출력
    value: 지표는 stdout으로 출력됩니다. 지정하면 runs/val/exp 아래에 플롯과 COCO JSON이 저장됩니다
snippets:
  examples:
    - label: 기본
      language: bash
      code: |
        libreyolo val model=LibreYOLO9s.pt data=coco8.yaml
    - label: 플롯과 COCO JSON
      language: bash
      code: |
        libreyolo val model=LibreYOLO9s.pt data=coco8.yaml \
          imgsz=640 batch=8 save_json=true save_plots=true \
          project=runs/val name=yolo9s-coco8 exist_ok=true
    - label: 기계 판독용
      language: bash
      code: |
        libreyolo val model=LibreYOLO9s.pt data=coco8.yaml json=true quiet=true
source_hash: f6507840568c3725
---

## 개요

```bash
libreyolo val model=<name|path> data=<dataset.yaml> [key=value ...]
```

인자는 `key=value` 쌍이며, POSIX 형식도 동작하므로 `batch=8`과 `--batch 8`은 같은
인자입니다.

## 인자

| 인자 | 기본값 | 의미 |
|---|---|---|
| `model` | | 모델 가중치 경로 또는 CLI 이름. 필수 |
| `data` | | 데이터셋 YAML 경로(YOLO 형식, 예: `coco8.yaml`). 필수 |
| `data_dir` | | YAML에 적힌 경로를 건너뛰고 지정하는 데이터셋 디렉터리 |
| `split` | `val` | 데이터셋 분할: `val`, `test`, `train` |
| `batch` | `16` | 배치 크기 |
| `imgsz` | | 이미지 크기: `640`(정사각형) 또는 `480x640`(HxW). 설정하지 않으면 모델 자체의 입력 크기 |
| `conf` | `0.001` | 신뢰도 임계값 |
| `iou` | `0.6` | NMS IoU 임계값 |
| `max_det` | `300` | NMS 이후 이미지당 최대 예측 수 |
| `eval_max_det` | | COCO 평가기 상한. 설정하지 않으면 pycocotools의 AP@100 관례 |
| `faster_coco_eval` | `true` | 설치되어 있으면 COCO 지표에 faster-coco-eval C++ 백엔드 사용; 없으면 pycocotools로 대체 |
| `half` | `false` | FP16 추론 |
| `amp_dtype` | `float16` | `half=true`일 때 CUDA autocast dtype: `float16` 또는 `bfloat16` |
| `save_json` | `false` | COCO 형식 JSON 결과 저장 |
| `save_plots` | `false` | 검증 플롯 저장: 지표, 클래스별 AP, 혼동 행렬, 샘플 |
| `workers` | `4` | 데이터로더 워커 수 |
| `device` | `auto` | 장치 |
| `project` | `runs/val` | 출력 디렉터리 루트 |
| `name` | `exp` | 실험 이름 |
| `exist_ok` | `false` | 출력 디렉터리 재사용 |
| `allow_download_scripts` | `false` | 데이터셋 YAML의 download 블록에 내장된 Python 실행 허용 |
| `json` | `false` | stdout으로 JSON 출력 |
| `quiet` | `false` | stderr 출력 억제 |
| `verbose` | `true` | 상세 출력 |
| `help_json` | `false` | 명령 스키마를 JSON으로 출력하고 종료 |

## 예제

<code-tabs name="examples" />

## 참고

### 출력되는 지표

출력되는 집합은 모델의 태스크를 따르며, JSON 출력도 같은 키를 사용합니다.

탐지, 분할, 회전 바운딩 박스는 `mAP50`, `mAP50_95`, `precision`, `recall`을
보고합니다. 모델이 한 가지가 넘는 출력 종류를 예측하는 경우, 종류별 그룹이
`box_metrics`, `mask_metrics`, `obb_metrics`로 나란히 나타나며 각각 같은 네 개의
키를 담습니다.

분류는 `accuracy_top1`과 `accuracy_top5`를 보고합니다. 점 탐지는 `precision`,
`recall`, `f1`, `MLE`, `MAE`, `RMSE`, `mAP_sweep`을 보고합니다. 깊이 추정은
`abs_rel`, `rmse`, `delta1`, `delta2`, `delta3`를 보고합니다. 시맨틱 분할은
`mIoU`와 `pixel_accuracy`를 보고합니다. 복원은 `PSNR`과 `SSIM`을 보고합니다.

JSON 결과에는 `eval_backend`도 함께 담겨서 그 숫자를 만들어 낸 COCO 평가
라이브러리와 버전을 알려 주며, 덕분에 두 실행을 비교할 때 같은 백엔드가 양쪽을
채점했는지 알고 비교할 수 있습니다.

### 임계값

여기의 기본값은 예측 기본값이 아니라 평가 기본값입니다: `conf`는 `0.001`,
`iou`는 `0.6`이며, [`libreyolo predict`](/docs/cli/predict)는 `0.25`와 `0.45`를
사용합니다. `conf`를 표시용 임계값까지 올리면 재현율이 낮아지고 그와 함께 mAP도
낮아지므로, 그렇게 얻은 수치는 공개된 수치와 비교할 수 없습니다.

`imgsz`는 기본적으로 설정되어 있지 않으며, 이는 모델 자체의 입력 크기를
뜻합니다. 값을 지정하면 주어진 크기로 평가하는데, 체크포인트를 원래 해상도가
아닌 조건에서 측정하는 방법이 바로 이것입니다.

### 자동으로 내려받는 데이터셋

`download` 필드가 URL인 데이터셋 YAML은 별도의 허가 없이 처음 사용할 때
내려받습니다. Python 다운로드 스크립트가 내장된 YAML은
`allow_download_scripts=true`가 필요하며, 이 경우 명령은 로컬 코드 실행이
활성화되었다고 stderr에 경고합니다. 함께 제공되는 `coco8.yaml`과
`coco128.yaml`은 URL 기반이므로 아무것도 필요하지 않습니다.

### 출력과 종료 코드

지표는 stdout으로 나가고, 진행 상황은 stderr로 갑니다. `json=true`는
`schema_version`이 담긴 객체 하나를 출력하고, `quiet=true`는 stderr 출력을
끕니다.

종료 코드는 성공 시 `0`, 사용법이나 설정 오류는 `2`, 데이터셋을 찾을 수 없으면
`3`, 모델을 불러올 수 없으면 `4`, 그 외 런타임 실패는 `1`입니다.

관련 문서: [`libreyolo train`](/docs/cli/train)은 `eval_interval`로 자체 일정에
따라 이와 동일한 평가를 실행합니다.
