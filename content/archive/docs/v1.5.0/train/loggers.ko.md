---
title: 실험 기록 장치
seo_title: LibreYOLO에서 실험 로거와 콜백
description: >-
  학습 지표를 TensorBoard, MLflow, Weights & Biases, Comet, ClearML, Neptune 또는
  DVCLive에 전송하고, 네 가지 학습 훅에 대해 직접 콜백을 작성하십시오.
lead: >-
  모든 학습 가능한 계열는 네 개의 학습 이벤트를 발생시킵니다. 내장 로거는 동일한 이벤트를 수신하는 콜백 객체이므로, 백엔드 통합과 사용자
  정의 훅은 하나의 인터페이스를 사용합니다.
keywords:
  - 텐서보드 학습
  - mlflow 추적
  - 가중치와 편향
  - 클리어엠엘
  - 코멧 ML
  - 해왕성
  - dvclive
  - 학습 콜백
  - 학습 지표 CSV
  - 리브리욜로 모니터
last_verified: 1.5.0
snippets:
  logger:
    - label: 이름으로
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(data="coco8.yaml", epochs=10, loggers="tensorboard")
    - label: 구성된 인스턴스
      language: python
      code: |
        from libreyolo import LibreYOLO
        from libreyolo.training import MLflowLogger

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(
            data="coco8.yaml",
            epochs=10,
            loggers=[MLflowLogger(tracking_uri="sqlite:///mlflow.db"), "tensorboard"],
        )
  callback:
    - label: 일반 함수
      language: python
      code: |
        from libreyolo import LibreYOLO
        from libreyolo.training import TrainEpochEvent


        def on_epoch(event: TrainEpochEvent) -> None:
            print(f"epoch {event.epoch}/{event.total_epochs} loss={event.train_loss:.4f}")


        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(data="coco8.yaml", epochs=10, callbacks=on_epoch)
    - label: 여러 개의 갈고리가 있는 물체
      language: python
      code: >
        from libreyolo import LibreYOLO

        from libreyolo.training import TrainEndEvent, TrainEpochEvent,
        TrainStartEvent



        class RunLog:
            def on_train_start(self, event: TrainStartEvent) -> None:
                print(f"{event.model_family}{event.model_size} -> {event.save_dir}")

            def on_train_epoch_end(self, event: TrainEpochEvent) -> None:
                if event.is_best:
                    print(f"new best at epoch {event.epoch}: {event.best_metric}")

            def on_train_end(self, event: TrainEndEvent) -> None:
                print(f"done in {event.total_seconds:.0f}s")


        model = LibreYOLO("LibreYOLO9s.pt")

        model.train(data="coco8.yaml", epochs=10, callbacks=RunLog())
  monitor:
    - label: 브라우저에서 실행 보기
      language: bash
      code: |
        libreyolo monitor                     # runs/에서 가장 최근 실행
        libreyolo monitor runs/train/exp      # 특정 실행
source_hash: de035acbaed32804
---

## 로거를 켜다

`loggers=`는 등록된 이름, 구성된 인스턴스 또는 둘 다를 혼합한 반복 가능한 항목을 사용합니다.

<code-tabs name="logger" />

이름은 대소문자를 구분하지 않습니다. 등록된 세트는 `tensorboard`, `mlflow`, `wandb`, `comet`, `clearml`, `neptune`, `dvclive` 및 `dvc`이며, 마지막 이름은 `dvclive`의 별칭입니다. 다른 이름을 사용하면 즉시 오류가 발생하고 유효한 이름이 나열됩니다. 모든 이름을 활성화하는 값은 없으며, CLI 플래그도 없습니다: `loggers=`는 Python 인수입니다.

## 모든 백엔드가 기록하는 것

모두 같은 메트릭 이름을 작성하므로, 어떤 것을 선택해도 대시보드는 동일하게 보입니다:

| 열쇠 | 값 |
|---|---|
| `train/loss` | 에포크의 평균 학습 손실 |
| `train/loss/<component>` | 계열이 보고하는 각 손실 요소 |
| `lr/<group>` | 각 옵티마이저 매개변수 그룹의 학습률 |
| `val/<metric>` | 각 검증 지표, `metrics/` 접두사가 제거된 상태로 |
| `time/epoch_seconds` | 시대를 위한 벽시계 |

스텝은 1부터 시작하는 에폭입니다. 완전히 해석된 학습 구성은 학습 시작 시 매개변수로 기록되며, 실행 이름은 기본적으로 `<family><size>-<task>`이며 예를 들어 `yolo9s-detect`입니다.

열차가 끝날 때, 존재하는 경우 아티팩트 업로드를 지원하는 백엔드 `results.csv`, `train_config.yaml` 및 `summary.json`가 업로드하며, `weights/best.pt`와 `log_checkpoints=True`도 업로드됩니다. TensorBoard는 아티팩트 개념이 없기 때문에 아무것도 업로드하지 않습니다. 어떤 로거도 검증 플롯 이미지를 업로드하지 않습니다.

## 실패 동작

백엔드 패키지가 없으면 생성 시 설치 명령어를 명시하며 오류를 발생시킵니다. 로거를 요청하고 아무것도 받지 못하면 버그가 숨겨지기 때문입니다.

실행 중 백엔드 실패는 그 반대입니다. 핸들러에서 발생한 첫 번째 예외는 해당 로거를 나머지 실행 동안 비활성화하고, 이를 기록하며, 백엔드 실행을 실패로 종료시키지만, 학습은 계속 진행됩니다. 트래킹 서버가 다운되더라도 학습에는 영향을 주지 않습니다.

## 백엔드

각각은 자체적인 추가가 필요합니다.

| 이름 | 추가 | 생성자 |
|---|---|---|
| `tensorboard` | `libreyolo[tensorboard]` | `TensorBoardLogger(log_dir=None)` |
| `mlflow` | `libreyolo[mlflow]` | `MLflowLogger(tracking_uri, experiment_name, run_name, log_artifacts=True, log_checkpoints=False)` |
| `wandb` | `libreyolo[wandb]` | `WandbLogger(project, name, entity, log_checkpoints=False)` |
| `comet` | `libreyolo[comet]` | `CometLogger(project_name, workspace, name, api_key, online, log_artifacts=True, log_checkpoints=False)` |
| `clearml` | `libreyolo[clearml]` | `ClearMLLogger(project_name="LibreYOLO", task_name, tags, output_uri, log_artifacts=True, log_checkpoints=False)` |
| `neptune` | `libreyolo[neptune]` | `NeptuneLogger(project, api_token, name, run_id, tags, mode, capture_console=False, log_artifacts=True, log_checkpoints=False)` |
| `dvclive`, `dvc` | `libreyolo[dvclive]` | `DVCLiveLogger(log_dir, resume, report, save_dvc_exp=False, dvcyaml=None, monitor_system=False, log_checkpoints=False)` |

`libreyolo.training`에서 클래스를 가져오십시오.

첫 실행 전에 알아두면 좋은 백엔드 관련 참고 사항:

TensorBoard 이벤트 파일은 기본적으로 `<save_dir>/tensorboard`로 설정됩니다. `tensorboard --logdir runs/train`로 확인하십시오.

MLflow 3.x는 로컬 `./mlruns` 파일 스토어를 더 이상 사용하지 않으며 `MLFLOW_ALLOW_FILE_STORE=true`가 아닌 경우 예외를 발생시킵니다. 서버 없는 로컬 추적을 위해서는 위 스니펫과 같이 데이터베이스 URI를 전달하고 `mlflow ui --backend-store-uri sqlite:///mlflow.db`로 읽으십시오.

Weights & Biases는 `WANDB_PROJECT` 환경 변수로 대체되고 그 다음 `libreyolo`로 대체됩니다. Comet은 `COMET_PROJECT_NAME`로 대체되고 그 다음 `libreyolo`로 대체되며, 자체 구성에서 인증 정보를 가져옵니다; `online=False`는 오프라인 실험을 제공합니다. ClearML은 새로운 작업을 생성하고 `TrainConfig`에서 구성 정보를 보고하며, 지표가 두 번 보고되지 않도록 자동 프레임워크 캡처를 비활성화합니다. Neptune은 이전 패키지 대신 현재의 `neptune-scale` 클라이언트를 사용하며, `mode="offline"`는 로컬에서 로그를 기록합니다.

DVCLive는 `<save_dir>/dvclive`에 씁니다. 그것은 `/`에서 요약 트리를 구성하며, 부모이기도 한 경로에는 부동 소수를 저장할 수 없으므로 `train/loss/box`는 `train/loss.box`로 작성되고 `train/loss`는 이름을 유지합니다. LibreYOLO는 또한 DVCLive의 일반 기본값인 DVC 실험 저장 및 루트 `dvc.yaml` 작성 기능을 끄므로, 선택적 로거는 실행 디렉터리 외부에 버전 관리 상태를 생성하지 않습니다; `save_dvc_exp=True` 또는 명시적인 `dvcyaml=`를 전달하면 다시 가져올 수 있습니다.

해왕성은 `libreyolo[all]`에서 의도적으로 제외되었습니다: 안정적인 클라이언트는 7 미만의 protobuf를 요구하는 반면, TFLite 추가 기능은 protobuf 7을 요구합니다. TFLite 추가 기능이 없는 환경에 `libreyolo[neptune]`를 설치하십시오.

## 콜백 작성

같은 네 가지 이벤트가 모든 동작을 이끕니다.

<code-tabs name="callback" />

| 이벤트 | 언제 | 운반하다 |
|---|---|---|
| `TrainStartEvent` | 설정 후, 에포크 1 이전 | `start_epoch`, `total_epochs`, `model_family`, `model_size`, `task`, `save_dir`, `config` |
| `TrainEpochEvent` | 각 에포크 후, 학습과 검증 | `epoch`, `train_loss`, `train_loss_items`, `lr`, `val_metrics`, `validated`, `is_best`, `current_metric`, `best_metric`, `best_epoch`, `epoch_seconds` |
| `TrainEndEvent` | 학습이 완료된 후 | `completed_epochs`, `final_loss`, `best_metric`, `best_epoch`, `total_seconds`, `results` |
| `TrainExceptionEvent` | 학습이 증가하면 | `epoch`, `exception`, `exception_type`, `exception_message`, `elapsed_seconds` |

일반 호출 가능한 객체는 `TrainEpochEvent`만 받습니다. 객체는 `on_train_start`, `on_train_epoch_end`, `on_train_end` 및 `on_train_exception`의 임의의 부분 집합을 구현할 수 있습니다. 누락된 메서드는 건너뜁니다.

`TrainStartEvent.config`는 완전히 해석된 구성으로, 사용자 kwargs가 계열 기본값과 병합된 읽기 전용 매핑입니다. 이벤트는 동결된 데이터 클래스이며 해당 매핑도 읽기 전용이므로, 콜백이 하나에 쓰기를 통해 실행을 변경할 수 없습니다.

`on_train_start`, `on_train_epoch_end` 또는 `on_train_end`에서 발생한 예외는 전파되어 실행을 종료합니다. 오직 `on_train_exception`만 보호되므로 원래 실패를 숨길 수 없습니다.

멀티 GPU 학습에서는 콜백이 오직 랭크 0에서만 실행됩니다. 자동 DDP 스폰을 사용할 경우, 콜백은 피클 가능해야 하며, 이는 클로저나 람다보다는 모듈 수준의 클래스나 함수를 의미합니다. 자세한 내용은 [멀티 GPU 학습](/docs/train/multi-gpu)을 참조하십시오.

## 어쨌든 모든 실행이 쓰는 것

모든 계열에서 구성 없이 세 개의 파일이 실행 디렉토리에 들어옵니다:

| 파일 | 작성된 | 내용 |
|---|---|---|
| `status.json` | 원자적으로, 모든 에포크와 시작, 종료 및 실패 시 | `running`, `completed` 또는 `failed`, `current_epoch`, `total_epochs`, `progress`, `eta_seconds`, 최신 `metrics`, `best_metric`, `best_epoch`의 `state` 및 실패 시 `error` 객체 |
| `metrics.jsonl` | 에포크당 한 번 추가됨 | 에포크당 하나의 JSON 행, `results.csv`와 동일한 스키마 |
| `train.log` | 살다 | 실행의 콘솔 출력 |

`status.json`는 스크립트나 에이전트가 실행을 폴링할 때 저렴하게 읽는 방법이고, 원자적 쓰기는 읽는 사람이 절반만 기록된 파일을 절대 보지 않음을 의미합니다.

`results.csv`와 `summary.json`는 별도로 존재하며 계열 단위로 구분됩니다. 이들은 YOLOv9, YOLOv9-E2E, YOLOv9-P2, YOLOv7, YOLO-NAS, RF-DETR, EC 및 DINOv2용으로 작성되었으며, 다른 계열용이 아닙니다. `results.csv`는 손실 구성 요소, 검증 지표 및 학습률을 열로 하여 각 에포크마다 한 행을 기록하고, 새로운 열이 나타나면 헤더가 확장됩니다. 재개 시점의 이력(resume)에서는 같은 행을 중복하지 않고 재개된 에포크 이전의 행으로만 잘립니다.

그와 함께, 트레이너는 항상 설정 시 `train_config.yaml`를 쓰고 `weights/` 아래에 체크포인트를 기록합니다.

## 달리기를 생중계로 시청하기

<code-tabs name="monitor" />

`libreyolo monitor`는 위의 파일들에 대해 브라우저 대시보드를 제공하며 표준 라이브러리만 사용합니다: 메트릭 차트, 로그 꼬리, 그리고 모든 검증 이미지들을 포함하며, 실행 중일 때 자동으로 새로 고침됩니다. 이는 읽기 전용이며 학습 과정에는 절대 영향을 주지 않으므로, 실행 중인 작업에 연결하거나 완료된 작업을 다시 열거나 크래시된 작업을 검사할 수 있습니다.

## 관련된

- [검증 및 지표](/docs/train/validation) `val/` 키가 의미하는 것과 검증 손실을 추가하는 방법에 대하여.
- 프로파일러의 [학습 성과](/docs/train/performance)는 다른 질문을 가진 다른 도구입니다.
