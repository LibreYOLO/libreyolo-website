---
title: Experiment loggers
seo_title: Experiment loggers and callbacks in LibreYOLO
description: >-
  Send training metrics to TensorBoard, MLflow, Weights & Biases, Comet,
  ClearML, Neptune or DVCLive, and write your own callback on the four training
  hooks.
lead: >-
  Every trainable family emits four training events. The built-in loggers are
  callback objects listening to those same events, so a backend integration and
  a custom hook use one interface.
keywords:
  - tensorboard training
  - mlflow tracking
  - weights and biases
  - clearml
  - comet ml
  - neptune
  - dvclive
  - training callbacks
  - training metrics csv
  - libreyolo monitor
last_verified: 1.5.0
snippets:
  logger:
    - label: By name
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(data="coco8.yaml", epochs=10, loggers="tensorboard")
    - label: Configured instance
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
    - label: A plain function
      language: python
      code: |
        from libreyolo import LibreYOLO
        from libreyolo.training import TrainEpochEvent


        def on_epoch(event: TrainEpochEvent) -> None:
            print(f"epoch {event.epoch}/{event.total_epochs} loss={event.train_loss:.4f}")


        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(data="coco8.yaml", epochs=10, callbacks=on_epoch)
    - label: An object with several hooks
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
    - label: Watch a run in the browser
      language: bash
      code: |
        libreyolo monitor                     # the most recent run under runs/
        libreyolo monitor runs/train/exp      # a specific run
source_hash: de035acbaed32804
---

## Turn a logger on

`loggers=` takes a registered name, a configured instance, or an iterable mixing
both.

<code-tabs name="logger" />

Names are case-insensitive. The registered set is `tensorboard`, `mlflow`,
`wandb`, `comet`, `clearml`, `neptune`, `dvclive` and `dvc`, the last being an
alias for `dvclive`. Anything else raises immediately and lists the valid names.
There is no value that enables all of them, and there is no CLI flag: `loggers=`
is a Python argument.

## What every backend records

All of them write the same metric names, so a dashboard looks the same whichever
you pick:

| Key | Value |
|---|---|
| `train/loss` | the epoch's mean training loss |
| `train/loss/<component>` | each loss component the family reports |
| `lr/<group>` | the learning rate of each optimizer parameter group |
| `val/<metric>` | each validation metric, with its `metrics/` prefix stripped |
| `time/epoch_seconds` | wall clock for the epoch |

The step is the 1-based epoch. The fully resolved training configuration is
logged as parameters at train start, and the run name defaults to
`<family><size>-<task>`, for example `yolo9s-detect`.

At train end the backends that support artifacts upload `results.csv`,
`train_config.yaml` and `summary.json` when those exist, plus
`weights/best.pt` with `log_checkpoints=True`. TensorBoard uploads nothing,
because it has no artifact concept. No logger uploads validation plot images.

## Failure behavior

A missing backend package raises at construction, naming the install command,
because asking for a logger and silently getting nothing hides a bug.

A backend failure during the run does the opposite. The first exception from a
handler disables that logger for the rest of the run, logs it, tears the backend
run down as failed, and training continues. A tracking server going down does not
cost you the training.

## The backends

Each needs its own extra.

| Name | Extra | Constructor |
|---|---|---|
| `tensorboard` | `libreyolo[tensorboard]` | `TensorBoardLogger(log_dir=None)` |
| `mlflow` | `libreyolo[mlflow]` | `MLflowLogger(tracking_uri, experiment_name, run_name, log_artifacts=True, log_checkpoints=False)` |
| `wandb` | `libreyolo[wandb]` | `WandbLogger(project, name, entity, log_checkpoints=False)` |
| `comet` | `libreyolo[comet]` | `CometLogger(project_name, workspace, name, api_key, online, log_artifacts=True, log_checkpoints=False)` |
| `clearml` | `libreyolo[clearml]` | `ClearMLLogger(project_name="LibreYOLO", task_name, tags, output_uri, log_artifacts=True, log_checkpoints=False)` |
| `neptune` | `libreyolo[neptune]` | `NeptuneLogger(project, api_token, name, run_id, tags, mode, capture_console=False, log_artifacts=True, log_checkpoints=False)` |
| `dvclive`, `dvc` | `libreyolo[dvclive]` | `DVCLiveLogger(log_dir, resume, report, save_dvc_exp=False, dvcyaml=None, monitor_system=False, log_checkpoints=False)` |

Import the classes from `libreyolo.training`.

Backend-specific notes worth knowing before the first run:

TensorBoard event files default to `<save_dir>/tensorboard`. View with
`tensorboard --logdir runs/train`.

MLflow 3.x deprecated the local `./mlruns` file store and raises unless
`MLFLOW_ALLOW_FILE_STORE=true`. For server-less local tracking, pass a database
URI instead, as in the snippet above, and read it with
`mlflow ui --backend-store-uri sqlite:///mlflow.db`.

Weights & Biases falls back to the `WANDB_PROJECT` environment variable and then
to `libreyolo`. Comet falls back to `COMET_PROJECT_NAME` and then to `libreyolo`,
and takes credentials from its own configuration; `online=False` gives an offline
experiment. ClearML creates a fresh task, reports the config under
`TrainConfig`, and disables automatic framework capture so metrics are not
reported twice. Neptune uses the current `neptune-scale` client rather than the
legacy package, and `mode="offline"` logs locally.

DVCLive writes to `<save_dir>/dvclive`. It builds its summary tree from `/`, and
cannot hold a float at a path that is also a parent, so `train/loss/box` is
written as `train/loss.box` while `train/loss` keeps its name. LibreYOLO also
turns off DVCLive's usual defaults of saving a DVC experiment and writing a root
`dvc.yaml`, so an opt-in logger creates no version-control state outside the run
directory; pass `save_dvc_exp=True` or an explicit `dvcyaml=` to get them back.

Neptune is deliberately excluded from `libreyolo[all]`: its stable client
requires protobuf below 7 while the TFLite extra requires protobuf 7. Install
`libreyolo[neptune]` in an environment without the TFLite extra.

## Writing a callback

The same four events drive everything.

<code-tabs name="callback" />

| Event | When | Carries |
|---|---|---|
| `TrainStartEvent` | after setup, before epoch 1 | `start_epoch`, `total_epochs`, `model_family`, `model_size`, `task`, `save_dir`, `config` |
| `TrainEpochEvent` | after each epoch, training and validation | `epoch`, `train_loss`, `train_loss_items`, `lr`, `val_metrics`, `validated`, `is_best`, `current_metric`, `best_metric`, `best_epoch`, `epoch_seconds` |
| `TrainEndEvent` | after training completes | `completed_epochs`, `final_loss`, `best_metric`, `best_epoch`, `total_seconds`, `results` |
| `TrainExceptionEvent` | if training raises | `epoch`, `exception`, `exception_type`, `exception_message`, `elapsed_seconds` |

A plain callable receives `TrainEpochEvent` only. An object may implement any
subset of `on_train_start`, `on_train_epoch_end`, `on_train_end` and
`on_train_exception`; missing methods are skipped.

`TrainStartEvent.config` is the fully resolved configuration, user kwargs merged
with family defaults, as a read-only mapping. The events are frozen dataclasses
and their mappings are read-only, so a callback cannot change the run by writing
to one.

An exception raised from `on_train_start`, `on_train_epoch_end` or
`on_train_end` propagates and ends the run. Only `on_train_exception` is guarded,
so it cannot mask the original failure.

Under multi-GPU training, callbacks fire on rank 0 only. With the automatic DDP
spawn they also have to be picklable, which means a module-level class or
function rather than a closure or a lambda. See
[Multi-GPU training](/docs/train/multi-gpu).

## What every run writes anyway

Three files land in the run directory with no configuration at all, on every
family:

| File | Written | Contents |
|---|---|---|
| `status.json` | atomically, every epoch and on start, end and failure | `state` of `running`, `completed` or `failed`, `current_epoch`, `total_epochs`, `progress`, `eta_seconds`, latest `metrics`, `best_metric`, `best_epoch`, and an `error` object on failure |
| `metrics.jsonl` | appended once per epoch | one JSON row per epoch, the same schema as `results.csv` |
| `train.log` | live | the run's console output |

`status.json` is the cheap read for a script or an agent polling a run, and the
atomic write means a reader never sees a half-written file.

`results.csv` and `summary.json` are separate and family-gated. They are written
for YOLOv9, YOLOv9-E2E, YOLOv9-P2, YOLOv7, YOLO-NAS, RF-DETR, EC and DINOv2, and
not for the other families. `results.csv` gets one row per epoch with the loss
components, validation metrics and learning rates as columns, and its header
widens when a new column appears. On a resume it is trimmed back to the rows
before the resumed epoch rather than duplicating them.

Alongside those, the trainer always writes `train_config.yaml` at setup and the
checkpoints under `weights/`.

## Watch a run live

<code-tabs name="monitor" />

`libreyolo monitor` serves a browser dashboard over the files above using only
the standard library: metric charts, the log tail, and any validation images,
refreshing while the run is active. It is read-only and never touches the
training process, so it attaches to a live run, reopens a finished one, or
inspects a crashed one.

## Related

- [Validation and metrics](/docs/train/validation) for what the `val/` keys mean
  and how to add a validation loss.
- [Training performance](/docs/train/performance) for the profiler, which is a
  different tool with a different question.


