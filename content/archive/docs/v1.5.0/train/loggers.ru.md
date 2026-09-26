---
title: Логгеры экспериментов
seo_title: Логгеры экспериментов и колбэки в LibreYOLO
description: >-
  Отправляйте метрики обучения в TensorBoard, MLflow, Weights & Biases, Comet,
  ClearML, Neptune или DVCLive и пишите свой колбэк на четырёх хуках обучения.
lead: >-
  Каждое обучаемое семейство порождает четыре события обучения. Встроенные
  логгеры — это объекты-колбэки, которые слушают те же самые события, поэтому
  интеграция с бэкендом и собственный хук работают через один интерфейс.
keywords:
  - tensorboard обучение
  - mlflow tracking
  - weights and biases
  - clearml
  - comet ml
  - neptune
  - dvclive
  - колбэки обучения
  - метрики обучения csv
  - libreyolo monitor
last_verified: 1.5.0
snippets:
  logger:
    - label: По имени
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(data="coco8.yaml", epochs=10, loggers="tensorboard")
    - label: Настроенный экземпляр
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
    - label: Обычная функция
      language: python
      code: |
        from libreyolo import LibreYOLO
        from libreyolo.training import TrainEpochEvent


        def on_epoch(event: TrainEpochEvent) -> None:
            print(f"epoch {event.epoch}/{event.total_epochs} loss={event.train_loss:.4f}")


        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(data="coco8.yaml", epochs=10, callbacks=on_epoch)
    - label: Объект с несколькими хуками
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
    - label: Наблюдение за запуском в браузере
      language: bash
      code: |
        libreyolo monitor                     # последний запуск в runs/
        libreyolo monitor runs/train/exp      # конкретный запуск
source_hash: de035acbaed32804
---

## Включение логгера

`loggers=` принимает зарегистрированное имя, настроенный экземпляр или
итерируемый объект, в котором есть и то и другое.

<code-tabs name="logger" />

Регистр в именах не важен. Зарегистрированный набор — `tensorboard`, `mlflow`,
`wandb`, `comet`, `clearml`, `neptune`, `dvclive` и `dvc`, где последнее —
псевдоним для `dvclive`. Всё остальное сразу вызывает ошибку со списком
допустимых имён. Значения, которое включало бы их все, нет, и CLI-флага тоже нет:
`loggers=` — это Python-аргумент.

## Что записывает каждый бэкенд

Все они пишут одни и те же имена метрик, поэтому дашборд выглядит одинаково,
какой бы бэкенд вы ни выбрали:

| Ключ | Значение |
|---|---|
| `train/loss` | среднее значение функции потерь на обучении за эпоху |
| `train/loss/<component>` | каждая компонента функции потерь, о которой сообщает семейство |
| `lr/<group>` | скорость обучения каждой группы параметров оптимизатора |
| `val/<metric>` | каждая метрика валидации с убранным префиксом `metrics/` |
| `time/epoch_seconds` | реальное время выполнения эпохи |

Шаг — это номер эпохи, начиная с 1. Итоговая конфигурация обучения со всеми
подставленными значениями логируется как параметры в начале обучения, а имя
запуска по умолчанию —
`<family><size>-<task>`, например `yolo9s-detect`.

В конце обучения бэкенды, поддерживающие артефакты, загружают `results.csv`,
`train_config.yaml` и `summary.json`, если те есть, а с `log_checkpoints=True` —
ещё и `weights/best.pt`. TensorBoard не загружает ничего, потому что в нём нет
понятия артефакта. Изображения графиков валидации не загружает ни один логгер.

## Поведение при сбоях

Отсутствие пакета бэкенда приводит к ошибке при создании логгера, с указанием
команды установки: запросить логгер и молча не получить ничего — значит скрыть
баг.

Со сбоем бэкенда по ходу запуска всё наоборот. Первое исключение из обработчика
отключает этот логгер до конца запуска, записывает его в лог, закрывает запуск
на стороне бэкенда как неуспешный, и обучение продолжается. Из-за упавшего
сервера трекинга вы не потеряете обучение.

## Бэкенды

Каждому нужен свой extra.

| Имя | Extra | Конструктор |
|---|---|---|
| `tensorboard` | `libreyolo[tensorboard]` | `TensorBoardLogger(log_dir=None)` |
| `mlflow` | `libreyolo[mlflow]` | `MLflowLogger(tracking_uri, experiment_name, run_name, log_artifacts=True, log_checkpoints=False)` |
| `wandb` | `libreyolo[wandb]` | `WandbLogger(project, name, entity, log_checkpoints=False)` |
| `comet` | `libreyolo[comet]` | `CometLogger(project_name, workspace, name, api_key, online, log_artifacts=True, log_checkpoints=False)` |
| `clearml` | `libreyolo[clearml]` | `ClearMLLogger(project_name="LibreYOLO", task_name, tags, output_uri, log_artifacts=True, log_checkpoints=False)` |
| `neptune` | `libreyolo[neptune]` | `NeptuneLogger(project, api_token, name, run_id, tags, mode, capture_console=False, log_artifacts=True, log_checkpoints=False)` |
| `dvclive`, `dvc` | `libreyolo[dvclive]` | `DVCLiveLogger(log_dir, resume, report, save_dvc_exp=False, dvcyaml=None, monitor_system=False, log_checkpoints=False)` |

Классы импортируются из `libreyolo.training`.

Что стоит знать об отдельных бэкендах до первого запуска:

Файлы событий TensorBoard по умолчанию пишутся в `<save_dir>/tensorboard`.
Смотреть их командой `tensorboard --logdir runs/train`.

MLflow 3.x объявил локальное файловое хранилище `./mlruns` устаревшим и падает
с ошибкой, если не задать `MLFLOW_ALLOW_FILE_STORE=true`. Для локального
трекинга без сервера передайте вместо этого URI базы данных, как в сниппете
выше, и читайте её командой `mlflow ui --backend-store-uri sqlite:///mlflow.db`.

Weights & Biases при отсутствии явного значения берёт переменную окружения
`WANDB_PROJECT`, а затем `libreyolo`. Comet берёт `COMET_PROJECT_NAME`, затем
`libreyolo`, а учётные данные — из собственной конфигурации; `online=False` даёт
офлайн-эксперимент. ClearML создаёт новую задачу, публикует конфигурацию под
`TrainConfig` и отключает автоматический перехват фреймворка, чтобы метрики не
отправлялись дважды. Neptune использует актуальный клиент `neptune-scale`, а не
устаревший пакет, и с `mode="offline"` пишет логи локально.

DVCLive пишет в `<save_dir>/dvclive`. Он строит дерево сводки по `/` и не может
хранить число с плавающей точкой по пути, который одновременно служит
родительским, поэтому `train/loss/box` записывается как `train/loss.box`, а
`train/loss` сохраняет своё имя. LibreYOLO также отключает обычные умолчания
DVCLive — сохранение DVC-эксперимента и запись корневого `dvc.yaml`, — чтобы
явно включённый логгер не создавал состояние в системе контроля версий за
пределами каталога запуска; чтобы вернуть их, передайте `save_dvc_exp=True` или
явный `dvcyaml=`.

Neptune намеренно не входит в `libreyolo[all]`: его стабильному клиенту нужен
protobuf ниже 7, а extra для TFLite требует protobuf 7. Ставьте
`libreyolo[neptune]` в окружении без extra для TFLite.

## Написание колбэка

Всё держится на тех же четырёх событиях.

<code-tabs name="callback" />

| Событие | Когда | Что несёт |
|---|---|---|
| `TrainStartEvent` | после подготовки, до первой эпохи | `start_epoch`, `total_epochs`, `model_family`, `model_size`, `task`, `save_dir`, `config` |
| `TrainEpochEvent` | после каждой эпохи — обучения и валидации | `epoch`, `train_loss`, `train_loss_items`, `lr`, `val_metrics`, `validated`, `is_best`, `current_metric`, `best_metric`, `best_epoch`, `epoch_seconds` |
| `TrainEndEvent` | после завершения обучения | `completed_epochs`, `final_loss`, `best_metric`, `best_epoch`, `total_seconds`, `results` |
| `TrainExceptionEvent` | если обучение падает с исключением | `epoch`, `exception`, `exception_type`, `exception_message`, `elapsed_seconds` |

Обычный вызываемый объект получает только `TrainEpochEvent`. Объект может
реализовать любое подмножество из `on_train_start`, `on_train_epoch_end`,
`on_train_end` и `on_train_exception`; отсутствующие методы пропускаются.

`TrainStartEvent.config` — это итоговая конфигурация, то есть пользовательские
kwargs, слитые с умолчаниями семейства, в виде отображения только для чтения.
События — замороженные датаклассы, а их отображения доступны
только для чтения, поэтому колбэк не может изменить запуск записью в событие.

Исключение из `on_train_start`, `on_train_epoch_end` или `on_train_end`
пробрасывается наружу и завершает запуск. Защищён только `on_train_exception` —
чтобы он не мог скрыть исходный сбой.

При обучении на нескольких GPU колбэки срабатывают только на rank 0. При
автоматическом spawn для DDP они ещё и должны сериализоваться через `pickle`,
то есть быть классом или функцией уровня модуля, а не замыканием или лямбдой. См.
[Обучение на нескольких GPU](/docs/train/multi-gpu).

## Что каждый запуск пишет в любом случае

Три файла появляются в каталоге запуска вообще без настройки, у каждого
семейства:

| Файл | Когда пишется | Содержимое |
|---|---|---|
| `status.json` | атомарно, каждую эпоху, а также на старте, в конце и при сбое | `state` со значением `running`, `completed` или `failed`, `current_epoch`, `total_epochs`, `progress`, `eta_seconds`, последние `metrics`, `best_metric`, `best_epoch`, а при сбое — объект `error` |
| `metrics.jsonl` | дописывается раз в эпоху | одна JSON-строка на эпоху, та же схема, что и у `results.csv` |
| `train.log` | в реальном времени | консольный вывод запуска |

`status.json` дёшево прочитать скрипту или агенту, который опрашивает запуск, а
атомарная запись означает, что читатель никогда не увидит недописанный файл.

`results.csv` и `summary.json` идут отдельно и зависят от семейства. Они пишутся
для YOLOv9, YOLOv9-E2E, YOLOv9-P2, YOLOv7, YOLO-NAS, RF-DETR, EC и DINOv2 и не
пишутся для остальных семейств. В `results.csv` попадает по строке на эпоху, где
колонками идут компоненты функции потерь, метрики валидации и скорости обучения,
а заголовок расширяется, когда появляется новая колонка. При возобновлении файл
обрезается до строк, идущих до возобновляемой эпохи, вместо того чтобы
дублировать их.

Кроме них, трейнер всегда пишет `train_config.yaml` на этапе подготовки и
чекпойнты в `weights/`.

## Наблюдение за запуском в реальном времени

<code-tabs name="monitor" />

`libreyolo monitor` поднимает браузерный дашборд поверх перечисленных выше
файлов средствами одной только стандартной библиотеки: графики метрик, хвост
лога и изображения валидации, если они есть, с обновлением, пока запуск активен.
Он работает только на чтение и никогда не трогает процесс обучения, поэтому
подключается к живому запуску, снова открывает завершённый или помогает
разобраться в упавшем.

## Связанные страницы

- [Валидация и метрики](/docs/train/validation) — о том, что означают ключи
  `val/` и как добавить функцию потерь на валидации.
- [Производительность обучения](/docs/train/performance) — о профилировщике: это
  другой инструмент для другого вопроса.
