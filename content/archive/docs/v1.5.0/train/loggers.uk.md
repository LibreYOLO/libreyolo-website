---
title: Логери експериментів
seo_title: Логери експериментів і зворотні виклики в LibreYOLO
description: >-
  Надсилайте метрики навчання до TensorBoard, MLflow, Weights & Biases, Comet,
  ClearML, Neptune або DVCLive та створюйте власні зворотні виклики для чотирьох
  подій навчання.
lead: >-
  Кожне навчуване сімейство породжує чотири події навчання. Вбудовані логери є
  об'єктами зворотного виклику, що слухають ті самі події, тому інтеграція із
  сервісом і власний hook використовують один інтерфейс.
keywords:
  - tensorboard навчання
  - mlflow tracking
  - weights and biases
  - clearml
  - comet ml
  - neptune
  - dvclive
  - callbacks навчання
  - метрики навчання csv
  - libreyolo monitor
last_verified: 1.5.0
snippets:
  logger:
    - label: За назвою
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(data="coco8.yaml", epochs=10, loggers="tensorboard")
    - label: Налаштований екземпляр
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
    - label: Звичайна функція
      language: python
      code: |
        from libreyolo import LibreYOLO
        from libreyolo.training import TrainEpochEvent


        def on_epoch(event: TrainEpochEvent) -> None:
            print(f"epoch {event.epoch}/{event.total_epochs} loss={event.train_loss:.4f}")


        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(data="coco8.yaml", epochs=10, callbacks=on_epoch)
    - label: Об'єкт із кількома hooks
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
    - label: Спостерігати за запуском у браузері
      language: bash
      code: |
        libreyolo monitor                     # останній запуск у runs/
        libreyolo monitor runs/train/exp      # конкретний запуск
source_hash: de035acbaed32804
---

## Увімкнення логера

`loggers=` приймає зареєстровану назву, налаштований екземпляр або ітерабельний
об'єкт, що поєднує обидва типи.

<code-tabs name="logger" />

Назви не залежать від регістру. Зареєстрований набір містить `tensorboard`, `mlflow`,
`wandb`, `comet`, `clearml`, `neptune`, `dvclive` і `dvc`; останній є псевдонімом
`dvclive`. Будь-яке інше значення одразу породжує помилку зі списком дійсних назв.
Значення для ввімкнення всіх логерів немає, як і прапорця CLI: `loggers=` є
аргументом Python.

## Дані, які записує кожен сервіс

Усі сервіси записують однакові назви метрик, тому панель має однаковий вигляд
незалежно від вибору:

| Ключ | Значення |
|---|---|
| `train/loss` | середня функція втрат навчання за епоху |
| `train/loss/<component>` | кожен компонент функції втрат, про який повідомляє сімейство |
| `lr/<group>` | швидкість навчання кожної групи параметрів оптимізатора |
| `val/<metric>` | кожна метрика валідації без префікса `metrics/` |
| `time/epoch_seconds` | реальний час епохи |

Кроком є номер епохи, починаючи з 1. Повністю визначена конфігурація навчання
записується як параметри на початку, а назва запуску типово має форму
`<family><size>-<task>`, наприклад `yolo9s-detect`.

Після завершення навчання сервіси з підтримкою артефактів вивантажують `results.csv`,
`train_config.yaml` і `summary.json`, якщо вони існують, а за `log_checkpoints=True`
також `weights/best.pt`. TensorBoard нічого не вивантажує, бо не має поняття артефакту.
Жоден логер не вивантажує зображення графіків валідації.

## Поведінка в разі помилки

Відсутній пакет сервісу породжує помилку під час створення із зазначенням команди
встановлення, оскільки явний запит логера без отримання результату приховує помилку.

Помилка сервісу під час запуску обробляється інакше. Перший виняток з обробника
вимикає цей логер до кінця запуску, записується в лог, завершує запуск сервісу
зі станом помилки, а навчання продовжується. Недоступність сервера відстеження
не призводить до втрати навчання.

## Сервіси

Для кожного потрібне власне доповнення.

| Назва | Доповнення | Конструктор |
|---|---|---|
| `tensorboard` | `libreyolo[tensorboard]` | `TensorBoardLogger(log_dir=None)` |
| `mlflow` | `libreyolo[mlflow]` | `MLflowLogger(tracking_uri, experiment_name, run_name, log_artifacts=True, log_checkpoints=False)` |
| `wandb` | `libreyolo[wandb]` | `WandbLogger(project, name, entity, log_checkpoints=False)` |
| `comet` | `libreyolo[comet]` | `CometLogger(project_name, workspace, name, api_key, online, log_artifacts=True, log_checkpoints=False)` |
| `clearml` | `libreyolo[clearml]` | `ClearMLLogger(project_name="LibreYOLO", task_name, tags, output_uri, log_artifacts=True, log_checkpoints=False)` |
| `neptune` | `libreyolo[neptune]` | `NeptuneLogger(project, api_token, name, run_id, tags, mode, capture_console=False, log_artifacts=True, log_checkpoints=False)` |
| `dvclive`, `dvc` | `libreyolo[dvclive]` | `DVCLiveLogger(log_dir, resume, report, save_dvc_exp=False, dvcyaml=None, monitor_system=False, log_checkpoints=False)` |

Імпортуйте класи з `libreyolo.training`.

Перед першим запуском варто знати особливості окремих сервісів.

Файли подій TensorBoard типово записуються до `<save_dir>/tensorboard`. Переглядайте
їх командою `tensorboard --logdir runs/train`.

MLflow 3.x позначив локальне файлове сховище `./mlruns` застарілим і породжує помилку
без `MLFLOW_ALLOW_FILE_STORE=true`. Для локального відстеження без сервера передайте
натомість URI бази даних, як у фрагменті вище, і читайте його командою
`mlflow ui --backend-store-uri sqlite:///mlflow.db`.

Weights & Biases спочатку звертається до змінної середовища `WANDB_PROJECT`, а потім
до `libreyolo`. Comet спочатку звертається до `COMET_PROJECT_NAME`, а потім до
`libreyolo`, і отримує облікові дані з власної конфігурації; `online=False` створює
офлайн-експеримент. ClearML створює нове завдання, записує конфігурацію в `TrainConfig`
і вимикає автоматичне захоплення фреймворку, щоб метрики не надсилалися двічі.
Neptune використовує поточний клієнт `neptune-scale`, а не застарілий пакет,
а `mode="offline"` записує дані локально.

DVCLive записує до `<save_dir>/dvclive`. Він будує дерево підсумків від `/` і не може
зберігати число з рухомою крапкою за шляхом, який водночас є батьківським, тому
`train/loss/box` записується як `train/loss.box`, а `train/loss` зберігає свою назву.
LibreYOLO також вимикає типові для DVCLive збереження експерименту DVC і запис
кореневого `dvc.yaml`, тому добровільно ввімкнений логер не створює стану керування
версіями поза каталогом запуску. Передайте `save_dvc_exp=True` або явний `dvcyaml=`,
щоб повернути цю поведінку.

Neptune навмисно не входить до `libreyolo[all]`: його стабільний клієнт потребує
версію protobuf нижче 7, а доповнення TFLite потребує protobuf 7. Установлюйте
`libreyolo[neptune]` у середовищі без доповнення TFLite.

## Створення зворотного виклику

Усі засоби використовують ті самі чотири події.

<code-tabs name="callback" />

| Подія | Коли | Містить |
|---|---|---|
| `TrainStartEvent` | після налаштування, до епохи 1 | `start_epoch`, `total_epochs`, `model_family`, `model_size`, `task`, `save_dir`, `config` |
| `TrainEpochEvent` | після кожної епохи, навчання й валідації | `epoch`, `train_loss`, `train_loss_items`, `lr`, `val_metrics`, `validated`, `is_best`, `current_metric`, `best_metric`, `best_epoch`, `epoch_seconds` |
| `TrainEndEvent` | після завершення навчання | `completed_epochs`, `final_loss`, `best_metric`, `best_epoch`, `total_seconds`, `results` |
| `TrainExceptionEvent` | якщо навчання породжує виняток | `epoch`, `exception`, `exception_type`, `exception_message`, `elapsed_seconds` |

Звичайний callable отримує лише `TrainEpochEvent`. Об'єкт може реалізувати будь-яку
підмножину `on_train_start`, `on_train_epoch_end`, `on_train_end` і
`on_train_exception`; відсутні методи пропускаються.

`TrainStartEvent.config` є повністю визначеною конфігурацією, у якій kwargs користувача
об'єднано з типовими значеннями сімейства, у формі відображення лише для читання.
Події є замороженими dataclass, а їхні відображення доступні лише для читання,
тому зворотний виклик не може змінити запуск записом до них.

Виняток з `on_train_start`, `on_train_epoch_end` або `on_train_end` поширюється
й завершує запуск. Захищено лише `on_train_exception`, тому він не може приховати
початкову помилку.

Під час навчання на кількох GPU зворотні виклики спрацьовують лише на ранзі 0.
За автоматичного запуску DDP вони також мають підтримувати pickle, тобто бути класом
або функцією рівня модуля, а не замиканням чи lambda. Див. розділ
[Навчання на кількох GPU](/docs/train/multi-gpu).

## Файли, які записує кожен запуск

Для кожного сімейства три файли потрапляють до каталогу запуску без жодних налаштувань:

| Файл | Коли записується | Вміст |
|---|---|---|
| `status.json` | атомарно, після кожної епохи, під час початку, завершення й помилки | `state` зі значенням `running`, `completed` або `failed`, `current_epoch`, `total_epochs`, `progress`, `eta_seconds`, останні `metrics`, `best_metric`, `best_epoch` та об'єкт `error` у разі помилки |
| `metrics.jsonl` | доповнюється один раз за епоху | один рядок JSON на епоху з тією самою схемою, що й `results.csv` |
| `train.log` | у реальному часі | консольний вивід запуску |

`status.json` придатний для дешевого читання скриптом або агентом, що опитує запуск,
а атомарний запис гарантує, що читач ніколи не побачить частково записаного файлу.

`results.csv` і `summary.json` є окремими файлами, що залежать від сімейства.
Вони записуються для YOLOv9, YOLOv9-E2E, YOLOv9-P2, YOLOv7, YOLO-NAS, RF-DETR,
EC і DINOv2, але не для інших сімейств. `results.csv` отримує один рядок на епоху
з компонентами функції втрат, метриками валідації та швидкостями навчання у стовпцях,
а його заголовок розширюється з появою нового стовпця. Під час відновлення файл
обрізається до рядків перед відновленою епохою, щоб не дублювати їх.

Крім того, засіб навчання завжди записує `train_config.yaml` під час налаштування
та контрольні точки до `weights/`.

## Спостереження за активним запуском

<code-tabs name="monitor" />

`libreyolo monitor` надає браузерну панель для перелічених вище файлів лише засобами
стандартної бібліотеки: графіки метрик, останні рядки логу й усі зображення валідації
з оновленням під час активного запуску. Панель працює лише для читання й ніколи
не взаємодіє з процесом навчання, тому може під'єднатися до активного запуску,
повторно відкрити завершений або дослідити аварійний.

## Пов'язані матеріали

- [Валідація та метрики](/docs/train/validation) пояснює значення ключів `val/`
  і додавання функції втрат валідації.
- [Продуктивність навчання](/docs/train/performance) описує профайлер, інший засіб
  для іншого запитання.
