---
title: Швидкий старт
seo_title: Швидкий старт із LibreYOLO
description: >-
  Запустіть детектор на зображенні, донавчіть його на малому датасеті та
  експортуйте в TorchScript або ONNX приблизно десятьма рядками Python, усе на
  CPU.
lead: >-
  Найкоротший шлях через LibreYOLO: передбачення для одного зображення, навчання
  на малому датасеті й експорт результату. Кожна команда тут працює на CPU.
keywords:
  - швидкий старт LibreYOLO
  - посібник LibreYOLO
  - передбачення LibreYOLO
  - навчання LibreYOLO
  - експорт LibreYOLO
  - приклад YOLO Python
last_verified: 1.5.0
meta:
  - label: Встановлення
    value: pip install libreyolo
    mono: true
  - label: Контрольна точка
    value: LibreYOLO9t.pt
    mono: true
  - label: Обладнання
    value: Для всього на цій сторінці достатньо CPU
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Під час першого використання завантажує контрольну точку, потім кешує
        її у weights/.

        model = LibreYOLO("LibreYOLO9t.pt")


        # Одне зображення повертає один об'єкт Results.

        result = model(SAMPLE_IMAGE, save=True)


        for box in result.boxes:
            print(result.names[int(box.cls)], float(box.conf), box.xyxy.tolist())
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=yolo9-t save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Відео й потоки
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # stream=True видає по одному Results на кадр замість побудови списку.
        # Замініть шлях індексом вебкамери, URL RTSP або каталогом.
        for result in model.predict("clip.mp4", stream=True, save=True):
            print(len(result.boxes))
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO9t.pt")


        # coco8, це датасет із 8 зображень у комплекті бібліотеки. Під час
        першого

        # використання він завантажується з URL, тому виконувати скрипт не
        потрібно.

        results = model.train(
            data="coco8.yaml",
            epochs=1,
            imgsz=640,
            batch=4,
            device="cpu",
        )


        print(results["save_dir"])

        print(results["best_checkpoint"])
    - label: CLI
      language: bash
      code: |
        libreyolo train model=yolo9-t data=coco8.yaml \
          epochs=1 imgsz=640 batch=4 device=cpu
    - label: Валідація
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # val() повертає звичайний dict, а не об'єкт.
        metrics = model.val(data="coco8.yaml", device="cpu")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
  export:
    - label: TorchScript
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreYOLO9t.pt")


        # export() повертає записаний шлях.

        path = model.export(format="torchscript")

        print(path)


        # Фабрика вибирає маршрут за суфіксом файла, тому артефакт
        завантажується

        # як контрольна точка й повертає той самий об'єкт Results.

        exported = LibreYOLO(path)

        result = exported(SAMPLE_IMAGE)

        print(len(result.boxes))
    - label: ONNX
      language: bash
      code: |
        pip install "libreyolo[onnx]"
        libreyolo export model=yolo9-t format=onnx imgsz=640
source_hash: c11b6bdbf0b6fdf1
---

## Встановлення

```bash
pip install libreyolo
```

Це все, що потрібно для наведених нижче розділів передбачення та навчання.
Експорт до ONNX додає одну залежність; повний список наведено в розділі
[встановлення](/docs/install).

## Передбачення

<code-tabs name="predict" />

`LibreYOLO()`, це фабрика. Вона читає файл, визначає сімейство ваг і
повертає модель цього сімейства, тому заміна детектора потребує зміни одного
рядка. Передавання `LibreYOLO9t.pt` без каталогу виконує пошук у
`weights/LibreYOLO9t.pt` відносно робочого каталогу й завантажує файл туди
за відсутності. Правила завантаження й автономну роботу описано в розділі
[контрольні точки й ваги](/docs/weights).

`save=True` записує анотовану копію в `runs/detect/`, до каталогу
`predict`, номер якого збільшується для кожного запуску. Повернений
`Results` містить `boxes`, а `names` зіставляє індекс класу з міткою.
Шлях до одного зображення повертає один `Results`; каталог, список
зображень або `stream=True` повертає їх список чи генератор.

## Навчання

<code-tabs name="train" />

`data`, це YAML датасету. `coco8.yaml` постачається з бібліотекою, тому
фрагмент працює без змін; ім'я, якого немає в комплекті, читається як шлях.
Датасети визначаються в `~/datasets` або в `LIBREYOLO_DATASETS_DIR`,
якщо цю змінну задано.

Запуск записує дані до `project/name`, стандартно до каталогу під
`runs/train`, з `weights/best.pt` і `weights/last.pt` усередині.
`train()` повертає словник із `save_dir`, `best_checkpoint`,
`last_checkpoint`, втратами кожної епохи й метриками валідації кожної
епохи. Навчена контрольна точка завантажується через `LibreYOLO()` так само,
як попередньо навчена.

Не кожне сімейство підтримує навчання. Якщо сімейство постачає лише інференс,
`train()` спричиняє `NotImplementedError` із поясненням.
[Основні поняття](/docs/concepts) пояснюють значення рівнів підтримки.

## Експорт

<code-tabs name="export" />

TorchScript не потребує нічого понад базове встановлення. Інші цілі мають
власні додаткові залежності, а підтримка визначається окремо для сімейства та
задачі: дивіться [експорт і розгортання](/docs/export).

Спільні для всіх форматів аргументи містять `imgsz` (ціле число або пара
висоти й ширини), `batch` (стандартно 1), `half`, `int8` із YAML
`data` для калібрування, `dynamic` (стандартно True), `simplify`
(стандартно True), `opset`, `device` і `output_path`. Якщо
`output_path` не задано, файл записується до `weights/` з ім'ям,
утвореним від контрольної точки.

## Наступні кроки

- [Основні поняття](/docs/concepts) для задач, сімейств, розмірів та імен
  контрольних точок.
- [Контрольні точки й ваги](/docs/weights) для автоматичного завантаження,
  автономної роботи й безпеки завантаження.
- [Імпорт наявних ваг](/docs/migrate), якщо у вас уже є контрольна точка з
  проєкту upstream.
- [Усі моделі](/docs/models), щоб вибрати сімейство для своєї задачі.
- [Навчання](/docs/train), [передбачення](/docs/predict) і
  [експорт](/docs/export) для повних робочих процесів.

