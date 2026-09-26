---
title: Быстрый старт
seo_title: Быстрый старт LibreYOLO
description: >-
  Запустить детектор на изображении, дообучить его на небольшом датасете и
  экспортировать в TorchScript или ONNX — всё на CPU и примерно за десять строк
  Python.
lead: >-
  Самый короткий путь через LibreYOLO: предсказание на одном изображении,
  обучение на небольшом датасете, затем экспорт результата. Все команды на этой
  странице работают на CPU.
keywords:
  - libreyolo быстрый старт
  - libreyolo туториал
  - libreyolo predict
  - обучить libreyolo на своём датасете
  - экспорт libreyolo в onnx
  - пример yolo на python
last_verified: 1.5.0
meta:
  - label: Установка
    value: pip install libreyolo
    mono: true
  - label: Чекпойнт
    value: LibreYOLO9t.pt
    mono: true
  - label: Оборудование
    value: Для всего на этой странице хватит CPU
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Скачивает чекпойнт при первом запуске и кэширует его в weights/.
        model = LibreYOLO("LibreYOLO9t.pt")

        # Одно изображение возвращает один объект Results.
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(result.names[int(box.cls)], float(box.conf), box.xyxy.tolist())
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=yolo9-t save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Видео и потоки
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO9t.pt")


        # stream=True отдаёт по одному Results на кадр, а не собирает весь
        список.

        # Вместо пути можно указать индекс веб-камеры, RTSP-URL или каталог.

        for result in model.predict("clip.mp4", stream=True, save=True):
            print(len(result.boxes))
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # coco8 — датасет из 8 изображений в комплекте с библиотекой. Он
        # скачивается по URL при первом запуске — запускать ничего не нужно.
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
    - label: Валидация
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # val() возвращает обычный dict, а не объект.
        metrics = model.val(data="coco8.yaml", device="cpu")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
  export:
    - label: TorchScript
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9t.pt")

        # export() возвращает путь, по которому записал файл.
        path = model.export(format="torchscript")
        print(path)

        # Фабрика определяет формат по суффиксу файла, поэтому артефакт
        # загружается обратно как чекпойнт и возвращает тот же объект Results.
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

## Установка

```bash
pip install libreyolo
```

Этого достаточно для разделов про предсказание и обучение ниже. Для экспорта в
ONNX добавляется одна дополнительная зависимость; полный список — в разделе
[установка](/docs/install).

## Предсказание

<code-tabs name="predict" />

`LibreYOLO()` — фабрика. Она читает файл, определяет, какому семейству
принадлежат веса, и возвращает модель этого семейства, поэтому переход на другой
детектор — правка в одну строку. Если передать `LibreYOLO9t.pt` без каталога,
файл ищется как `weights/LibreYOLO9t.pt` относительно рабочего каталога и
скачивается туда, когда его нет. Правила скачивания и работа без сети описаны в
разделе [чекпойнты и веса](/docs/weights).

`save=True` записывает аннотированную копию в `runs/detect/`, в каталог
`predict`, номер которого увеличивается с каждым запуском. Возвращаемый
`Results` содержит `boxes`, а `names` сопоставляет индекс класса с его меткой.
Путь к одному изображению возвращает один `Results`; каталог, список изображений
или `stream=True` возвращают список или генератор из них.

## Обучение

<code-tabs name="train" />

`data` — YAML датасета. `coco8.yaml` идёт вместе с библиотекой, поэтому сниппет
работает сразу после вставки; имя, которого нет в комплекте, читается как путь.
Датасеты ищутся в `~/datasets` или в `LIBREYOLO_DATASETS_DIR`, когда эта
переменная задана.

Запуск пишет в `project/name`, по умолчанию — в каталог внутри `runs/train`, а
внутри него лежат `weights/best.pt` и `weights/last.pt`. `train()` возвращает
словарь, в котором есть `save_dir`, `best_checkpoint`, `last_checkpoint`,
значения функции потерь по эпохам и метрики валидации по эпохам. Обученный
чекпойнт загружается через `LibreYOLO()` точно так же, как предобученный.

Обучать можно не каждое семейство. Там, где семейство поставляется только для
инференса, `train()` выбрасывает `NotImplementedError` и сообщает об этом.
[Основные понятия](/docs/concepts) объясняют, что означает каждый уровень
поддержки.

## Экспорт

<code-tabs name="export" />

TorchScript не требует ничего сверх базовой установки. У остальных целей есть
своя дополнительная зависимость, а покрытие зависит от семейства и задачи и
одинаково не для всех: см. [экспорт и развёртывание](/docs/export).

Аргументы, которые принимают все форматы: `imgsz` (int или пара из высоты и
ширины), `batch` (по умолчанию 1), `half`, `int8` с YAML-файлом `data` для
калибровки, `dynamic` (по умолчанию True), `simplify` (по умолчанию True),
`opset`, `device` и `output_path`. Когда `output_path` не задан, файл пишется в
`weights/` с именем, производным от чекпойнта.

## Что дальше

- [Основные понятия](/docs/concepts) — задачи, семейства, размеры и имена чекпойнтов.
- [Чекпойнты и веса](/docs/weights) — автоскачивание, работа без сети и безопасность загрузки.
- [Импорт существующих весов](/docs/migrate), если у вас уже есть чекпойнт из upstream-проекта.
- [Все модели](/docs/models) — семейство, которое подходит под вашу задачу.
- [Обучение](/docs/train), [Предсказание](/docs/predict) и [Экспорт](/docs/export) — полные сценарии работы.
