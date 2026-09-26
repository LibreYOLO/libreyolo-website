---
title: D-FINE
families:
  - dfine
seo_title: 'D-FINE: дообучение, валидация и экспорт под MIT'
description: >-
  Использование D-FINE в LibreYOLO для детекции объектов и сегментации
  экземпляров. Установка, предсказание, дообучение, валидация и экспорт с кодом
  под лицензией MIT.
lead: >-
  Трансформер для детекции, который переформулирует регрессию рамки как
  распределение вероятностей по каждой стороне рамки, уточняемое по слоям
  декодера. LibreYOLO поддерживает его для детекции и сегментации экземпляров.
keywords:
  - D-FINE
  - DETR
  - трансформер для детекции объектов
  - детекция объектов в реальном времени
  - сегментация экземпляров python
  - дообучить d-fine на своём датасете
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDFINEn.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreDFINEn.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Сегментация экземпляров
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Суффикс -seg в имени файла выбирает голову масок, поэтому аргумент
        # task здесь не нужен.
        model = LibreYOLO("LibreDFINEn-seg.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.masks.data.shape)
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreDFINEn.pt")

        model.train(data="my-dataset.yaml", epochs=50, imgsz=640, batch=8,
        lr0=2e-4)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreDFINEn.pt data=my-dataset.yaml \
          epochs=50 imgsz=640 batch=8 lr0=2e-4
    - label: Сегментация экземпляров
      language: bash
      code: |
        # Продолжает с опубликованных весов сегментации, вместе с головой масок.
        libreyolo train model=LibreDFINEn-seg.pt data=my-dataset.yaml \
          task=segment epochs=50 imgsz=640
    - label: Сегментация из весов детекции
      language: bash
      code: |
        # В весах детекции головы масок нет, поэтому это явный перенос:
        # голова стартует необученной и полезна только после обучения. Именно
        # запрос task=segment здесь разрешает такой перенос.
        libreyolo train model=LibreDFINEn.pt data=my-dataset.yaml \
          task=segment epochs=50 imgsz=640
    - label: LoRA
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDFINEn.pt")
        model.train(data="my-dataset.yaml", epochs=50, lora=True)
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreDFINEn.pt data=my-dataset.yaml \
          epochs=50 device=0,1 batch=16
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDFINEn.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDFINEn.pt data=my-dataset.yaml
    - label: Сегментация экземпляров
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDFINEn-seg.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95(M)"])   # маски
        print(metrics["metrics/mAP50-95(B)"])   # рамки
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDFINEn.pt")
        model.export(format="onnx", imgsz=640)
        model.export(format="tensorrt", imgsz=640, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreDFINEn.pt format=onnx imgsz=640

        libreyolo export model=LibreDFINEn.pt format=tensorrt imgsz=640
        half=True
    - label: Использование экспортированного файла
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Фабрика выбирает загрузчик по расширению файла, поэтому
        # экспортированный артефакт загружается как любой чекпойнт и
        # возвращает тот же объект Results.
        model = LibreYOLO("LibreDFINEn.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 0216631a26185524
---

## Установка

D-FINE не требует опциональных extra. Всё, что он импортирует, входит в базовую
установку.

```bash
pip install libreyolo
```

Исключение — дообучение адаптерами с `lora=True`: ему нужен extra `lora`.

```bash
pip install "libreyolo[lora]"
```

## Предсказание

Веса скачиваются с Hugging Face при первом запуске и кэшируются локально.

<code-tabs name="predict" />

Возвращаемый объект `Results` — тот же, что возвращает любое семейство, поэтому
замена на другой детектор занимает одну строку. Имя файла с `-seg` само по себе
выбирает задачу сегментации, и тогда `result.masks` несёт маски экземпляров
рядом с рамками. `conf` и `max_det` фильтруют отбор запросов; `iou` принимается
ради единообразия API, но ни на что не влияет, потому что декодер предсказывает
сразу весь набор объектов и шага NMS в нём нет. Про источники, стриминг и
обработку результатов см. [предсказание](/docs/predict).

## Варианты

Пять размеров. Все они работают на одном и том же входном разрешении, поэтому
таблица разделяет их по числу параметров и точности.

<benchmark-table task="detect" />

<va-embed />

Сегментация переиспользует бэкбон, энкодер и декодер детекции и добавляет голову
масок, поэтому `-seg`-чекпойнт принимает те же аргументы, что и его детекционный
аналог. Семейство RT-DETRv4 в LibreYOLO написано как подкласс обёртки D-FINE:
оно наследует эту линию декодера, а затем фиксирует список задач обратно на
детекцию, потому что головы масок у него нет.

## Обучение

Обучение начинается с опубликованного чекпойнта — для обеих задач.

<code-tabs name="train" />

Если ничего не менять, обучение идёт 132 эпохи с `lr0=2e-4` и `amp=False`,
батчем 16 и ранней остановкой после 50 эпох без улучшения. Веса детекции —
допустимая стартовая точка для обучения сегментации, но только как явный
перенос: голова масок стартует необученной и иначе возвращала бы бессмысленные
маски. Разрешает это передача `task=segment` в CLI. В Python путь уже:
`LibreDFINE` приходится создавать напрямую с
`allow_detect_to_segment_transfer=True`, потому что фабрика `LibreYOLO()`
такого аргумента не принимает, а прямое создание ничего не скачивает, так что
файл весов должен уже лежать на диске.

`lora=True` относится к детекции. Обучение сегментации его отклоняет и указывает
вместо него на `freeze='backbone'`, потому что голова масок с адаптерами не
тестировалась. На Apple silicon весь запуск обучения переносится на CPU:
обратный проход по бинированному matmul в Integral упирается в ошибку
компиляции Metal. На инференс на MPS это не влияет.

Про датасеты, аугментацию, multi-GPU и логгеры см. [обучение](/docs/train).

## Валидация

`val()` возвращает словарь с ключами по именам метрик и печатает результаты по
классам, если `verbose` оставлен включённым.

<code-tabs name="val" />

Для `-seg`-чекпойнта обычный ключ `metrics/mAP50-95` содержит оценку по маскам,
и тот же запуск дополнительно сообщает рамки под `(B)` и маски под `(M)`, так
что оба значения доступны за один проход.

## Экспорт

<export-matrix />

Экспортированный артефакт загружается обратно через `LibreYOLO()` по расширению
файла, поэтому файл `.onnx` или `.engine` ведёт себя как чекпойнт и возвращает
тот же `Results`. Пути экспорта в OpenVINO, Paddle, MNN и Core AI работают с
фиксированным холстом, а не с динамическими формами. [Экспорт](/docs/export)
перечисляет аргументы, которые принимает каждый формат, и те дополнительные,
что добавляют некоторые из них.

<code-tabs name="export" />

## Чекпойнты

Все опубликованные файлы весов этого семейства.

<checkpoint-table />

## Лицензирование

<provenance-box>

У весов сегментации есть второй апстрим: их декодер масок, сопоставление масок и
функция потерь по маскам взяты из ArgoHA/D-FINE-seg, тоже под Apache-2.0, чей
мейнтейнер разрешил повторное использование с указанием авторства.

</provenance-box>

## Цитирование

<citation-block />
