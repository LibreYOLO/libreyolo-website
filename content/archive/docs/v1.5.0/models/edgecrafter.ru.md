---
title: EdgeCrafter
families:
  - ec
seo_title: 'EdgeCrafter: детекция, оценка позы и сегментация в LibreYOLO'
description: >-
  Использование EdgeCrafter в LibreYOLO для детекции, оценки позы и сегментации
  экземпляров. Установка, предсказание, валидация и экспорт с кодом под
  лицензией MIT.
lead: >-
  Компактный визуальный трансформер для плотного предсказания на
  edge-устройствах, опубликованный в апстриме как три родственные модели: ECDet,
  ECPose и ECSeg. LibreYOLO загружает все три как одно семейство, а задачу несёт
  чекпойнт.
keywords:
  - EdgeCrafter
  - ECDet
  - ECPose
  - ECSeg
  - компактный визуальный трансформер
  - детекция объектов python
  - оценка позы человека
  - сегментация экземпляров python
  - инференс на edge-устройствах
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreECs.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreECs.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Оценка позы
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Суффикс -pose в имени файла выбирает голову ключевых точек,
        # поэтому аргумент task здесь не нужен.
        model = LibreYOLO("LibreECs-pose.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.keypoints.xy)
        print(result.boxes.conf)
    - label: Сегментация экземпляров
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreECs-seg.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.masks.data.shape)
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=50,
            imgsz=640,
            batch=8,
            lr0=5e-4,
        )
    - label: CLI
      language: bash
      code: >
        libreyolo train model=LibreECs.pt data=my-dataset.yaml epochs=50
        imgsz=640 batch=8 lr0=5e-4
    - label: Оценка позы
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Нужен одноклассовый датасет с ключевыми точками, в data.yaml
        # которого объявлен kpt_shape, и imgsz, равный родному размеру
        # чекпойнта.
        model = LibreYOLO("LibreECs-pose.pt")
        model.train(
            data="my-pose-dataset.yaml",
            epochs=50,
            imgsz=640,
        )
    - label: Сегментация экземпляров
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Нужна полигональная разметка и imgsz, равный родному размеру
        чекпойнта.

        model = LibreYOLO("LibreECs-seg.pt")

        model.train(
            data="my-dataset.yaml",
            epochs=50,
            imgsz=640,
        )
    - label: LoRA
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=50,
            lora=True,
        )
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreECs.pt data=my-dataset.yaml
    - label: Оценка позы
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs-pose.pt")
        metrics = model.val(data="my-pose-dataset.yaml")

        print(metrics["metrics/keypoints_mAP50-95"])
        print(metrics["metrics/keypoints_mAP50"])
    - label: Сегментация экземпляров
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs-seg.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95(M)"])   # маски
        print(metrics["metrics/mAP50-95(B)"])   # рамки
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs.pt")
        model.export(format="onnx", imgsz=640)
        model.export(format="tensorrt", imgsz=640, half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreECs.pt format=onnx imgsz=640
        libreyolo export model=LibreECs-pose.pt format=onnx imgsz=640
        libreyolo export model=LibreECs-seg.pt format=onnx imgsz=640
    - label: Использование экспортированного файла
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Фабрика выбирает загрузчик по расширению файла, поэтому
        # экспортированный артефакт загружается как любой чекпойнт и
        # возвращает тот же объект Results.
        model = LibreYOLO("LibreECs.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 39c6975fc16b3ff1
---

## Установка

EdgeCrafter не требует опциональных extra. Всё, что он импортирует, входит в
базовую установку.

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

Задача определяется по имени файла, поэтому чекпойнт с `-pose` или `-seg` сам
выбирает свою голову и не принимает аргумент задачи. Все три возвращают тот же
объект `Results`, что возвращает любое семейство, с добавлением
`result.keypoints` для позы и `result.masks` для сегментации. Оценка позы
охватывает один класс, человека, с 17 ключевыми точками COCO, и их количество
фиксируется при сборке модели. Головы рамок у неё нет, поэтому каждая рамка
позы — это габариты её собственных ключевых точек, а третий канал ключевой
точки — константа, а не оценка по каждой точке.

`conf` и `max_det` фильтруют отбор запросов; `iou` принимается ради
единообразия API, но ни на что не влияет, потому что все три головы декодируют
набор запросов без шага NMS. Про источники, стриминг и обработку результатов
см. [предсказание](/docs/predict).

## Варианты

Четыре размера. Все они работают на одном и том же входном разрешении, поэтому
таблица разделяет их по числу параметров и точности.

<benchmark-table task="detect" />

<va-embed />

Апстрим публикует ECDet, ECPose и ECSeg как три отдельные модели, а не одну
модель с тремя головами. У них общий бэкбон ECViT и общий гибридный энкодер, а
различаются они только головой, поэтому LibreYOLO сводит их в одно семейство и
оставляет задачу за именем файла чекпойнта. Поэтому буква размера означает один
и тот же бэкбон и энкодер во всех трёх, а предсказание, валидация и экспорт
принимают одни и те же аргументы, какую бы из них вы ни загрузили.

## Обучение

Все три задачи обучаются через `train()`: он читает задачу из загруженного
чекпойнта и подбирает подходящий тренер.

<code-tabs name="train" />

Что проверено для детекции и сегментации: совпадение инференса с апстримом с
точностью 1e-5, послойно и по каждому размеру, и что функция потерь и один шаг
обучения отрабатывают на синтетическом входе. Что не проверено, согласно
докстрингу самого `train()`: сходимость полного дообучения, обучение на
нескольких GPU, шаг отключения аугментации с перезагрузкой лучших весов и
переразметка классов из Objects365 в COCO. Путь для позы следует
опубликованному рецепту DETRPose — венгерское сопоставление по стоимостям
класса, L1 по ключевым точкам и OKS с контрастивным шумоподавлением ключевых
точек, — и его сходимость тоже не проверялась от начала до конца.

Если ничего не менять, обучение идёт 74 эпохи с `lr0=5e-4` и включённой
смешанной точностью, следуя рецепту апстрима: AdamW, плоский косинусный
планировщик, EMA с 0.9999 и вход, нормализованный по ImageNet. Для позы и
сегментации требуется `imgsz`, равный родному размеру чекпойнта, потому что их
сетка якорей для оценки строится при создании модели; другое значение вызывает
ошибку ещё до старта запуска. Для позы дополнительно нужен одноклассовый
датасет, в `data.yaml` которого объявлен `kpt_shape`, с числом ключевых точек,
совпадающим с головой.

`lora=True` относится только к детекции; для позы и сегментации он вызывает
`ValueError`. На Apple silicon тренер оставляет запуск на GPU и отправляет на
CPU одну операцию — обратный проход grid-sample внутри deformable attention,
который PyTorch не реализует в Metal.

Про датасеты, аугментацию, multi-GPU и логгеры см. [обучение](/docs/train).

## Валидация

`val()` возвращает словарь с ключами по именам метрик и печатает результаты по
классам, если `verbose` оставлен включённым.

<code-tabs name="val" />

Для позы метрики OKS по ключевым точкам сообщаются под `metrics/keypoints_*`.
Сегментация сообщает маски под обычным ключом `metrics/mAP50-95` и повторяет оба
среза за один проход: рамки под `(B)` и маски под `(M)`.

## Экспорт

<export-matrix />

Экспортированный артефакт загружается обратно через `LibreYOLO()` по расширению
файла, поэтому файл `.onnx` или `.engine` ведёт себя как чекпойнт и возвращает
тот же `Results`. Поза и сегментация экспортируются с фиксированным входом
640 на 640, а не с динамическими формами, и несколько целей детекции тоже
работают с фиксированным холстом, включая OpenVINO, Paddle, MNN, ExecuTorch и
Core AI. [Экспорт](/docs/export) перечисляет аргументы, которые принимает каждый
формат, и те дополнительные, что добавляют некоторые из них.

<code-tabs name="export" />

## Чекпойнты

Все опубликованные файлы весов этого семейства.

<checkpoint-table />

## Лицензирование

<provenance-box></provenance-box>

## Цитирование

<citation-block />
